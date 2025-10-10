"""Tool definitions and utilities."""

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, cast
import subprocess

import jsonschema
from anthropic import BadRequestError
from termcolor import colored
from typing_extensions import final

# Inlined ClaudeTokenCounter class (was in utils.token_counter)
class ClaudeTokenCounter:
    def count_tokens(self, prompt_chars: str) -> int:
        return len(prompt_chars) // 3
from llm_server.compat import (
    AnthropicRedactedThinkingBlock,
    AnthropicThinkingBlock,
    ToolCall,
    ToolFormattedResult,
    AssistantContentBlock,
    GeneralContentBlock,
    LLMMessages,
    TextPrompt,
    TextResult,
    ToolParam,
)

ToolInputSchema = dict[str, Any]
"""A JSON schema describing the input to a tool."""


RIGHT = ""  # "▶"
LEFT = ""  # "◀"


@dataclass
class ToolCallParameters:
    tool_call_id: str
    tool_name: str
    tool_input: Any


@dataclass
class ToolImplOutput:
    """Output from an LLM tool implementation.

    Attributes:
        tool_output: The main output string that will be shown to the model.
        tool_result_message: A description of what the tool did, for logging purposes.
        auxiliary_data: Additional data that the tool wants to pass along for logging only.
    """

    tool_output: str
    tool_result_message: str
    auxiliary_data: dict[str, Any] = field(default_factory=dict)


class ConversationFlow:
    """Keeps track of messages that compose a dialog.

    A dialog alternates between user and assistant turns. Each turn consists
    of one or more messages, represented by GeneralContentBlock.

    A user turn consists of one or more prompts and tool results.
    An assistant turn consists of a model answer and tool calls.
    """

    def __init__(
        self,
        logger_for_agent_logs: logging.Logger,
        use_prompt_budgeting: bool = False,
    ):
        self.logger_for_agent_logs = logger_for_agent_logs
        self._message_lists: list[list[GeneralContentBlock]] = []
        self.token_counter = ClaudeTokenCounter()
        self.use_prompt_budgeting = use_prompt_budgeting
        self.truncation_history_token_cts: list[int] = []
        self.token_budget_to_trigger_truncation = 120_000
        self.truncate_all_but_N = 3

    def add_user_prompt(
        self, message: str, allow_append_to_tool_call_results: bool = False
    ):
        """Add a user prompt to the dialog.

        Args:
            message: The message to add.
            allow_append_to_tool_call_results: If True, and if the last message
                is a tool call result, then the message will be appended to that
                turn.
        """
        if self.is_user_turn():
            self._message_lists.append([TextPrompt(message)])
        else:
            if allow_append_to_tool_call_results:
                user_messages = self._message_lists[-1]
                for user_message in user_messages:
                    if isinstance(user_message, TextPrompt):
                        raise ValueError(
                            f"Last user turn already contains a text prompt: {user_message}"
                        )
                user_messages.append(TextPrompt(message))
            else:
                self._assert_user_turn()

    def add_tool_call_result(self, parameters: ToolCallParameters, result: str):
        """Add the result of a tool call to the dialog."""
        self.add_tool_call_results([parameters], [result])

    def add_tool_call_results(
        self, parameters: list[ToolCallParameters], results: list[str]
    ):
        """Add the result of a tool call to the dialog."""
        self._assert_user_turn()
        self._message_lists.append(
            [
                ToolFormattedResult(
                    tool_call_id=params.tool_call_id,
                    tool_name=params.tool_name,
                    tool_output=result,
                )
                for params, result in zip(parameters, results)
            ]
        )

    def add_model_response(self, response: list[AssistantContentBlock]):
        """Add the result of a model call to the dialog."""
        self._assert_assistant_turn()
        self._message_lists.append(cast(list[GeneralContentBlock], response))

    def count_tokens(self) -> int:
        """Count the total number of tokens in the dialog."""
        total_tokens = 0
        for i, message_list in enumerate(self._message_lists):
            # assert self._message_lists is None, f"self.message_lists is None in {self._message_lists}"
            is_last_message_list = i == len(self._message_lists) - 1
            for message in message_list:
                if isinstance(message, (TextPrompt, TextResult)):
                    assert message.text is not None, f"Message text is None in {message}"
                    total_tokens += self.token_counter.count_tokens(message.text)
                elif isinstance(message, ToolFormattedResult):
                    assert message.tool_output is not None, f"Tool output is None in {message}"
                    total_tokens += self.token_counter.count_tokens(message.tool_output)
                elif isinstance(message, ToolCall):
                    assert message.tool_input is not None, f"Tool input is None in {message}"

                    total_tokens += self.token_counter.count_tokens(
                        json.dumps(message.tool_input)
                    )
                elif isinstance(message, AnthropicRedactedThinkingBlock):
                    total_tokens += 0
                elif isinstance(message, AnthropicThinkingBlock):
                    total_tokens += (
                        self.token_counter.count_tokens(message.thinking)
                        if is_last_message_list
                        else 0
                    )
                else:
                    raise ValueError(f"Unknown message type: {type(message)}")
        return total_tokens

    def run_truncation_strategy(self) -> None:
        """Truncate all the tool results apart from the last N turns."""

        print(
            colored(
                f"Truncating all but the last {self.truncate_all_but_N} turns as we hit the token budget {self.token_budget_to_trigger_truncation}.",
                "yellow",
            )
        )
        self.logger_for_agent_logs.info(
            f"Truncating all but the last {self.truncate_all_but_N} turns as we hit the token budget {self.token_budget_to_trigger_truncation}."
        )

        old_token_ct = self.count_tokens()

        new_message_lists: list[list[GeneralContentBlock]] = copy.deepcopy(
            self._message_lists
        )

        for message_list in new_message_lists[: -self.truncate_all_but_N]:
            for message in message_list:
                if isinstance(message, ToolFormattedResult):
                    message.tool_output = (
                        "[Truncated...re-run tool if you need to see output again.]"
                    )
                elif isinstance(message, ToolCall):
                    if message.tool_name == "sequential_thinking":
                        message.tool_input["thought"] = (
                            "[Truncated...re-run tool if you need to see input/output again.]"
                        )
                    elif message.tool_name == "str_replace_editor":
                        if "file_text" in message.tool_input:
                            message.tool_input["file_text"] = (
                                "[Truncated...re-run tool if you need to see input/output again.]"
                            )
                        if "old_str" in message.tool_input:
                            message.tool_input["old_str"] = (
                                "[Truncated...re-run tool if you need to see input/output again.]"
                            )
                        if "new_str" in message.tool_input:
                            message.tool_input["new_str"] = (
                                "[Truncated...re-run tool if you need to see input/output again.]"
                            )

        self._message_lists = new_message_lists

        new_token_ct = self.count_tokens()
        print(
            colored(
                f" [dialog_messages] Token count after truncation: {new_token_ct}",
                "yellow",
            )
        )

        self.truncation_history_token_cts.append(old_token_ct - new_token_ct)

    def get_messages_for_llm_client(self) -> LLMMessages:
        """Returns messages in the format the LM client expects."""

        if (
            self.use_prompt_budgeting
            and self.count_tokens() > self.token_budget_to_trigger_truncation
        ):
            self.run_truncation_strategy()
        return list(self._message_lists)

    def drop_final_assistant_turn(self):
        """Remove the final assistant turn.

        This allows dialog messages to be passed to tools as they are called,
        without containing the final tool call.
        """
        if self.is_user_turn():
            self._message_lists.pop()

    def drop_tool_calls_from_final_turn(self):
        """Remove tool calls from the final assistant turn.

        This allows dialog messages to be passed to tools as they are called,
        without containing the final tool call.
        """
        if self.is_user_turn():
            new_turn_messages = [
                message
                for message in self._message_lists[-1]
                if not isinstance(message, ToolCall)
            ]
            self._message_lists[-1] = cast(list[GeneralContentBlock], new_turn_messages)

    def get_pending_tool_calls(self) -> list[ToolCallParameters]:
        """Returns the tool calls from the last assistant turn.

        Returns an empty list of no tool calls are pending.
        """
        self._assert_user_turn()
        if len(self._message_lists) == 0:
            return []
        tool_calls = []
        for message in self._message_lists[-1]:
            if isinstance(message, ToolCall):
                tool_calls.append(
                    ToolCallParameters(
                        tool_call_id=message.tool_call_id,
                        tool_name=message.tool_name,
                        tool_input=message.tool_input,
                    )
                )
        return tool_calls

    def get_last_model_text_response(self):
        """Returns the last model response as a string."""
        self._assert_user_turn()
        for message in self._message_lists[-1]:
            if isinstance(message, TextResult):
                return message.text
        raise ValueError("No text response found in last model response")

    def get_last_user_prompt(self) -> str:
        """Returns the last user prompt."""
        self._assert_assistant_turn()
        for message in self._message_lists[-1]:
            if isinstance(message, TextPrompt):
                return message.text
        raise ValueError("No text prompt found in last user prompt")

    def replace_last_user_prompt(self, new_prompt: str):
        """Replace the last user prompt with a new one."""
        self._assert_assistant_turn()
        for i, message in enumerate(self._message_lists[-1]):
            if isinstance(message, TextPrompt):
                self._message_lists[-1][i] = TextPrompt(new_prompt)
                return
        raise ValueError("No text prompt found in last user prompt")

    def clear(self):
        """Delete all messages."""
        self._message_lists = []

    def is_user_turn(self):
        return len(self._message_lists) % 2 == 0

    def is_assistant_turn(self):
        return len(self._message_lists) % 2 == 1

    def __str__(self) -> str:
        json_serializable = [
            [message.to_dict() for message in message_list]
            for message_list in self._message_lists
        ]
        return json.dumps(json_serializable, indent=2)

    def get_summary(self, max_str_len: int = 100) -> str:
        """Returns a summary of the dialog."""

        def truncate_strings(obj):
            # Truncate all leaf strings to 100 characters
            if isinstance(obj, str):
                if len(obj) > max_str_len:
                    return obj[:max_str_len] + "..."
            elif isinstance(obj, dict):
                return {k: truncate_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [truncate_strings(item) for item in obj]
            return obj

        json_serializable = truncate_strings(
            [
                [message.to_dict() for message in message_list]
                for message_list in self._message_lists
            ]
        )
        return json.dumps(json_serializable, indent=2)

    def _assert_user_turn(self):
        assert self.is_user_turn(), "Can only add user prompts on user's turn"

    def _assert_assistant_turn(self):
        assert self.is_assistant_turn(), (
            "Can only get/replace last user prompt on assistant's turn"
        )


class Tool:
    """A tool that can be called by an LLM.

    A general tool may require additional parameters that the model does not
    provide. It may also return arbitrary structured output. Therefore, a
    general tool does not have a well-defined interface for calling it.
    """

    name: str
    description: str
    input_schema: ToolInputSchema


class LLMTool:
    """A tool that fits into the standard LLM tool-calling paradigm.

    An LLM tool can be called by supplying the parameters specified in its
    input_schema, and returns a string that is then shown to the model.
    """

    name: str
    description: str
    input_schema: ToolInputSchema

    @property
    def should_stop(self) -> bool:
        """Whether the tool wants to stop the current agentic run."""
        return False

    # Final is here to indicate that subclasses should override run_impl(), not
    # run(). There may be a reason in the future to override run() itself, and
    # if such a reason comes up, this @final decorator can be removed.
    @final
    def run(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[ConversationFlow] = None,
    ) -> str:
        """Run the tool.

        Args:
            tool_input: The input to the tool.
            dialog_messages: The dialog messages so far, if available. The tool
                is allowed to modify this object, so the caller should make a copy
                if that's not desired. The dialog messages should not contain
                pending tool calls. They should end where it's the user's turn.
        """
        if dialog_messages:
            assert dialog_messages.is_user_turn()

        try:
            self._validate_tool_input(tool_input)
            result = self.run_impl(tool_input, dialog_messages)
            tool_output = result.tool_output
        except jsonschema.ValidationError as exc:
            tool_output = "Invalid tool input: " + exc.message
        except BadRequestError as exc:
            raise RuntimeError("Bad request: " + exc.message)

        return tool_output

    def get_tool_start_message(self, tool_input: ToolInputSchema) -> str:
        """Return a user-friendly message to be shown to the model when the tool is called."""
        return f"Calling tool '{self.name}'"

    def run_impl(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[ConversationFlow] = None,
    ) -> ToolImplOutput:
        """Subclasses should implement this.

        Returns:
            A ToolImplOutput containing the output string, description, and any auxiliary data.
        """
        raise NotImplementedError()

    def get_tool_param(self) -> ToolParam:
        return ToolParam(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def _validate_tool_input(self, tool_input: dict[str, Any]):
        """Validates the tool input.

        Raises:
            jsonschema.ValidationError: If the tool input is invalid.
        """
        jsonschema.validate(instance=tool_input, schema=self.input_schema)



def create_modification(git_repo, reverse=False, container_id=None):
    """Generate the patch for the prediction."""
    import os
    import subprocess
    import time

    logging.info(f"Generating patch in {git_repo}")

    # If container_id is provided, generate patch from inside the container
    if container_id:
        return generate_patch_from_container(container_id, reverse)

    # Check if the git repository path exists
    if not os.path.exists(git_repo):
        logging.error(f"Git repository path does not exist: {git_repo}")
        raise FileNotFoundError(f"Git repository path does not exist: {git_repo}")

    # Check if it's a git repository
    git_dir = os.path.join(git_repo, '.git')
    if not os.path.exists(git_dir):
        logging.error(f"Path is not a git repository: {git_repo}")
        raise ValueError(f"Path is not a git repository: {git_repo}")

    cmd = [
        "git",
        "--no-pager",
        "diff",
        "-U5",  # Include 5 lines of context
        "--no-color",  # Don't include color codes in the output
        "HEAD",  # Compare against the current commit
    ]
    if reverse:
        cmd.append("-R")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            diff = subprocess.check_output(
                cmd,
                cwd=git_repo,
                text=True,
                errors="backslashreplace",
            )
            return diff
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(
                    f"Error {e} occurred. Retrying... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(5)  # Add a small delay before retrying
            else:
                logging.error(
                    f"Failed to decode git diff output after {max_retries} attempts."
                )
                raise


def generate_patch_from_container(container_id, reverse=False):
    """Generate patch from inside a Docker container."""
    import subprocess
    import logging

    logging.info(f"Generating patch from container {container_id}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # First, check what git diff options are available
            check_cmd = [
                "docker", "exec", "-i", container_id,
                "bash", "-c", "cd /testbed && git status --porcelain && echo '---' && git log --oneline -1 2>/dev/null || echo 'NO_COMMITS'"
            ]

            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
            if check_result.returncode == 0:
                check_output = check_result.stdout.strip()
                logging.info(f"Git status check: {check_output}")

                # Determine the best diff strategy
                if 'NO_COMMITS' in check_output:
                    # No commits yet, show all files as new
                    git_cmd_str = "git diff --no-index /dev/null . || git ls-files | head -20"
                    logging.info("No commits found, will show file listing instead of diff")
                else:
                    # Normal diff against HEAD
                    git_cmd_parts = [
                        "git", "--no-pager", "diff", "-U5", "--no-color", "HEAD"
                    ]
                    if reverse:
                        git_cmd_parts.append("-R")
                    git_cmd_str = " ".join(git_cmd_parts)
            else:
                # Fallback to basic diff
                git_cmd_str = "git --no-pager diff -U5 --no-color HEAD"

            # Execute the git command inside the container
            docker_cmd = [
                "docker", "exec", "-i", container_id,
                "bash", "-c", f"cd /testbed && {git_cmd_str}"
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                errors="backslashreplace"
            )

            if result.returncode == 0:
                diff = result.stdout
                logging.info(f"Successfully generated patch from container (length: {len(diff)})")
                return diff if diff is not None else ""
            else:
                error_msg = result.stderr or "Unknown error"
                logging.warning(f"Git diff failed in container (attempt {attempt + 1}): {error_msg}")

                # Special handling for common git errors
                if "unknown revision" in error_msg or "ambiguous argument 'HEAD'" in error_msg:
                    logging.info("HEAD not found, trying alternative diff method...")
                    # Try to get diff of working directory changes
                    alt_cmd = [
                        "docker", "exec", "-i", container_id,
                        "bash", "-c", "cd /testbed && git diff --no-index /dev/null . 2>/dev/null | head -100 || echo 'No changes to show'"
                    ]
                    try:
                        alt_result = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=30)
                        if alt_result.returncode == 0:
                            return alt_result.stdout
                    except Exception as alt_e:
                        logging.warning(f"Alternative diff method failed: {alt_e}")

                if attempt < max_retries - 1:
                    # Try to check git status for debugging
                    status_cmd = ["docker", "exec", "-i", container_id, "bash", "-c", "cd /testbed && git status && git log --oneline -3 2>/dev/null || echo 'No commits'"]
                    try:
                        status_result = subprocess.run(status_cmd, capture_output=True, text=True, timeout=30)
                        logging.info(f"Git status in container: {status_result.stdout}")
                    except Exception as status_e:
                        logging.warning(f"Failed to get git status: {status_e}")

                    time.sleep(3)  # Wait before retry
                else:
                    # On final attempt, return empty diff instead of raising exception
                    logging.warning("All attempts failed, returning empty diff")
                    return ""

        except subprocess.TimeoutExpired:
            logging.warning(f"Git diff timeout in container (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return ""
            time.sleep(3)
        except Exception as e:
            logging.warning(f"Error generating patch from container (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return ""
            time.sleep(3)

    return ""
