"""
🔧 Advanced Patch Generation Agent

A sophisticated multi-component agent system designed for intelligent patch generation
and code modification tasks. Features modular architecture with specialized components
for task orchestration, tool management, and execution context handling.

Architecture Components:
- PatchEngine: Core patch generation and coordination engine
- TaskOrchestrator: Intelligent task decomposition and workflow management
- ToolRegistry: Dynamic tool registration and lifecycle management
- ExecutionContext: Environment state and resource management
- ConversationHandler: Multi-turn dialog processing and token management
- ResultProcessor: Output formatting and result aggregation
"""

import logging
from copy import deepcopy
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from termcolor import colored

from llm_server.compat import LLMClient, TextResult
from prompts.system_prompt import SYSTEM_PROMPT
from tools.bash_tool import create_bash_tool, create_docker_bash_tool
from tools.complete_tool import CompleteTool
from tools.sequential_thinking_tool import SequentialThinkingTool
from tools.str_replace_tool import StrReplaceEditorTool
from utils.common import ConversationFlow, LLMTool, ToolImplOutput
from utils.workspace_manager import ProjectSpaceHandler


class ExecutionPhase(Enum):
    """Execution phase enumeration for state tracking."""
    INITIALIZATION = "initialization"
    TASK_ANALYSIS = "task_analysis"
    TOOL_PREPARATION = "tool_preparation"
    EXECUTION = "execution"
    RESULT_PROCESSING = "result_processing"
    COMPLETION = "completion"


@dataclass
class ExecutionMetrics:
    """Metrics tracking for execution performance."""
    total_turns: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    token_usage: int = 0
    execution_time: float = 0.0
    phase_transitions: List[str] = field(default_factory=list)


@dataclass
class TaskContext:
    """Context container for task execution state."""
    primary_instruction: str
    current_phase: ExecutionPhase = ExecutionPhase.INITIALIZATION
    remaining_iterations: int = 10
    is_interrupted: bool = False
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    🛠️ Dynamic Tool Registration and Management System

    Manages the lifecycle of specialized tools with dynamic registration,
    validation, and execution coordination capabilities.
    """

    def __init__(self, workspace_manager: ProjectSpaceHandler):
        self._registered_tools = {}
        self._tool_instances = []
        self._workspace_manager = workspace_manager
        self._completion_tracker = CompleteTool()

    def register_execution_environment(self, container_id: Optional[str],
                                     require_permission: bool) -> None:
        """Register execution environment (Docker or local)."""
        if container_id:
            execution_tool = create_docker_bash_tool(
                container=container_id,
                ask_user_permission=require_permission
            )
            self._log_environment_setup("Docker", container_id)
        else:
            execution_tool = create_bash_tool(ask_user_permission=require_permission)
            self._log_environment_setup("Local", "host")

        self._tool_instances.append(execution_tool)

    def register_core_tools(self) -> None:
        """Register essential tools for patch generation."""
        core_tools = [
            StrReplaceEditorTool(workspace_manager=self._workspace_manager),
            SequentialThinkingTool(),
            self._completion_tracker
        ]
        self._tool_instances.extend(core_tools)

    def get_tool_parameters(self) -> List:
        """Generate tool parameters for LLM integration."""
        tool_params = [tool.get_tool_param() for tool in self._tool_instances]
        self._validate_tool_uniqueness(tool_params)
        return tool_params

    def find_tool_by_name(self, tool_name: str):
        """Locate tool instance by name."""
        for tool in self._tool_instances:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"🚨 Tool '{tool_name}' not found in registry")

    def get_completion_tracker(self) -> CompleteTool:
        """Access completion tracking tool."""
        return self._completion_tracker

    def _validate_tool_uniqueness(self, tool_params: List) -> None:
        """Ensure no duplicate tool names exist."""
        tool_names = sorted([param.name for param in tool_params])
        for i in range(len(tool_names) - 1):
            if tool_names[i] == tool_names[i + 1]:
                raise ValueError(f"🚨 Duplicate tool detected: {tool_names[i]}")

    def _log_environment_setup(self, env_type: str, identifier: str) -> None:
        """Log execution environment configuration."""
        setup_msg = f"🔧 {env_type} execution environment: {identifier}"
        print(colored(setup_msg, "cyan"))


class ConversationHandler:
    """
    Advanced Conversation Processing Engine

    Manages multi-turn conversations with intelligent token budgeting,
    context preservation, and response processing capabilities.
    """

    def __init__(self, logger: logging.Logger, enable_budgeting: bool = True):
        self._dialog_manager = ConversationFlow(
            logger_for_agent_logs=logger,
            use_prompt_budgeting=enable_budgeting
        )
        self._logger = logger
        self._conversation_history = []

    def initialize_conversation(self, instruction: str, quiet_mode: bool = False) -> None:
        """Initialize new conversation with user instruction."""
        self._log_user_instruction(instruction, quiet_mode)
        self._dialog_manager.add_user_prompt(instruction)

    def process_model_response(self, response, metadata) -> None:
        """Process and log model response."""
        self._log_model_interaction(response, metadata)
        self._dialog_manager.add_model_response(response)
        self._conversation_history.append(("model", response))

    def get_pending_tool_calls(self):
        """Retrieve pending tool calls from dialog."""
        return self._dialog_manager.get_pending_tool_calls()

    def add_tool_result(self, tool_call, result) -> None:
        """Add tool execution result to conversation."""
        processed_result = result[0] if isinstance(result, tuple) else result
        self._dialog_manager.add_tool_call_result(tool_call, processed_result)

    def get_messages_for_llm(self):
        """Get formatted messages for LLM client."""
        return self._dialog_manager.get_messages_for_llm_client()

    def get_final_response(self) -> str:
        """Get the last model text response."""
        return self._dialog_manager.get_last_model_text_response()

    def calculate_token_usage(self) -> int:
        """Calculate current token usage."""
        return self._dialog_manager.count_tokens()

    def reset_conversation(self) -> None:
        """Reset conversation state."""
        self._dialog_manager.clear()
        self._conversation_history.clear()

    def _log_user_instruction(self, instruction: str, quiet_mode: bool = False) -> None:
        """Log user instruction with formatting."""
        if not quiet_mode:
            separator = "═" * 25
            header = f"\n{separator} 👤 USER INSTRUCTION {separator}\n{instruction}\n"
            self._logger.info(header)
        else:
            # In quiet mode, log only brief information
            self._logger.info(f"User instruction received (length: {len(instruction)} chars)")

    def _log_model_interaction(self, response, metadata) -> None:
        """Log model response with metadata."""
        interaction_log = f"🤖 Model Response: {response} | 📊 Metadata: {metadata}"
        print(interaction_log)


class TaskOrchestrator:
    """
    Intelligent Task Orchestration Engine

    Coordinates complex task execution with phase management,
    progress tracking, and adaptive workflow control.
    """

    def __init__(self, max_iterations: int = 10, token_limit: int = 8192):
        self._max_iterations = max_iterations
        self._token_limit = token_limit
        self._current_context: Optional[TaskContext] = None

    def create_task_context(self, instruction: str) -> TaskContext:
        """Create new task execution context."""
        self._current_context = TaskContext(
            primary_instruction=instruction,
            remaining_iterations=self._max_iterations
        )
        return self._current_context

    def advance_phase(self, new_phase: ExecutionPhase) -> None:
        """Advance to next execution phase."""
        if self._current_context:
            old_phase = self._current_context.current_phase.value
            self._current_context.current_phase = new_phase
            self._current_context.metrics.phase_transitions.append(
                f"{old_phase} -> {new_phase.value}"
            )

    def consume_iteration(self) -> bool:
        """Consume one iteration and check if more remain."""
        if self._current_context:
            self._current_context.remaining_iterations -= 1
            self._current_context.metrics.total_turns += 1
            return self._current_context.remaining_iterations > 0
        return False

    def mark_interruption(self) -> None:
        """Mark task as interrupted."""
        if self._current_context:
            self._current_context.is_interrupted = True

    def get_current_context(self) -> Optional[TaskContext]:
        """Get current task context."""
        return self._current_context

    def log_iteration_status(self, logger: logging.Logger, quiet_mode: bool = False) -> None:
        """Log current iteration status."""
        if self._current_context and not quiet_mode:
            separator = "─" * 50
            status_info = f"\n{separator} 🔄 ITERATION STATUS {separator}"
            logger.info(status_info)
            logger.info(f"📊 Phase: {self._current_context.current_phase.value}")
            logger.info(f"🔢 Remaining: {self._current_context.remaining_iterations}")


class ResultProcessor:
    """
    Result Processing and Formatting Engine

    Handles result aggregation, formatting, and output generation
    with comprehensive error handling and status reporting.
    """

    @staticmethod
    def create_success_result(output: str, message: str = "Task completed") -> ToolImplOutput:
        """Create successful execution result."""
        return ToolImplOutput(
            tool_output=output,
            tool_result_message=message
        )

    @staticmethod
    def create_error_result(error: Exception) -> ToolImplOutput:
        """Create error result from exception."""
        error_msg = f"🚨 Execution error: {str(error)}"
        return ToolImplOutput(
            tool_output=error_msg,
            tool_result_message=error_msg
        )

    @staticmethod
    def create_interruption_result() -> ToolImplOutput:
        """Create interruption result."""
        interrupt_msg = "⚠️ Execution interrupted by user"
        return ToolImplOutput(
            tool_output=interrupt_msg,
            tool_result_message=interrupt_msg
        )

    @staticmethod
    def create_timeout_result() -> ToolImplOutput:
        """Create timeout result."""
        timeout_msg = "⏰ Maximum iterations reached"
        return ToolImplOutput(
            tool_output=timeout_msg,
            tool_result_message=timeout_msg
        )

    @staticmethod
    def create_completion_result(completion_tool: CompleteTool) -> ToolImplOutput:
        """Create completion result from completion tool."""
        return ToolImplOutput(
            tool_output=completion_tool.answer,
            tool_result_message="Task completed successfully"
        )


class ExecutionContext:
    """
    Execution Environment and State Management

    Manages execution environment configuration, resource allocation,
    and state persistence throughout the patch generation lifecycle.
    """

    def __init__(self, workspace_manager: ProjectSpaceHandler, console: Console,
                 logger: logging.Logger):
        self.workspace_manager = workspace_manager
        self.console = console
        self.logger = logger
        self._environment_config = {}
        self._resource_limits = {}

    def configure_environment(self, container_id: Optional[str] = None,
                            require_permission: bool = False) -> Dict[str, Any]:
        """Configure execution environment settings."""
        config = {
            "container_id": container_id,
            "require_permission": require_permission,
            "workspace_root": self.workspace_manager.root,
            "environment_type": "docker" if container_id else "local"
        }
        self._environment_config = config
        return config

    def get_workspace_context(self) -> str:
        """Generate workspace context for system prompt."""
        return str(self.workspace_manager.root)

    def log_environment_info(self) -> None:
        """Log current environment configuration."""
        env_type = self._environment_config.get("environment_type", "unknown")
        self.logger.info(f"🌐 Environment: {env_type}")
        if env_type == "docker":
            container = self._environment_config.get("container_id")
            self.logger.info(f"🐳 Container: {container}")


class PatchEngine(LLMTool):
    """
    Advanced Patch Generation Engine

    The core orchestration engine that coordinates all components for intelligent
    patch generation. Features modular architecture with specialized subsystems
    for task management, tool coordination, and execution control.

    Key Capabilities:
    - Intelligent task decomposition and workflow orchestration
    - Dynamic tool registration and lifecycle management
    - Advanced conversation handling with token optimization
    - Comprehensive error handling and recovery mechanisms
    - Detailed execution metrics and performance tracking
    """

    # Tool metadata for LLM integration
    name = "patch_generation_engine"
    description = """
    🔧 Advanced patch generation engine with modular architecture.

    Specializes in complex code modification tasks through intelligent
    orchestration of specialized tools and adaptive workflow management.

    ✨ Features: Multi-phase execution, dynamic tool coordination, smart conversation handling
    """

    input_schema = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "Detailed instruction for patch generation task.",
            },
        },
        "required": ["instruction"],
    }

    def __init__(
        self,
        llm_client: LLMClient,
        workspace_manager: ProjectSpaceHandler,
        console: Console,
        activity_logger: logging.Logger,
        max_response_tokens: int = 8192,
        max_iterations: int = 10,
        enable_token_budgeting: bool = True,
        require_user_confirmation: bool = False,
        container_id: Optional[str] = None,
        quiet_mode: bool = False,
    ):
        """
        Initialize Advanced Patch Generation Engine

        Sets up the complete modular architecture with specialized components
        for intelligent patch generation and code modification workflows.

        Args:
            llm_client: Language model client for AI interactions
            workspace_manager: File system and workspace management
            console: Enhanced console interface for user interaction
            activity_logger: Dedicated logger for tracking agent activities
            max_response_tokens: Maximum tokens per model response
            max_iterations: Maximum execution iterations before timeout
            enable_token_budgeting: Enable intelligent token management
            require_user_confirmation: Require user approval for actions
            container_id: Optional Docker container for isolated execution
        """
        super().__init__()

        # Initialize core subsystems
        self._llm_client = llm_client
        self._execution_context = ExecutionContext(workspace_manager, console, activity_logger)
        self._task_orchestrator = TaskOrchestrator(max_iterations, max_response_tokens)
        self._conversation_handler = ConversationHandler(activity_logger, enable_token_budgeting)
        self._tool_registry = ToolRegistry(workspace_manager)

        # Configure execution environment
        self._setup_execution_environment(container_id, require_user_confirmation)

        # Initialize component state
        self._max_response_tokens = max_response_tokens
        self._activity_logger = activity_logger
        self._quiet_mode = quiet_mode

    def _setup_execution_environment(self, container_id: Optional[str],
                                   require_confirmation: bool) -> None:
        """Configure and initialize execution environment."""
        # Configure execution context
        self._execution_context.configure_environment(container_id, require_confirmation)

        # Register tools in registry
        self._tool_registry.register_execution_environment(container_id, require_confirmation)
        self._tool_registry.register_core_tools()

        # Log environment setup
        self._execution_context.log_environment_info()


    @property
    def dialog(self):
        """Backward-compatibility shim.
        Some legacy code expects a `.dialog` attribute on the engine.
        Expose the internal dialog manager to avoid AttributeError while keeping
        the new architecture intact.
        """
        try:
            return self._conversation_handler._dialog_manager
        except Exception:
            return None

    def _generate_system_prompt(self) -> str:
        """Generate contextualized system prompt."""
        workspace_context = self._execution_context.get_workspace_context()
        return SYSTEM_PROMPT.format(workspace_root=workspace_context)

    def run_impl(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[ConversationFlow] = None,
    ) -> ToolImplOutput:
        """
        Execute Patch Generation Workflow

        Orchestrates the complete patch generation process through
        intelligent task coordination and multi-phase execution.

        Args:
            tool_input: Input parameters containing task instruction
            dialog_messages: Optional dialog context (maintained for compatibility)

        Returns:
            ToolImplOutput: Comprehensive execution result with generated patch
        """
        instruction = tool_input["instruction"]

        # Initialize task execution context
        task_context = self._task_orchestrator.create_task_context(instruction)
        self._conversation_handler.initialize_conversation(instruction, self._quiet_mode)

        # Execute main workflow loop
        return self._execute_main_workflow(task_context)

    def _execute_main_workflow(self, task_context: TaskContext) -> ToolImplOutput:
        """Execute the main patch generation workflow."""
        self._task_orchestrator.advance_phase(ExecutionPhase.EXECUTION)

        while self._task_orchestrator.consume_iteration():
            # Log iteration status
            self._task_orchestrator.log_iteration_status(self._activity_logger, self._quiet_mode)

            # Prepare tools for this iteration
            tool_parameters = self._tool_registry.get_tool_parameters()

            try:
                # Execute single iteration
                result = self._execute_single_iteration(tool_parameters)
                if result:
                    return result

            except Exception as error:
                return ResultProcessor.create_error_result(error)

        # Handle maximum iterations reached
        return ResultProcessor.create_timeout_result()

    def _execute_single_iteration(self, tool_parameters: List) -> Optional[ToolImplOutput]:
        """Execute a single workflow iteration."""
        # Generate model response
        model_response, metadata = self._llm_client.generate(
            messages=self._conversation_handler.get_messages_for_llm(),
            max_tokens=self._max_response_tokens,
            tools=tool_parameters,
            system_prompt=self._generate_system_prompt(),
            max_retries=10,  # Explicitly set retry count to ensure it is not overridden
        )

        # Process model response
        self._conversation_handler.process_model_response(model_response, metadata)

        # Handle tool calls
        return self._process_tool_invocations(model_response)

    def _process_tool_invocations(self, model_response) -> Optional[ToolImplOutput]:
        """Process tool calls from model response."""
        pending_calls = self._conversation_handler.get_pending_tool_calls()

        # Handle no tool calls (potential completion)
        if not pending_calls:
            self._activity_logger.info("✅ No tool calls - checking completion status")
            final_response = self._conversation_handler.get_final_response()
            return ResultProcessor.create_success_result(final_response)

        # Validate single tool call constraint
        if len(pending_calls) > 1:
            raise ValueError("🚨 Multiple simultaneous tool calls not supported")

        # Execute single tool call
        tool_call = pending_calls[0]
        return self._execute_tool_call(tool_call)

    def _execute_tool_call(self, tool_call) -> Optional[ToolImplOutput]:
        """Execute specific tool call with error handling."""
        try:
            # Locate tool in registry
            tool_instance = self._tool_registry.find_tool_by_name(tool_call.tool_name)

            # Log tool execution
            self._log_tool_execution(tool_call)

            # Execute tool
            execution_result = tool_instance.run(
                tool_call.tool_input,
                deepcopy(self._conversation_handler._dialog_manager)
            )

            # Process result
            self._conversation_handler.add_tool_result(tool_call, execution_result)

            # Check for completion
            completion_tracker = self._tool_registry.get_completion_tracker()
            if completion_tracker.should_stop:
                return ResultProcessor.create_completion_result(completion_tracker)

        except KeyboardInterrupt:
            return ResultProcessor.create_interruption_result()

        return None  # Continue workflow

    def _log_tool_execution(self, tool_call) -> None:
        """Log tool execution details."""
        if not self._quiet_mode:
            input_summary = "\n".join([
                f"  • {key}: {value}" for key, value in tool_call.tool_input.items()
            ])
            execution_log = f"🔧 Executing {tool_call.tool_name}:\n{input_summary}"
            self._activity_logger.info(execution_log)
        else:
            # In quiet mode, log only brief information
            self._activity_logger.info(f"🔧 Executing {tool_call.tool_name}")

    def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
        """
        Generate tool activation message

        Creates an informative message when the patch engine is activated.

        Args:
            tool_input: Input parameters for the engine

        Returns:
            str: Formatted activation message
        """
        instruction_preview = tool_input['instruction'][:120]
        if len(tool_input['instruction']) > 120:
            instruction_preview += "..."
        return f"🔧 Patch Engine activated for task: {instruction_preview}"

    def execute_patch_generation(
        self,
        task_instruction: str,
        resume_from_previous: bool = False,
        context_guidance: str | None = None,
    ) -> str:
        """
        Launch Patch Generation Workflow

        Initiates comprehensive patch generation with advanced state management,
        supporting both fresh execution and workflow resumption.

        Args:
            task_instruction: Primary task instruction for patch generation
            resume_from_previous: Continue from previous conversation state
            context_guidance: Optional contextual guidance for execution

        Returns:
            str: Final patch generation result
        """
        # Reset completion state for new execution
        completion_tracker = self._tool_registry.get_completion_tracker()
        completion_tracker.reset()

        # Handle execution state based on resume flag
        if resume_from_previous:
            # Validate resumption capability
            if not hasattr(self._conversation_handler._dialog_manager, 'is_user_turn'):
                raise ValueError("🚨 Cannot resume - invalid conversation state")
        else:
            # Reset to clean state for fresh execution
            self._reset_engine_state()

        # Prepare execution parameters
        execution_parameters = {"instruction": task_instruction}
        if context_guidance:
            execution_parameters["context_guidance"] = context_guidance

        # Execute patch generation workflow
        result = self.run(execution_parameters, self._conversation_handler._dialog_manager)
        return result

    def _reset_engine_state(self) -> None:
        """Reset engine to clean state for new execution."""
        self._conversation_handler.reset_conversation()

        # Reset task orchestrator state
        self._task_orchestrator._current_context = None

    def clear_engine_state(self) -> None:
        """
        Clear Engine State

        Resets the patch engine to a clean state, clearing all conversation
        history, task context, and execution state.
        """
        self._conversation_handler.reset_conversation()
        self._task_orchestrator._current_context = None

    def get_execution_metrics(self) -> Optional[ExecutionMetrics]:
        """
        📊 Get Execution Metrics

        Retrieves comprehensive execution metrics including performance
        statistics, resource usage, and workflow progression data.

        Returns:
            ExecutionMetrics: Current execution metrics or None if no active context
        """
        current_context = self._task_orchestrator.get_current_context()
        return current_context.metrics if current_context else None

    def get_current_phase(self) -> Optional[ExecutionPhase]:
        """
        Get Current Execution Phase

        Returns the current execution phase for monitoring and debugging.

        Returns:
            ExecutionPhase: Current phase or None if no active execution
        """
        current_context = self._task_orchestrator.get_current_context()
        return current_context.current_phase if current_context else None
