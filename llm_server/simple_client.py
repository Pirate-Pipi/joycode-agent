"""
Simplified LLM client, a replacement for the complex utils/llm_client.py
Uses only the OpenAI package; supports tool calling and message handling
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from openai import OpenAI

# Import required types using internal type definitions
from .types import TextResult, ToolCall, AssistantContentBlock
from .performance_monitor import record_api_call


@dataclass
class ToolCallParameters:
    """Parameters for tool calls"""
    tool_call_id: str
    tool_name: str
    tool_input: Dict[str, Any]


class SimpleLLMClient:
    """Simplified LLM client compatible with the previous interface"""

    def __init__(self, openai_client: OpenAI, model_name: str):
        self.client = openai_client
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        system_prompt: str = None,
        temperature: float = None,  # Set to None to use the default from the config file
        tools: List[Dict] = None,
        tool_choice: Dict[str, str] = None,
        thinking_tokens: int = None,
        max_retries: int = None,
    ) -> Tuple[List[AssistantContentBlock], Dict[str, Any]]:
        """
        Generate a response, compatible with the previous interface, with a full retry mechanism.

        Args:
            messages: List of messages
            max_tokens: Maximum number of tokens
            system_prompt: System prompt
            temperature: Temperature
            tools: List of tools
            tool_choice: Tool selection
            thinking_tokens: Number of thinking tokens (ignored)
            max_retries: Maximum number of retries; if not specified, use the value from the config file

        Returns:
            (List of content blocks, metadata dict)
        """
        # Get retry count configuration
        if max_retries is None:
            # Get retry count from the config file rather than from the OpenAI client
            from .client_manager import get_model_config, ModelPurpose
            try:
                # Try to infer the purpose based on the model name
                purpose = self._infer_purpose_from_model_name()
                config = get_model_config(purpose)
                max_retries = config.max_retries
                print(f"🔧 [Retry Config] Loaded retry count from config: {max_retries} (purpose: {purpose.value})")
            except Exception as e:
                # If inference fails, use a higher default
                max_retries = 10  # Increase the default retry count
                print(f"⚠️ [Retry Config] Failed to load retry count from config, using default: {max_retries} (error: {e})")
        else:
            print(f"🔧 [Retry Config] Using provided retry count: {max_retries}")

        # Force debug logs - ensure we know the actual retry count used
        print(f"🎯 [Retry Confirm] Final retries: {max_retries}, total attempts: {max_retries + 1}")

        # Safety check: ensure the number of retries is not too low
        if max_retries < 3:
            print(f"⚠️ [Retry Warning] Retry count too low ({max_retries}); forcing to 5")
            max_retries = 5

        # Get temperature configuration
        if temperature is None:
            # Get default temperature from the config file
            from .client_manager import get_model_config, ModelPurpose
            try:
                config = get_model_config(ModelPurpose.PATCH_GENERATION)
                temperature = config.temperature
            except:
                temperature = 1.0  # If retrieval fails, use GPT-5's default

        # Output model invocation information
        start_time = time.time()
        print(f"🤖 [LLM call] Model: {self.model_name}, API: {self.client.base_url}, Timeout: {self.client.timeout}s")
        self.logger.info(f"Calling LLM model: {self.model_name} at {self.client.base_url}")

        # Retry mechanism
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                result = self._single_generate_attempt(
                    messages, max_tokens, system_prompt, temperature,
                    tools, tool_choice, start_time, attempt + 1, max_retries + 1
                )

                # Log successful API calls
                end_time = time.time()
                input_tokens = result[1].get("usage", {}).get("input_tokens", 0)
                output_tokens = result[1].get("usage", {}).get("output_tokens", 0)

                record_api_call(
                    model_name=self.model_name,
                    base_url=str(self.client.base_url),
                    start_time=start_time,
                    end_time=end_time,
                    success=True,
                    attempt_count=attempt + 1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )

                return result

            except Exception as e:
                last_exception = e
                attempt_time = time.time() - start_time

                # Check if it is a retryable error
                error_str = str(e).lower()
                error_type = type(e).__name__.lower()

                # More comprehensive error detection (supports both English and Chinese)
                retryable_keywords = [
                    'timeout', 'connection', 'network', 'rate limit', 'overloaded',
                    '500', '502', '503', '504', '429', 'internal server error',
                    'service unavailable', 'bad gateway', 'gateway timeout',
                    'request timed out', 'read timeout', 'connect timeout',
                    '调用模型服务失败', '服务不可用', '网络错误', '连接超时',
                    '服务器内部错误', '网关超时', '服务过载',
                    'internal', 'failed_response', 'service_error'
                ]

                is_retryable = any(keyword in error_str for keyword in retryable_keywords) or \
                              any(keyword in error_type for keyword in ['timeout', 'connection', 'network'])

                # Special handling for common error types
                if 'timeouterror' in error_type:
                    is_retryable = True

                if attempt < max_retries and is_retryable:
                    # Use different retry strategies for different error types
                    if '504' in error_str or 'gateway timeout' in error_str or '网关超时' in error_str:
                        # Quick retry for 504 errors, as these are often temporary gateway issues
                        wait_time = 0.5 + attempt * 0.5  # 0.5s, 1s, 1.5s, 2s, 2.5s
                        print(f"🚨 [Gateway timeout] Attempt {attempt + 1} failed, quick retry in {wait_time}s")
                    elif 'timeout' in error_str or '超时' in error_str:
                        # General timeout: slightly longer wait
                        wait_time = 1 + attempt  # 1s, 2s, 3s, 4s, 5s
                        print(f"⏱️ [Request timeout] Attempt {attempt + 1} failed, retrying in {wait_time}s")
                    elif 'internal' in error_str or '调用模型服务失败' in error_str or 'failed_response' in error_str:
                        # Internal service error: longer wait
                        wait_time = 2 + attempt * 2  # 2s, 4s, 6s, 8s, 10s
                        print(f"🔧 [Service error] Attempt {attempt + 1} failed, retrying in {wait_time}s")
                    else:
                        # Other errors: use the original backoff strategy
                        wait_time = min(1 + attempt * 2, 5)
                        print(f"⚠️ [LLM retry] Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")

                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or reached max retries
                    print(f"❌ [LLM failure] Elapsed: {attempt_time:.2f}s, Error: {e}")

                    # Detailed error analysis logs
                    print(f"🔍 [Error analysis] Error type: {type(e).__name__}")
                    print(f"🔍 [Error analysis] Retryable: {is_retryable}")
                    print(f"🔍 [Error analysis] Attempt: {attempt + 1}/{max_retries + 1}")
                    print(f"🔍 [Error analysis] Error message: {str(e)[:200]}...")

                    self.logger.error(f"LLM generation failed after {attempt + 1} attempts: {e}")

                    if is_retryable:
                        print(f"⚠️ [Retries exhausted] Retryable error but reached max attempts")
                        raise e  # Retryable error but maximum attempts reached, throw exception
                    else:
                        print(f"🚫 [Not retryable] Error type does not support retry")
                        # Non-retryable error, return empty result
                        return [], {"error": str(e)}

        # If all retries fail
        print(f"❌ [LLM failure] All {max_retries + 1} attempts failed")
        self.logger.error(f"All {max_retries + 1} attempts failed")

        # Log failed API calls
        end_time = time.time()
        error_type = type(last_exception).__name__ if last_exception else "Unknown"
        error_message = str(last_exception) if last_exception else "Unknown error"

        record_api_call(
            model_name=self.model_name,
            base_url=str(self.client.base_url),
            start_time=start_time,
            end_time=end_time,
            success=False,
            attempt_count=max_retries + 1,
            error_type=error_type,
            error_message=error_message
        )

        raise last_exception

    def _infer_purpose_from_model_name(self):
        """Infer purpose based on the model name"""
        from .client_manager import ModelPurpose

        model_name_lower = self.model_name.lower()

        # Infer purpose from model name
        if 'claude' in model_name_lower:
            if 'test' in model_name_lower:
                return ModelPurpose.TEST_GENERATION
            else:
                return ModelPurpose.PATCH_GENERATION
        elif 'gpt-5' in model_name_lower:
            return ModelPurpose.PATCH_GENERATION
        elif 'gpt-4' in model_name_lower:
            return ModelPurpose.TRAJECTORY_COMPRESSION
        else:
            # Default to patch generation purpose
            return ModelPurpose.PATCH_GENERATION

    def _single_generate_attempt(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        system_prompt: str,
        temperature: float,
        tools: List[Dict],
        tool_choice: Dict[str, str],
        start_time: float,
        attempt_num: int,
        total_attempts: int,
    ) -> Tuple[List[AssistantContentBlock], Dict[str, Any]]:
        """Execute a single generation attempt"""
        if attempt_num > 1:
            print(f"🔄 [LLM retry] Attempt {attempt_num}/{total_attempts}...")

        # Check whether the total timeout has been exceeded
        elapsed_time = time.time() - start_time
        timeout_limit = getattr(self.client, 'timeout', 60)
        if elapsed_time > timeout_limit:
            raise TimeoutError(f"Total elapsed time {elapsed_time:.1f}s exceeds timeout limit {timeout_limit}s")

        try:
            # Build the full message list
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})

            # Convert message format
            for msg in messages:
                if isinstance(msg, dict):
                    full_messages.append(msg)
                else:
                    full_messages.append({"role": "user", "content": str(msg)})

            # Convert tool format
            openai_tools = None
            if tools:
                openai_tools = []
                for tool in tools:
                    if hasattr(tool, 'name'):  # ToolParam object
                        tool_def = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema
                            }
                        }
                        openai_tools.append(tool_def)
                    elif isinstance(tool, dict):  # Already in dict format
                        openai_tools.append(tool)

            # Call OpenAI API
            kwargs = {
                "model": self.model_name,
                "messages": full_messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            if openai_tools:
                kwargs["tools"] = openai_tools
                if tool_choice:
                    if isinstance(tool_choice, str):
                        kwargs["tool_choice"] = tool_choice
                    elif isinstance(tool_choice, dict):
                        kwargs["tool_choice"] = tool_choice

            response = self.client.chat.completions.create(**kwargs)

            # Process response
            message = response.choices[0].message
            content_blocks = []

            # Handle text content
            if message.content:
                content_blocks.append(TextResult(text=message.content))

            # Handle tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse tool arguments: {e}")
                        tool_input = {"error": "Failed to parse arguments"}

                    content_blocks.append(ToolCall(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.function.name,
                        tool_input=tool_input
                    ))

            # Build metadata
            message_metadata = {
                "usage": {
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0
                }
            }

            # Output call completion information
            end_time = time.time()
            duration = end_time - start_time
            input_tokens = message_metadata.get("usage", {}).get("input_tokens", 0)
            output_tokens = message_metadata.get("usage", {}).get("output_tokens", 0)

            if attempt_num == 1:
                print(f"✅ [LLM done] Duration: {duration:.2f}s, Input: {input_tokens} tokens, Output: {output_tokens} tokens")
            else:
                print(f"✅ [LLM retry succeeded] Attempt {attempt_num} succeeded, Duration: {duration:.2f}s, Input: {input_tokens} tokens, Output: {output_tokens} tokens")

            return content_blocks, message_metadata

        except Exception as e:
            # Rethrow the exception for the upper-level retry mechanism to handle
            raise e


def create_simple_client(purpose: str = "patch_generation") -> SimpleLLMClient:
    """
    Create a simplified LLM client; the model name is read from the config file.

    Args:
        purpose: Model purpose, defaults to "patch_generation"

    Returns:
        SimpleLLMClient instance
    """
    from .client_manager import get_client_for_purpose, ModelPurpose

    if isinstance(purpose, str):
        purpose_enum = ModelPurpose(purpose)
    else:
        purpose_enum = purpose

    return get_client_for_purpose(purpose_enum)


def create_simple_client_for_purpose(purpose) -> SimpleLLMClient:
    """
    Create a simplified LLM client for the specified purpose.

    Args:
        purpose: A ModelPurpose enum value

    Returns:
        SimpleLLMClient instance
    """
    return create_simple_client(purpose.value)


def extract_tool_calls_from_response(content_blocks: List[AssistantContentBlock]) -> List[ToolCallParameters]:
    """
    Extract tool call parameters from response content blocks.

    Args:
        content_blocks: List of response content blocks

    Returns:
        List of tool call parameters
    """
    tool_calls = []
    
    for block in content_blocks:
        if isinstance(block, ToolCall):
            tool_calls.append(ToolCallParameters(
                tool_call_id=block.tool_call_id,
                tool_name=block.tool_name,
                tool_input=block.tool_input
            ))
    
    return tool_calls


def format_messages_for_openai(messages: List[Any]) -> List[Dict[str, str]]:
    """Format messages to OpenAI format"""
    formatted_messages = []
    
    for msg in messages:
        if isinstance(msg, dict):
            formatted_messages.append(msg)
        elif hasattr(msg, 'role') and hasattr(msg, 'content'):
            formatted_messages.append({
                "role": msg.role,
                "content": str(msg.content)
            })
        else:
            formatted_messages.append({
                "role": "user",
                "content": str(msg)
            })
    
    return formatted_messages


def get_client(client_name: str = None, **kwargs) -> SimpleLLMClient:
    """Compatibility function that returns the simplified client"""
    # Ignore incoming parameters; use default configuration
    _ = client_name, kwargs  # Avoid unused variable warnings
    return create_simple_client("patch_generation")  # ✅ Explicitly specify patch_generation


# Export main interfaces
__all__ = [
    'SimpleLLMClient',
    'ToolCallParameters',
    'create_simple_client',
    'create_simple_client_for_purpose',
    'extract_tool_calls_from_response',
    'format_messages_for_openai',
    'get_client'
]
