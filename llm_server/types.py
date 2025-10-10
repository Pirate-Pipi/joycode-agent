"""
LLM Server Types - All LLM-related type definitions

This module contains all type definitions migrated from utils.llm_client,
ensuring the llm_server module is fully self-contained.
"""

import json
from dataclasses import dataclass
from typing import Any, Tuple, Union
from dataclasses_json import DataClassJsonMixin

# Import Anthropic-related types (for type definitions)
try:
    from anthropic.types import (
        RedactedThinkingBlock as AnthropicRedactedThinkingBlock,
        ThinkingBlock as AnthropicThinkingBlock,
        ToolParam as AnthropicToolParam,
        ToolResultBlockParam as AnthropicToolResultBlockParam,
    )
except ImportError:
    # If the anthropic package is unavailable, define placeholder types
    class AnthropicRedactedThinkingBlock:
        pass
    
    class AnthropicThinkingBlock:
        pass
    
    AnthropicToolParam = dict
    AnthropicToolResultBlockParam = dict

# Tool parameter type definition
@dataclass
class ToolParam(DataClassJsonMixin):
    """Tool parameter definition for LLM tools."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall(DataClassJsonMixin):
    """Internal representation of LLM tool call."""
    
    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class ToolFormattedResult(DataClassJsonMixin):
    """Internal representation of formatted LLM tool result."""
    
    tool_call_id: str
    tool_name: str
    tool_output: str


@dataclass
class TextPrompt(DataClassJsonMixin):
    """Internal representation of user-generated text prompt."""
    
    text: str


@dataclass
class TextResult(DataClassJsonMixin):
    """Internal representation of LLM-generated text result."""
    
    text: str


# Content block type definitions
AssistantContentBlock = Union[
    TextResult, 
    ToolCall, 
    AnthropicRedactedThinkingBlock, 
    AnthropicThinkingBlock
]

UserContentBlock = Union[TextPrompt, ToolFormattedResult]
GeneralContentBlock = Union[UserContentBlock, AssistantContentBlock]
LLMMessages = list[list[GeneralContentBlock]]


class LLMClient:
    """A client for LLM APIs for the use in agents."""
    
    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate responses.
        
        Args:
            messages: A list of messages.
            max_tokens: The maximum number of tokens to generate.
            system_prompt: A system prompt.
            temperature: The temperature.
            tools: A list of tools.
            tool_choice: A tool choice.
            
        Returns:
            A generated response.
        """
        raise NotImplementedError


def recursively_remove_invoke_tag(obj):
    """Recursively remove the </invoke> tag from a dictionary or list."""
    result_obj = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            result_obj[key] = recursively_remove_invoke_tag(value)
    elif isinstance(obj, list):
        result_obj = [recursively_remove_invoke_tag(item) for item in obj]
    elif isinstance(obj, str):
        if "</invoke>" in obj:
            result_obj = json.loads(obj.replace("</invoke>", ""))
        else:
            result_obj = obj
    else:
        result_obj = obj
    return result_obj


# Export all types and functions
__all__ = [
    # Data class
    'ToolCall',
    'ToolFormattedResult', 
    'TextPrompt',
    'TextResult',
    
    # Type alias
    'AssistantContentBlock',
    'UserContentBlock',
    'GeneralContentBlock',
    'LLMMessages',
    'ToolParam',
    
    # Base class
    'LLMClient',
    
    # Utility function
    'recursively_remove_invoke_tag',
    
    # Anthropic types (for compatibility)
    'AnthropicRedactedThinkingBlock',
    'AnthropicThinkingBlock',
    'AnthropicToolParam',
    'AnthropicToolResultBlockParam',
]
