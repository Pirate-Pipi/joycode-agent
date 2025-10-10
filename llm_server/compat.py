"""
LLM Server compatibility layer

This module provides a fully compatible interface with utils.llm_client,
allowing other modules to migrate to llm_server seamlessly.
"""

# Import all type definitions from the types module
from .types import (
    # Data classes
    ToolCall,
    ToolFormattedResult,
    TextPrompt,
    TextResult,
    
    # Type aliases
    AssistantContentBlock,
    UserContentBlock,
    GeneralContentBlock,
    LLMMessages,
    ToolParam,
    
    # Base classes
    LLMClient,
    
    # Utility functions
    recursively_remove_invoke_tag,
    
    # Anthropic types
    AnthropicRedactedThinkingBlock,
    AnthropicThinkingBlock,
)

# Import SimpleLLMClient from simple_client as an implementation of LLMClient
from .simple_client import SimpleLLMClient

# Create a compatible get_client function
def get_client(client_name: str, **kwargs) -> LLMClient:
    """
    A function compatible with utils.llm_client.get_client

    Args:
        client_name: Client name (kept for compatibility; not used)
        **kwargs: Other parameters (kept for compatibility)

    Returns:
        A SimpleLLMClient instance compatible with the LLMClient interface
    """
    from .simple_client import create_simple_client
    return create_simple_client()


# Export all compatible interfaces
__all__ = [
    # Data classes
    'ToolCall',
    'ToolFormattedResult',
    'TextPrompt',
    'TextResult',
    
    # Type aliases
    'AssistantContentBlock',
    'UserContentBlock',
    'GeneralContentBlock',
    'LLMMessages',
    'ToolParam',
    
    # Base class and implementation
    'LLMClient',
    'SimpleLLMClient',
    
    # Utility functions
    'recursively_remove_invoke_tag',
    'get_client',
    
    # Anthropic types
    'AnthropicRedactedThinkingBlock',
    'AnthropicThinkingBlock',
]
