"""
LLM Server - Unified large model client management module

This module contains all code related to LLM clients and invocations:
- Unified model manager
- Simplified LLM client
- Model configuration management
- Convenient invocation functions

Main components:
- client_manager: Unified model client manager
- simple_client: Simplified LLM client, compatible with previous interfaces
- config: Model configuration management
- utils: LLM-related utility functions
"""

# Import type definitions
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

    # Anthropic types (for compatibility)
    AnthropicRedactedThinkingBlock,
    AnthropicThinkingBlock,
)

# Import primary interfaces
from .client_manager import (
    ModelPurpose,
    ModelConfig,
    UnifiedModelManager,
    get_client_for_purpose,
    get_patch_generation_client,
    get_test_generation_client,
    get_trajectory_compression_client,
    get_similarity_search_client,
    get_retry_agent_client,
    get_model_config,
    update_model_config,
    switch_model_for_purpose,
    call_llm_with_tools,
    call_llm_simple,
    compress_trajectory,
    call_similarity_search_model,
    list_all_configs,
    get_client  # compatibility function
)

from .simple_client import (
    SimpleLLMClient,
    ToolCallParameters,
    create_simple_client,
    create_simple_client_for_purpose,
    extract_tool_calls_from_response,
    format_messages_for_openai
)

# Version info
__version__ = "2.0.0"
__author__ = "JoyCode Team"

# Export all primary interfaces
__all__ = [
    # Type definitions
    'ToolCall',
    'ToolFormattedResult',
    'TextPrompt',
    'TextResult',
    'AssistantContentBlock',
    'UserContentBlock',
    'GeneralContentBlock',
    'LLMMessages',
    'ToolParam',
    'LLMClient',
    'recursively_remove_invoke_tag',
    'AnthropicRedactedThinkingBlock',
    'AnthropicThinkingBlock',

    # Model management
    'ModelPurpose',
    'ModelConfig',
    'UnifiedModelManager',

    # Client access
    'get_client_for_purpose',
    'get_patch_generation_client',
    'get_test_generation_client',
    'get_trajectory_compression_client',
    'get_similarity_search_client',
    'get_retry_agent_client',

    # Configuration management
    'get_model_config',
    'update_model_config',
    'switch_model_for_purpose',

    # LLM calls
    'call_llm_with_tools',
    'call_llm_simple',
    'compress_trajectory',
    'call_similarity_search_model',
    'list_all_configs',

    # Simplified client
    'SimpleLLMClient',
    'ToolCallParameters',
    'create_simple_client',
    'create_simple_client_for_purpose',
    'extract_tool_calls_from_response',
    'format_messages_for_openai',

    # Compatibility
    'get_client'
]
