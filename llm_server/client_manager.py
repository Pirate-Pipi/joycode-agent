import os
import json
import logging
import random
import time
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from openai import OpenAI


class ModelPurpose(Enum):
    """Model Usage Enumeration"""
    PATCH_GENERATION = "patch_generation"
    TEST_GENERATION = "test_generation" 
    TRAJECTORY_COMPRESSION = "trajectory_compression"
    SIMILARITY_SEARCH = "similarity_search"
    RETRY_AGENT = "retry_agent"


@dataclass
class ModelConfig:
    """Model Configuration - Unified Use of OpenAI Client"""
    model_name: str
    max_retries: int = 10
    temperature: float = 0.0
    max_tokens: int = 32000
    # OpenAI client settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[int] = None


class UnifiedModelManager:
    """Unified Model Manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self._clients: Dict[ModelPurpose, Any] = {}
        self._configs: Dict[ModelPurpose, ModelConfig] = {}
        # Config file path - only use the config file inside the llm_server folder
        if config_file:
            self.config_file = config_file
        else:
            # Use the config file in the llm_server folder
            current_dir = Path(__file__).parent
            config_path = current_dir / "model_config.json"
            self.config_file = str(config_path)
        self._setup_configs()
    
    def _setup_configs(self):
        """Set Configuration - Prefer Loading from Config File, Otherwise Use Default Configuration"""
        try:
            if Path(self.config_file).exists():
                self._load_configs_from_file()
                self.logger.info(f"Loaded configs from {self.config_file}")
            else:
                self._setup_default_configs()
                self.logger.info("Using default configs")
        except Exception as e:
            self.logger.warning(f"Failed to load config file: {e}, using defaults")
            self._setup_default_configs()

    def _load_configs_from_file(self):
        """Load Configuration from Config File"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        models_config = config_data.get('models', {})

        for purpose in ModelPurpose:
            purpose_key = purpose.value
            if purpose_key in models_config:
                model_config = models_config[purpose_key]
                self._configs[purpose] = ModelConfig(
                    model_name=model_config.get('model_name', 'Claude-sonnet-4'),
                    max_retries=model_config.get('max_retries', 10),
                    temperature=model_config.get('temperature', 0.0),
                    max_tokens=model_config.get('max_tokens', 32000),
                    api_key=model_config.get('api_key'),
                    base_url=model_config.get('base_url'),
                    timeout=model_config.get('timeout')
                )
            else:
                self._setup_default_config_for_purpose(purpose)

    def _setup_default_configs(self):
        for purpose in ModelPurpose:
            self._setup_default_config_for_purpose(purpose)

    def _setup_default_config_for_purpose(self, purpose: ModelPurpose):
        default_api_key = "xxx"
        
        if purpose in [ModelPurpose.PATCH_GENERATION, ModelPurpose.TEST_GENERATION, ModelPurpose.RETRY_AGENT]:
            self._configs[purpose] = ModelConfig(
                model_name="Claude-sonnet-4",
                api_key=default_api_key,
                base_url="xxx",
                max_retries=10,
                temperature=0.0,
                max_tokens=32000,
                timeout=300
            )
        else:
            self._configs[purpose] = ModelConfig(
                model_name="gpt-4.1",
                api_key=default_api_key,
                base_url="xxx",
                max_retries=10,
                temperature=0.0,
                max_tokens=32000,
                timeout=300
            )

    def update_config(self, purpose: ModelPurpose, config: ModelConfig):
        """Update configuration for the specified purpose"""
        self._configs[purpose] = config
        # Clear the cached client to force re-creation
        if purpose in self._clients:
            del self._clients[purpose]
        self.logger.info(f"Updated config for {purpose.value}: {config.model_name}")
    
    def get_client(self, purpose: ModelPurpose):
        if purpose not in self._clients:
            self._clients[purpose] = self._create_client(purpose)

        # Log acquired client information
        config = self._configs[purpose]
        print(f"📋 [Client] Purpose: {purpose.value}, Model: {config.model_name}")
        self.logger.info(f"Getting client for {purpose.value} with model {config.model_name}")

        return self._clients[purpose]
    
    def _create_client(self, purpose: ModelPurpose):
        config = self._configs[purpose]

        # Create OpenAI client with optimized connection settings
        import httpx

        # Create a tuned HTTP client with strict timeouts
        actual_timeout = config.timeout or 60
        timeout_config = httpx.Timeout(
            connect=10.0,  # Connect timeout: 10s
            read=actual_timeout,  # Read timeout: use configured timeout
            write=30.0,    # Write timeout: 30s
            pool=10.0      # Pool timeout: 10s (fix for previous 5s setting)
        )

        http_client = httpx.Client(
            timeout=timeout_config,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0  # Keep connections alive for 30s
            ),
            follow_redirects=True
        )

        openai_client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout or 60,
            max_retries=0, 
            http_client=http_client
        )

        # Wrap into SimpleLLMClient
        from .simple_client import SimpleLLMClient
        return SimpleLLMClient(openai_client, config.model_name)
    
    def get_config(self, purpose: ModelPurpose) -> ModelConfig:
        return self._configs[purpose]

    def save_configs_to_file(self):
        try:
            # Read the existing config file to preserve other fields
            config_data = {}
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

            if 'models' not in config_data:
                config_data['models'] = {}

            for purpose, config in self._configs.items():
                config_data['models'][purpose.value] = {
                    "model_name": config.model_name,
                    "max_retries": config.max_retries,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "api_key": config.api_key,
                    "base_url": config.base_url,
                    "timeout": config.timeout
                }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved configs to {self.config_file}")

        except Exception as e:
            self.logger.error(f"Failed to save configs to file: {e}")
            raise


_model_manager = UnifiedModelManager()


def get_client_for_purpose(purpose: Union[ModelPurpose, str]) -> Any:
    """
    Get the client for the specified purpose

    Args:
        purpose: Model purpose, can be a ModelPurpose enum or a string
    Returns:
        Corresponding client instance
    """
    if isinstance(purpose, str):
        purpose = ModelPurpose(purpose)
    
    return _model_manager.get_client(purpose)


def update_model_config(purpose: Union[ModelPurpose, str], config: ModelConfig):
    """
    Update the model configuration for the specified purpose and save to file

    Args:
        purpose: Model purpose
        config: New configuration
    """
    if isinstance(purpose, str):
        purpose = ModelPurpose(purpose)

    _model_manager.update_config(purpose, config)

    _model_manager.save_configs_to_file()


def get_model_config(purpose: Union[ModelPurpose, str]) -> ModelConfig:
   
    if isinstance(purpose, str):
        purpose = ModelPurpose(purpose)
    
    return _model_manager.get_config(purpose)


# Compatibility function - keep backward compatibility
def get_client(client_name: str = None, **kwargs):
    return get_patch_generation_client()


# Convenience functions
def get_patch_generation_client():
    print(f"🔧 [Helper] Get patch-generation client")
    return get_client_for_purpose(ModelPurpose.PATCH_GENERATION)


def get_test_generation_client():
    print(f"🧪 [Helper] Get test-generation client")
    return get_client_for_purpose(ModelPurpose.TEST_GENERATION)


def get_trajectory_compression_client():
    print(f"📦 [Helper] Get trajectory-compression client")
    return get_client_for_purpose(ModelPurpose.TRAJECTORY_COMPRESSION)


def get_similarity_search_client():
    print(f"🔍 [Helper] Get similarity-search client")
    return get_client_for_purpose(ModelPurpose.SIMILARITY_SEARCH)


def get_retry_agent_client():
    print(f"🔄 [Helper] Get retry-agent client")
    return get_client_for_purpose(ModelPurpose.RETRY_AGENT)


def switch_model_for_purpose(purpose: Union[ModelPurpose, str], model_name: str):
    """
    Switch the model for the specified purpose

    Args:
        purpose: Model purpose
        model_name: New model name
    """
    if isinstance(purpose, str):
        purpose = ModelPurpose(purpose)
    
    current_config = get_model_config(purpose)
    new_config = ModelConfig(
        model_name=model_name,
        max_retries=current_config.max_retries,
        temperature=current_config.temperature,
        max_tokens=current_config.max_tokens,
        api_key=current_config.api_key,
        base_url=current_config.base_url,
        timeout=current_config.timeout
    )
    
    update_model_config(purpose, new_config)
    print(f"Switched {purpose.value} to model: {model_name}")


# 统一的LLM调用函数
def call_llm_with_tools(purpose: Union[ModelPurpose, str],
                       messages: List[Dict[str, str]],
                       tools: Optional[List[Dict]] = None,
                       max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None) -> Dict[str, Any]:
    """
    Unified LLM call function with tool-calling support

    Args:
        purpose: Model purpose
        messages: List of messages
        tools: List of tool definitions
        max_tokens: Maximum tokens
        temperature: Temperature parameter

    Returns:
        Model response result
    """
    if isinstance(purpose, str):
        purpose = ModelPurpose(purpose)

    client = get_client_for_purpose(purpose)
    config = get_model_config(purpose)

    # Use parameters from config if not provided
    final_max_tokens = max_tokens or config.max_tokens
    final_temperature = temperature if temperature is not None else config.temperature

    # Use SimpleLLMClient.generate(), which includes a full retry mechanism
    try:
        content_blocks, metadata = client.generate(
            messages=messages,
            max_tokens=final_max_tokens,
            temperature=final_temperature,
            tools=tools,
            max_retries=config.max_retries
        )

        # Extract content and tool calls
        content = None
        tool_calls = None

        for block in content_blocks:
            if hasattr(block, 'text') and block.text:
                content = block.text
            elif hasattr(block, 'tool_name'):
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(block)

        return {
            "success": True,
            "response": None,  # Do not return the raw response object
            "content": content,
            "tool_calls": tool_calls,
            "metadata": metadata
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": None,
            "tool_calls": None
        }


def call_llm_simple(purpose: Union[ModelPurpose, str],
                   prompt: str,
                   system_prompt: Optional[str] = None,
                   max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None) -> Optional[str]:
    """
    Simple LLM call function for text generation

    Args:
        purpose: Model purpose
        prompt: User prompt
        system_prompt: System prompt
        max_tokens: Maximum tokens
        temperature: Temperature parameter

    Returns:
        Generated text content
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    result = call_llm_with_tools(purpose, messages, max_tokens=max_tokens, temperature=temperature)

    if result["success"]:
        return result["content"]
    else:
        print(f"LLM call failed: {result['error']}")
        return None


# Advanced Function - Convenient Alias for Trajectory Compression
def compress_trajectory(trajectory_content: str, patch_content: str,
                       system_prompt: str, user_prompt: str,
                       max_retries: int = None) -> Optional[str]:
    """
    Helper function to call the trajectory compression model

    Args:
        trajectory_content: Trajectory content
        patch_content: Patch content
        system_prompt: System prompt
        user_prompt: User prompt template
        max_retries: Maximum retries

    Returns:
        Compressed trajectory content
    """
    config = get_model_config(ModelPurpose.TRAJECTORY_COMPRESSION)
    retries = max_retries or config.max_retries

    formatted_user_prompt = user_prompt.format(
        trajectory_content=trajectory_content,
        patch_content=patch_content
    )

    for attempt in range(retries):
        result = call_llm_simple(
            ModelPurpose.TRAJECTORY_COMPRESSION,
            formatted_user_prompt,
            system_prompt=system_prompt
        )

        if result:
            return result
        else:
            print(f"Attempt {attempt+1}: Empty response or content.")
            if attempt == retries - 1:
                return None

    return None


def call_similarity_search_model(messages: str, max_retries: int = None) -> Optional[str]:
    """
    Helper function to call the similarity search model

    Args:
        messages: Input messages
        max_retries: Maximum retries

    Returns:
        Model response content
    """
    config = get_model_config(ModelPurpose.SIMILARITY_SEARCH)
    retries = max_retries or config.max_retries

    for attempt in range(retries):
        result = call_llm_simple(
            ModelPurpose.SIMILARITY_SEARCH,
            messages
        )

        if result:
            return result
        else:
            print(f"Attempt {attempt+1}: Empty response or content.")
            if attempt == retries - 1:
                return None

    return None


def list_all_configs() -> Dict[str, Dict[str, Any]]:
    """
    List all model configurations

    Returns:
        A dictionary containing all configurations
    """
    configs = {}
    for purpose in ModelPurpose:
        config = get_model_config(purpose)
        configs[purpose.value] = {
            "model_name": config.model_name,
            "max_retries": config.max_retries,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "api_key": config.api_key[:10] + "..." if config.api_key else None,
            "base_url": config.base_url,
            "timeout": config.timeout
        }
    return configs
