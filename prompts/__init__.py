"""
Prompts package for SWE-bench agent.

This package contains all the prompt templates used by the agent.
"""

from .instruction import INSTRUCTION_PROMPT
from .system_prompt import SYSTEM_PROMPT
from .test_case import (
    TEST_CASE_GENERATION_INSTRUCTIONS,
    TEST_CASE_GENERATION_AGENT_CONFIG
)
from .diff_retry import INSTRUCTION_PROMPT as DIFF_RETRY_INSTRUCTION_PROMPT

__all__ = [
    "INSTRUCTION_PROMPT",
    "SYSTEM_PROMPT",
    "TEST_CASE_GENERATION_INSTRUCTIONS",
    "TEST_CASE_GENERATION_AGENT_CONFIG",
    "DIFF_RETRY_INSTRUCTION_PROMPT"
]
