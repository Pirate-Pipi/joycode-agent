"""
Test Case Generator Package

This package provides functionality for generating test cases within SWE-bench containers.
"""

from .integration import (
    generate_test_cases_in_container,
    copy_test_cases_to_output,
    validate_test_cases_against_original_code,
    regenerate_test_cases_with_improved_prompts
)

__all__ = [
    "generate_test_cases_in_container",
    "copy_test_cases_to_output",
    "validate_test_cases_against_original_code",
    "regenerate_test_cases_with_improved_prompts"
]
