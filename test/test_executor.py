#!/usr/bin/env python3
"""
Test Executor Module

This module provides functionality to execute generated test cases and validate diff quality.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console


def execute_tests_in_container(container_id: str, console: Console = None) -> Tuple[bool, Dict]:
    """Execute generated test cases"""
    console = console or Console()
    
    try:
        console.print("🧪 Executing generated test cases...")
        
        # New: ensure pytest is preinstalled and available
        console.print("🔍 Checking and installing pytest...")
        _ensure_pytest_available(container_id, console)
        
        test_results = {}

        # Execute all three test files individually
        for test_file in ["test_happy_path.py", "test_edge_case.py", "test_failure_scenario.py"]:
            cmd = [
                "docker", "exec", container_id,
                "bash", "-c", f"""
                cd /testbed
                source /opt/miniconda3/etc/profile.d/conda.sh
                conda activate testbed
                pytest test_cases/{test_file} -v -s
                """
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Simple judgment
            if result.returncode == 0:
                test_results[test_file] = {
                    "status": "PASSED",
                    "output": result.stdout,
                    "error": "",
                    "exit_code": 0
                }
                console.print(f"✅ {test_file}: PASSED")
            else:
                test_results[test_file] = {
                    "status": "FAILED",
                    "output": result.stdout,
                    "error": result.stderr,
                    "exit_code": result.returncode
                }
                console.print(f"❌ {test_file}: FAILED (exit code: {result.returncode})")

        # Check if all tests passed
        all_passed = all(
            result.get("status") == "PASSED"
            for result in test_results.values()
        )

        if all_passed:
            console.print("✅ All tests passed successfully")
        else:
            console.print("❌ Some tests failed")

        return all_passed, test_results
        
    except Exception as e:
        console.print(f"💥 Error during test execution: {e}")
        return False, {}

def _ensure_pytest_available(container_id: str, console: Console):
    """Ensure pytest is available; install if missing"""
    try:
        # Check whether pytest is available
        check_cmd = [
            "docker", "exec", container_id,
            "bash", "-c", "python -c 'import pytest; print(\"pytest available\")' 2>/dev/null || echo 'pytest not found'"
        ]
        
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if "pytest available" in result.stdout:
            console.print("✅ pytest is available")
            return
        
        console.print("⚠️ pytest not available, installing...")
        
        # Install pytest using a mirror source
        install_cmd = [
            "docker", "exec", container_id,
            "bash", "-c", "pip install pytest"
        ]
        
        install_result = subprocess.run(install_cmd, capture_output=True, text=True)
        
        if install_result.returncode == 0:
            console.print("✅ pytest installed successfully")
        else:
            console.print(f"❌ pytest installation failed: {install_result.stderr}")
            # Try installing via conda as a fallback
            conda_cmd = [
                "docker", "exec", container_id,
                "bash", "-c", "conda install -y pytest"
            ]
            conda_result = subprocess.run(conda_cmd, capture_output=True, text=True)
            if conda_result.returncode == 0:
                console.print("✅ pytest installed successfully via conda")
            else:
                console.print("❌ pytest installation failed; test execution may fail")
                
    except Exception as e:
        console.print(f"💥 Exception during pytest availability check: {e}")


def validate_diff_quality(test_results: Dict, console: Console = None) -> Dict:
    """
    Validate the quality of the generated diff based on test results.

    Args:
        test_results: Results from test execution
        console: Rich console for output

    Returns:
        Dictionary containing validation results
    """
    console = console or Console()

    # Count test results
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result.get("status") == "PASSED")
    failed_tests = total_tests - passed_tests

    # Calculate success rate
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    # Determine overall quality
    if success_rate >= 80:
        quality = "EXCELLENT"
    elif success_rate >= 60:
        quality = "GOOD"
    elif success_rate >= 40:
        quality = "FAIR"
    else:
        quality = "POOR"

    # Add debug information for failed tests
    debug_info = {}
    for test_name, result in test_results.items():
        if result.get("status") == "FAILED":
            debug_info[test_name] = {
                "exit_code": result.get("exit_code"),
                "error_summary": _extract_error_summary(result.get("output", ""))
            }

    validation_result = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "success_rate": f"{success_rate:.1f}%",
        "quality": quality,
        "test_summary": {
            test_name: {
                "status": result.get("status"),
                "exit_code": result.get("exit_code")
            }
            for test_name, result in test_results.items()
        },
        "debug_info": debug_info
    }

    console.print(f"📊 Diff Quality Assessment: {quality} ({success_rate:.1f}% success rate)")
    console.print(f"   - Total tests: {total_tests}")
    console.print(f"   - Passed: {passed_tests}")
    console.print(f"   - Failed: {failed_tests}")

    # Log debug information for failed tests
    if failed_tests > 0:
        console.print("🔍 Debug information for failed tests:")
        for test_name, debug in debug_info.items():
            console.print(f"   - {test_name}: exit_code={debug['exit_code']}, error={debug['error_summary']}")

    return validation_result


def _extract_error_summary(output: str) -> str:
    """
    Extract a summary of the error from test output.
    
    Args:
        output: Test output string
        
    Returns:
        Error summary string
    """
    if not output:
        return "No output"
    
    lines = output.strip().split('\n')
    
    # Look for error lines (usually the last few lines contain the error)
    for line in reversed(lines):
        line = line.strip()
        if line and any(error_indicator in line.lower() for error_indicator in [
            "error", "fail", "exception", "traceback", "syntaxerror"
        ]):
            return line[:100] + "..." if len(line) > 100 else line
    
    # If no clear error line found, return first few lines
    return lines[0][:100] + "..." if lines and len(lines[0]) > 100 else (lines[0] if lines else "No error details")


def save_test_results(test_results: Dict, validation_result: Dict, output_path: Path) -> bool:
    """
    Save test results and validation to a file.
    
    Args:
        test_results: Results from test execution
        validation_result: Validation results
        output_path: Path to save the results
        
    Returns:
        True if saved successfully
    """
    try:
        # Combine all results
        combined_results = {
            "test_execution": test_results,
            "validation": validation_result,
            "timestamp": str(Path().cwd())
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save test results: {e}")
        return False
