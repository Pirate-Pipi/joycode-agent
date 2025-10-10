#!/usr/bin/env python3
"""
Utility functions for analyzing SWE-bench agent results.

This module provides functions to categorize and analyze the results
from running the agent on SWE-bench problems.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from rich.console import Console


def _validate_pre_execution_behavior(pre_validation: Dict[str, Any]) -> bool:
    """
    Validate that pre-execution validation was successful.
    
    New simplified logic: only check if validation_success is true
    
    Args:
        pre_validation: Pre-execution validation results
        
    Returns:
        True if validation_success is true, False otherwise
    """
    if not pre_validation:
        return False
    
    # Simply check if validation_success is true
    return pre_validation.get("validation_success", False) == True


def analyze_agent_results(output_files_dir: str = "output_files") -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Analyze agent results from output_files directory and categorize them.
    
    New categorization logic:
    1. Empty Diff: model_patch is empty or missing
    2. Successful: model_patch exists + post_execution_validation exists +
       validation_summary.total_tests == 3 + pre_execution_validation.validation_success == true
    3. Test Generation Failed: no test_cases directory found
    4. Test Validation Failed: test_cases exist but diff failed validation
    5. Other Failed: all other failure cases
    
    Args:
        output_files_dir: Path to the output_files directory
        
    Returns:
        Tuple of five lists:
        - successful_instances: List of instance_ids that succeeded
        - test_gen_failed_instances: List of instance_ids with test generation failure
        - test_validation_failed_instances: List of instance_ids with test validation failure
        - other_failed_instances: List of other failed instance_ids
        - empty_diff_instances: List of instance_ids with empty diffs
    """
    output_path = Path(output_files_dir)
    if not output_path.exists():
        return [], [], [], [], []
    
    successful_instances = []
    test_gen_failed_instances = []
    test_validation_failed_instances = []
    other_failed_instances = []
    empty_diff_instances = []
    
    # Iterate through all subdirectories in output_files
    for problem_dir in output_path.iterdir():
        if not problem_dir.is_dir():
            continue
            
        # Look for predictions.json file
        predictions_file = problem_dir / "predictions.json"
        if not predictions_file.exists():
            continue
            
        try:
            # Parse predictions.json
            with open(predictions_file, 'r') as f:
                predictions_data = json.load(f)
            
            if not predictions_data or not isinstance(predictions_data, list):
                continue
                
            # Get the first (and should be only) prediction
            prediction = predictions_data[0]
            instance_id = prediction.get("instance_id")
            model_patch = prediction.get("model_patch", "")
            
            if not instance_id:
                continue
                
            # Check if diff is empty
            if not model_patch or model_patch.strip() == "":
                empty_diff_instances.append(instance_id)
                continue
            
            # Check if test_cases directory exists in the problem directory
            test_cases_dir = problem_dir / "test_cases"
            if not test_cases_dir.exists() or not test_cases_dir.is_dir():
                # No test_cases directory means test generation failed
                test_gen_failed_instances.append(instance_id)
                continue
            
            # Check if post_execution_validation exists
            post_validation = prediction.get("post_execution_validation", {})
            if not post_validation:
                # Test cases exist but no post_execution_validation means test validation failed
                test_validation_failed_instances.append(instance_id)
                continue
            
            # Check validation_summary exists and total_tests == 3
            validation_summary = post_validation.get("validation_summary", {})
            if not validation_summary:
                test_validation_failed_instances.append(instance_id)
                continue

            total_tests = validation_summary.get("total_tests", 0)
            if total_tests != 3:
                test_validation_failed_instances.append(instance_id)
                continue
            
            # Check pre_execution_validation.validation_success == true
            pre_validation = prediction.get("pre_execution_validation", {})
            if not _validate_pre_execution_behavior(pre_validation):
                test_validation_failed_instances.append(instance_id)
                continue
            
            # All conditions met - this is a successful case
            successful_instances.append(instance_id)
                
        except (json.JSONDecodeError, KeyError, Exception) as e:
            # If there's any error parsing, consider it an other failure
            if problem_dir.name not in other_failed_instances:
                other_failed_instances.append(problem_dir.name)
    
    return successful_instances, test_gen_failed_instances, test_validation_failed_instances, other_failed_instances, empty_diff_instances


def save_classification_results(output_files_dir: str = "output_files") -> None:
    """
    Save classification results to output_files/count directory.

    Args:
        output_files_dir: Path to the output_files directory
    """
    # Get analysis results
    successful, test_gen_failed, test_validation_failed, other_failed, empty = analyze_agent_results(output_files_dir)

    # Create count directory
    count_dir = Path(output_files_dir) / "count"
    count_dir.mkdir(parents=True, exist_ok=True)

    # Combine all failed cases
    all_failed_instances = test_gen_failed + test_validation_failed + other_failed

    # Save each category to separate files
    categories = {
        "successful_cases.txt": successful,
        "test_gen_failed_cases.txt": test_gen_failed,
        "test_validation_failed_cases.txt": test_validation_failed,
        "all_failed_cases.txt": all_failed_instances,
        "empty_diff_cases.txt": empty
    }
    
    category_descriptions = {
        "successful_cases.txt": "Successful Cases (passed all validations)",
        "test_gen_failed_cases.txt": "Test Generation Failed Cases (no test_cases directory)",
        "test_validation_failed_cases.txt": "Test Validation Failed Cases (test_cases exist but validation failed)",
        "all_failed_cases.txt": "All Failed Cases (test_gen_failed + test_validation_failed + other_failed)",
        "empty_diff_cases.txt": "Empty Diff Cases (no model_patch generated)"
    }

    for filename, instance_ids in categories.items():
        file_path = count_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{category_descriptions[filename]}:\n")
            f.write(f"Total: {len(instance_ids)}\n\n")
            for instance_id in instance_ids:
                f.write(f"- {instance_id}\n")

    # Save summary statistics
    total_instances = len(successful) + len(test_gen_failed) + len(test_validation_failed) + len(other_failed) + len(empty)
    summary_data = {
        "summary": {
            "total_instances": total_instances,
            "successful": len(successful),
            "test_gen_failed": len(test_gen_failed),
            "test_validation_failed": len(test_validation_failed),
            "other_failed": len(other_failed),
            "empty_diff": len(empty)
        },
        "successful_instances": successful,
        "test_gen_failed_instances": test_gen_failed,
        "test_validation_failed_instances": test_validation_failed,
        "other_failed_instances": other_failed,
        "empty_diff_instances": empty
    }

    summary_file = count_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    console = Console()
    console.print(f"✅ Classification results have been saved to directory {count_dir}:")
    console.print(f"- successful_cases.txt: {len(successful)} items")
    console.print(f"- test_gen_failed_cases.txt: {len(test_gen_failed)} items")
    console.print(f"- test_validation_failed_cases.txt: {len(test_validation_failed)} items")
    console.print(f"- all_failed_cases.txt: {len(all_failed_instances)} items")
    console.print(f"- empty_diff_cases.txt: {len(empty)} items")
    console.print(f"- summary.json: complete statistics")


if __name__ == "__main__":
    save_classification_results()
