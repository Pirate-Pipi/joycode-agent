"""
Main script for patch generation and test verification

Extracts patch generation functionality from the original run_agent_on_swebench_problem.py,
and can optionally invoke test generation and validation functions.
"""

from functools import partial
import os
import logging
import threading
import sys
import json
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, Manager
import time
import numpy as np
from typing import Optional, Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from datasets import load_dataset

from utils.docker_utils import MAX_RUNTIME_PARALLELISM, configure_project_space, terminate_runtime_pod
from utils.common import create_modification
from utils.trajectory_recorder import create_execution_path_tracker
from cli import main as cli_main
from test import execute_tests_in_container, validate_diff_quality
import uuid


def get_dataset_name(dataset: str) -> str:
    """Get the dataset name for the specified dataset."""
    return {
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
        "lite": "princeton-nlp/SWE-bench_Lite",
    }[dataset]


def generate_tests_if_needed(
    problem_id: str,
    problem_statement: str,
    container_id: str,
    workspace_path: Path,
    console: Console,
    generate_tests: bool = True
) -> Tuple[Dict, Dict]:
    """
    Generate test cases if needed

    Args:
        problem_id: Problem ID
        problem_statement: Problem description
        container_id: Docker container ID
        workspace_path: Workspace path
        console: Console object
        generate_tests: Whether to generate tests

    Returns:
        tuple: (test generation result, pre-validation result)
    """
    if not generate_tests:
        console.print(f"[bold blue]{problem_id}[/bold blue] Skipping test generation (disabled)")
        return {"test_generation_success": False, "skipped": True}, {"validation_success": False, "skipped": True}

    # Check if test cases already exist
    output_dir = Path(f"output_files/{problem_id}")
    test_result_file = output_dir / "test_generation_result.json"

    if test_result_file.exists():
        console.print(f"[bold blue]{problem_id}[/bold blue] Found existing test generation results")
        try:
            with open(test_result_file, 'r') as f:
                existing_result = json.load(f)
            if existing_result.get("test_generation_success", False):
                console.print(f"[bold blue]{problem_id}[/bold blue] Using existing test cases")
                return existing_result.get("test_generation_result", {}), existing_result.get("pre_validation_result", {})
        except Exception as e:
            console.print(f"[bold blue]{problem_id}[/bold blue] Failed to load existing test results: {e}")

    # Generate new test cases
    console.print(f"[bold blue]{problem_id}[/bold blue] Generating new test cases...")

    try:
        from test_case_generator import (
            generate_test_cases_in_container,
            copy_test_cases_to_output,
            validate_test_cases_against_original_code
        )

        test_generation_result = generate_test_cases_in_container(
            container_id=container_id,
            problem_statement=problem_statement,
            problem_id=problem_id,
            console=console,
            workspace_path=workspace_path / problem_id,
            output_file=workspace_path / "test_generation_logs.txt",
            quiet=True  # Reduce verbose output
        )

        pre_validation_result = {"validation_success": False}

        if test_generation_result["test_generation_success"]:
            console.print(f"[bold blue]{problem_id}[/bold blue] Test cases generated successfully")

            # Pre-execute test cases for validation
            console.print(f"[bold blue]{problem_id}[/bold blue] Pre-executing test cases to validate expected behavior...")
            pre_validation_result = validate_test_cases_against_original_code(container_id, console)

            # Copy test cases to output directory
            copy_success = copy_test_cases_to_output(container_id, problem_id, console)
            if copy_success:
                console.print(f"[bold blue]{problem_id}[/bold blue] ✅ Test cases copied to output directory")

            # Save test generation result
            test_result_data = {
                "instance_id": problem_id,
                "test_generation_success": True,
                "test_generation_result": test_generation_result,
                "pre_validation_result": pre_validation_result,
                "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }

            output_dir.mkdir(parents=True, exist_ok=True)
            with open(test_result_file, "w") as f:
                json.dump(test_result_data, f, indent=2)
        else:
            console.print(f"[bold blue]{problem_id}[/bold blue] Test case generation failed")

        return test_generation_result, pre_validation_result

    except Exception as e:
        console.print(f"[bold blue]{problem_id}[/bold blue] Test case generation failed: {e}")
        return {"test_generation_success": False, "error": str(e)}, {"validation_success": False}


def run_patch_generation_with_test(
    problem_id: str,
    problem_statement: str,
    rollout_idx: int,
    workspace_base_path: Path,
    lock: threading.Lock,
    semaphore: threading.Semaphore,
    generate_tests: bool = True,
    validate_with_tests: bool = True
) -> tuple[str, float]:
    """
    Run patch generation, optionally generate and validate tests

    Args:
        problem_id: Problem ID
        problem_statement: Problem description
        rollout_idx: rollout index
        workspace_base_path: Workspace base path
        lock: Thread lock
        semaphore: Semaphore
        generate_tests: Whether to generate tests
        validate_with_tests: Whether to validate the patch with tests

    Returns:
        tuple: (generated diff, duration)
    """
    console = Console()
    logs_prefix = f"[bold blue]{problem_id}[/bold blue]"

    workspace_path = workspace_base_path / problem_id / f"rollout_{rollout_idx}"
    output_file = workspace_path / "agent_logs.txt"

    # Ensure workspace directory exists
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Create trajectory recorder
    trajectory_recorder = create_execution_path_tracker(problem_id, problem_statement)

    # Start Docker container
    container_id = None
    diff = ""

    try:
        env, container_id = configure_project_space(workspace_path, problem_id, lock, semaphore)
        console.print(f"{logs_prefix} Docker container started with ID: {container_id}")

        # Set environment variables
        for key, value in env.items():
            os.environ[key] = value

        # Generate test cases (if enabled)
        test_generation_result, pre_validation_result = generate_tests_if_needed(
            problem_id, problem_statement, container_id, workspace_path, console, generate_tests
        )

        # Record test generation stage
        trajectory_recorder.record_test_generation(test_generation_result, pre_validation_result)

        # Save original sys.argv
        original_argv = sys.argv.copy()

        # Create new sys.argv for cli.py
        cli_args = [
            "cli.py",
            "--workspace",
            str(workspace_path / problem_id),
            "--problem-statement",
            problem_statement,
            "--docker-container-id",
            container_id,
            "--use-container-workspace",
            "/testbed",
            "--minimize-stdout-logs",
        ]

        # If output file is specified, set the log path
        if output_file:
            cli_args.extend(["--logs-path", str(output_file)])

        # Replace sys.argv with our custom arguments
        sys.argv = cli_args

        # Now run the main agent to generate diff
        console.print(f"{logs_prefix} Starting agent run to generate diff...")
        start_time = time.time()
        cli_main()
        agent_duration = time.time() - start_time
        console.print(f"{logs_prefix} Agent run completed in {agent_duration:.2f}s.")

        # Restore original sys.argv
        sys.argv = original_argv

        # Generate patch from Docker container
        console.print(f"{logs_prefix} Generating patch from container {container_id}...")

        try:
            # First try generating patch from the container
            diff = create_modification(None, container_id=container_id)
            console.print(f"{logs_prefix} Patch generated successfully from container")
        except Exception as container_e:
            console.print(f"{logs_prefix} ⚠️ Failed to generate patch from container: {container_e}")

            # Fallback: try generating from host path
            repo_path = str(workspace_path / problem_id)
            console.print(f"{logs_prefix} Trying fallback: generating patch from host path {repo_path}...")

            try:
                diff = create_modification(repo_path)
                console.print(f"{logs_prefix} Patch generated successfully from host path")
            except (FileNotFoundError, ValueError) as e:
                console.print(f"{logs_prefix} ❌ Failed to generate patch from both container and host: {e}")
                diff = ""

        # Record diff generation stage
        trajectory_recorder.record_diff_generation(diff, agent_duration, output_file)

        # Save predictions.json
        with (workspace_path / "predictions.json").open("w") as f:
            json.dump(
                [
                    {
                        "instance_id": problem_id,
                        "model_name_or_path": "JoyCode-agent",
                        "model_patch": diff,
                    }
                ],
                f,
                indent=2,
            )

        # Execute test validation (if enabled and tests exist)
        if validate_with_tests and test_generation_result.get("test_generation_success", False):
            console.print(f"{logs_prefix} Running test validation on generated test cases...")
            try:
                # Run tests to validate diff quality
                _, test_results = execute_tests_in_container(container_id, console)

                # Validate diff quality
                validation_result = validate_diff_quality(test_results, console)
                test_validation_passed = validation_result.get("validation_passed", False)

                if test_validation_passed:
                    console.print(f"{logs_prefix} Test validation completed: PASS")
                else:
                    console.print(f"{logs_prefix} Test validation completed: FAIL")

                # Record test validation stage
                trajectory_recorder.record_test_validation(test_validation_passed, test_results, validation_result)

                # Update predictions.json with validation results
                predictions_data = json.loads((workspace_path / "predictions.json").read_text())
                predictions_data[0]["pre_execution_validation"] = pre_validation_result
                predictions_data[0]["post_execution_validation"] = {
                    "test_results": test_results,
                    "validation_summary": validation_result
                }
                with (workspace_path / "predictions.json").open("w") as f:
                    json.dump(predictions_data, f, indent=2)
            except Exception as e:
                console.print(f"{logs_prefix} Test validation failed: {e}")
        else:
            if not validate_with_tests:
                console.print(f"{logs_prefix} Skipping test validation (disabled)")
            else:
                console.print(f"{logs_prefix} Skipping test validation - no test cases generated")

        # Save important output files
        console.print(f"{logs_prefix} Saving important output files...")
        try:
            # Create output directory
            output_dir = Path(f"output_files/{problem_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save predictions.json
            predictions_source = workspace_path / "predictions.json"
            predictions_dest = output_dir / "predictions.json"
            if predictions_source.exists():
                shutil.copy2(predictions_source, predictions_dest)
                console.print(f"{logs_prefix} Predictions saved to {predictions_dest}")

            # Save agent_logs.txt
            agent_logs_source = workspace_path / "agent_logs.txt"
            agent_logs_dest = output_dir / "agent_logs.txt"
            if agent_logs_source.exists():
                shutil.copy2(agent_logs_source, agent_logs_dest)
                console.print(f"{logs_prefix} Agent logs saved to {agent_logs_dest}")

            # Set final result to the trajectory recorder
            final_result = {
                "model_patch": diff,
                "predictions_file_saved": predictions_dest.exists() if predictions_dest else False,
                "agent_logs_saved": agent_logs_dest.exists() if agent_logs_dest else False
            }
            trajectory_recorder.set_final_result(final_result)

            console.print(f"{logs_prefix} Key output files saved")
            console.print(f"{logs_prefix} Complete trajectory saved to: {trajectory_recorder.trajectory_file}")

        except Exception as e:
            console.print(f"{logs_prefix} Warning: Failed to save some output files: {e}")

    finally:
        # Stop and clean up Docker container
        if container_id is not None:
            console.print(f"{logs_prefix} Stopping Docker container...")
            terminate_runtime_pod(container_id)
            console.print(f"{logs_prefix} Docker container stopped")

    assert diff is not None
    return diff, agent_duration


def _save_case_id_lists(successful: List[str], failed: List[str], empty: List[str]):
    """
    Save the IDs of successful, failed, and empty-diff cases to text files under the output_files directory.

    Args:
        successful: List of IDs of successful cases
        failed: List of IDs of failed cases
        empty: List of IDs of empty-diff cases
    """
    try:
        # Ensure output_files directory exists
        output_files_dir = Path("output_files")
        output_files_dir.mkdir(parents=True, exist_ok=True)

        with open(output_files_dir / "successful_cases.txt", "w") as f:
            f.write("Successful Cases:\n")
            for case_id in successful:
                f.write(f"- {case_id}\n")

        with open(output_files_dir / "failed_cases.txt", "w") as f:
            f.write("Failed Cases:\n")
            for case_id in failed:
                f.write(f"- {case_id}\n")

        with open(output_files_dir / "empty_diff_cases.txt", "w") as f:
            f.write("Empty Diff Cases:\n")
            for case_id in empty:
                f.write(f"- {case_id}\n")

        console = Console()
        console.print("✅ Case ID lists have been saved to files under the output_files directory.")
        console.print("    - output_files/successful_cases.txt")
        console.print("    - output_files/failed_cases.txt")
        console.print("    - output_files/empty_diff_cases.txt")

    except Exception as e:
        console = Console()
        console.print(f"💥 Failed to save case ID lists: {e}")


def _build_experience_prompt(similar_case: Optional[Dict] = None) -> str:
    """
    Build experience prompts based on a similar case

    Args:
        similar_case: Compressed trajectory info of the similar case; if None, return an empty prompt

    Returns:
        Experience prompt string
    """
    if not similar_case:
        return ""

    instance_id = similar_case.get("instance", "unknown")
    strategy = similar_case.get("strategy", "N/A")
    key_changes = similar_case.get("key_changes", "N/A")
    similarity_score = similar_case.get("similarity_score", -1)

    experience_prompt = f"""
Based on the experience from high-quality similar case {instance_id} (similarity: {similarity_score}), please retry solving the current problem:

**Successful Strategy**: {strategy}
**Key Changes**: {key_changes}

Please refer to this successful case's approach, re-analyze the current problem and generate a fix.
"""
    return experience_prompt


def _retry_generate_diff_with_experience(
    instance_id: str,
    original_problem_statement: str,
    experience_prompt: str,
    workspace_base_path: Path,
    lock,
    semaphore
) -> tuple[Optional[str], Optional[str]]:
    """
    Regenerate diff based on experience prompts (no test generation)

    Args:
        instance_id: Instance ID
        original_problem_statement: Original problem description
        experience_prompt: Experience prompt
        workspace_base_path: Workspace base path
        lock: Thread lock
        semaphore: Semaphore

    Returns:
        tuple: (generated diff string, retry log file path); (None, None) if failed
    """
    try:
        import os
        console = Console()
        logs_prefix = f"[bold blue]{instance_id}[/bold blue]"

        # 🔥 Use the same workspace path structure, but mark as retry
        workspace_path = workspace_base_path / instance_id / "retry"

        # Ensure workspace directory exists
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Start Docker container (reuse main workflow's startup logic)
        container_id = None

        try:
            env, container_id = configure_project_space(workspace_path, instance_id, lock, semaphore)
            console.print(f"{logs_prefix} Docker container started with ID: {container_id}")

            # Set environment variables
            for key, value in env.items():
                os.environ[key] = value

            # Save original sys.argv
            original_argv = sys.argv.copy()

            # Add experience guidance to the full prompt template
            if experience_prompt:
                problem_statement = original_problem_statement + f"\n\n--- Experience Guidance from Successful Case ---\n{experience_prompt}"
            else:
                problem_statement = original_problem_statement + "\n\n--- Retry Mode ---\nThis is retry mode, please re-analyze the current problem and generate a fix."

            cli_args = [
                "cli.py",
                "--agent-purpose", "retry_agent",  # Use retry_agent-specific configuration
                "--workspace",
                str(workspace_path / instance_id),
                "--problem-statement",
                problem_statement,
                "--docker-container-id",
                container_id,
                "--use-container-workspace",
                "/testbed",
                "--minimize-stdout-logs",
            ]

            # Set logs path
            output_file = workspace_path / "retry_agent_logs.txt"
            cli_args.extend(["--logs-path", str(output_file)])

            # Replace sys.argv
            sys.argv = cli_args

            # Run the agent to generate diff
            start_time = time.time()
            cli_main()
            agent_duration = time.time() - start_time
            console.print(f"{logs_prefix} Agent run completed in {agent_duration:.2f}s.")

            # Restore original sys.argv
            sys.argv = original_argv

            # Generate patch - prefer from container
            console.print(f"{logs_prefix} Generating patch from container {container_id}...")
            try:
                diff = create_modification(None, container_id=container_id)
                console.print(f"{logs_prefix} Patch generated successfully from container")
            except Exception as container_e:
                console.print(f"{logs_prefix} ⚠️ Failed to generate patch from container: {container_e}")

                # Fallback to host path
                repo_path = str(workspace_path / instance_id)
                console.print(f"{logs_prefix} Trying fallback: generating patch from host path {repo_path}...")
                console.print(f"{logs_prefix} Debug: repo_path exists: {os.path.exists(repo_path)}")
                if os.path.exists(repo_path):
                    console.print(f"{logs_prefix} Debug: repo_path is symlink: {os.path.islink(repo_path)}")
                    if os.path.islink(repo_path):
                        real_path = os.path.realpath(repo_path)
                        console.print(f"{logs_prefix} Debug: real_path: {real_path}")
                        console.print(f"{logs_prefix} Debug: real_path exists: {os.path.exists(real_path)}")

                try:
                    diff = create_modification(repo_path)
                    console.print(f"{logs_prefix} Patch generated successfully from host path")
                except Exception as host_e:
                    console.print(f"{logs_prefix} ❌ Failed to generate patch from both container and host: {host_e}")
                    diff = None

            if diff:
                console.print(f"{logs_prefix} ✅ Retry diff generation succeeded, time: {agent_duration:.2f}s")
                return diff, str(output_file)
            else:
                console.print(f"{logs_prefix} ❌ Retry diff generation failed")
                return None, str(output_file) if output_file.exists() else None

        finally:
            # Stop Docker container
            if container_id is not None:
                console.print(f"{logs_prefix} Stopping Docker container...")
                terminate_runtime_pod(container_id)
                console.print(f"{logs_prefix} Docker container stopped")

    except Exception as e:
        console.print(f"{logs_prefix} 💥 Retry process error: {e}")
        return None, None


def _save_similar_case_match_record(failed_instance_id: str, similar_case: Optional[Dict]):
    """
    Save similar-case matching record (simplified)

    Args:
        failed_instance_id: Failed instance ID
        similar_case: Compressed trajectory info of similar case
    """
    try:
        console = Console()

        # Build simplified match record
        if similar_case:
            similarity_score = similar_case.get("similarity_score", -1)
            threshold_met = similarity_score >= 80

            match_record = {
                "failed_instance_id": failed_instance_id,
                "similar_instance_id": similar_case.get("instance", "unknown"),
                "similarity_score": similarity_score,
                "similarity_threshold_met": threshold_met,
                "similarity_reasoning": similar_case.get("similarity_reasoning", "N/A"),
                "similar_case_strategy": similar_case.get("strategy", "N/A"),
                "similar_case_key_changes": similar_case.get("key_changes", "N/A")
            }
            console.print(f"📋 Match record: {failed_instance_id} -> {similar_case.get('instance', 'unknown')}")
        else:
            match_record = {
                "failed_instance_id": failed_instance_id,
                "similar_instance_id": None,
                "similarity_score": -1,
                "similarity_threshold_met": False,
                "similarity_reasoning": "N/A",
                "similar_case_strategy": "N/A",
                "similar_case_key_changes": "N/A"
            }
            console.print(f"📋 Match record: {failed_instance_id} -> No similar case found")

        # Save to file
        match_record_file = Path(f"output_files/{failed_instance_id}/similar_case_match.json")
        match_record_file.parent.mkdir(parents=True, exist_ok=True)
        with open(match_record_file, 'w', encoding='utf-8') as f:
            json.dump(match_record, f, indent=2, ensure_ascii=False)

    except Exception as e:
        console.print(f"❌ Failed to save similar-case matching record {failed_instance_id}: {e}")


def _generate_similar_case_summary_report(retry_instances: List[str]):
    """
    Generate summary report of similar-case matching (simplified)

    Args:
        retry_instances: List of instances to retry
    """
    try:
        console = Console()
        console.print("📊 Generating summary report for similar-case matching...")

        match_records = []
        successful_matches = 0

        # Collect all match records
        for instance_id in retry_instances:
            match_file = Path(f"output_files/{instance_id}/similar_case_match.json")

            if match_file.exists():
                try:
                    with open(match_file, 'r', encoding='utf-8') as f:
                        match_record = json.load(f)

                    match_records.append(match_record)

                    if match_record.get("similar_instance_id"):
                        successful_matches += 1

                except Exception as e:
                    console.print(f"⚠️ Failed to read match record {instance_id}: {e}")

        # Save simplified summary report
        summary_data = {
            "total_retry_instances": len(retry_instances),
            "successful_matches": successful_matches,
            "failed_matches": len(retry_instances) - successful_matches,
            "match_records": match_records
        }

        # Ensure output_files directory exists
        output_files_dir = Path("output_files")
        output_files_dir.mkdir(parents=True, exist_ok=True)

        summary_file = output_files_dir / "similar_case_matches_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        console.print(f"📋 Similar-case matching summary report saved: {summary_file}")
        console.print(f"📊 Match stats: {successful_matches}/{len(retry_instances)} successful")

    except Exception as e:
        console.print(f"❌ Failed to generate similar-case matching summary report: {e}")


def _find_similar_case_for_retry(instance_id: str, successful: List[str]) -> Optional[Dict]:
    """
    Find a similar case for a single instance (pipeline mode)

    Args:
        instance_id: Instance ID
        successful: List of successful cases

    Returns:
        Similar case info or None
    """
    try:
        from search_zip.flow import find_similar_case
        return find_similar_case(instance_id, successful)
    except Exception as e:
        console = Console()
        console.print(f"❌ Error finding similar case {instance_id}: {e}")
        return None


def _get_problem_statement_for_retry(instance_id: str, all_tasks: List[Tuple]) -> str:
    """
    Get problem statement for an instance (pipeline mode)

    Args:
        instance_id: Instance ID
        all_tasks: List of all tasks

    Returns:
        Problem description string
    """
    for task in all_tasks:
        if task[1] == instance_id:  # task[1] is problem_id
            return task[2]  # task[2] is problem_statement

    # If not found, use default description
    return f"Fix the issue for {instance_id}"


def _copy_retry_logs_to_output(instance_id: str, retry_logs_path: str) -> bool:
    """
    Copy retry agent logs to the output_files directory

    Args:
        instance_id: Instance ID
        retry_logs_path: Path to the retry log file

    Returns:
        Whether copying succeeded
    """
    try:
        console = Console()

        if not retry_logs_path or not Path(retry_logs_path).exists():
            console.print(f"⚠️ Retry log file does not exist: {retry_logs_path}")
            return False

        # Target directory
        output_dir = Path(f"output_files/{instance_id}")
        if not output_dir.exists():
            console.print(f"⚠️ Output directory does not exist: {output_dir}")
            return False

        # Copy retry logs to output_files directory
        retry_logs_dest = output_dir / "agent_logs_retry.txt"
        shutil.copy2(retry_logs_path, retry_logs_dest)
        console.print(f"📁 Copied retry logs: {instance_id} -> agent_logs_retry.txt")

        # Also copy as the current agent_logs.txt (replace original)
        current_logs_dest = output_dir / "agent_logs.txt"
        shutil.copy2(retry_logs_path, current_logs_dest)
        console.print(f"📁 Updated current logs: {instance_id} -> agent_logs.txt")

        return True

    except Exception as e:
        console.print(f"❌ Failed to copy retry logs {instance_id}: {e}")
        return False


def _update_single_retry_result(instance_id: str, retry_diff: str) -> bool:
    """
    Immediately update a single retry result (file replacement and backup)

    Args:
        instance_id: Instance ID
        retry_diff: Diff generated by the retry

    Returns:
        Whether update succeeded
    """
    try:
        console = Console()

        # Find the corresponding output_files subdirectory
        output_dir = Path(f"output_files/{instance_id}")
        if not output_dir.exists():
            console.print(f"⚠️ Directory does not exist: {output_dir}")
            return False

        predictions_file = output_dir / "predictions.json"
        if not predictions_file.exists():
            console.print(f"⚠️ predictions.json does not exist: {instance_id}")
            return False

        # 1. Back up the original predictions.json file
        original_backup = output_dir / "predictions_original.json"
        if not original_backup.exists():
            # Rename the original file
            predictions_file.rename(original_backup)
            console.print(f"📁 Backed up original predictions.json: {instance_id}")

        # 1.1 Back up the original agent_logs.txt file
        agent_logs_file = output_dir / "agent_logs.txt"
        agent_logs_original = output_dir / "agent_logs_original.txt"
        if agent_logs_file.exists() and not agent_logs_original.exists():
            # Rename original agent logs
            agent_logs_file.rename(agent_logs_original)
            console.print(f"📁 Backed up original agent_logs.txt: {instance_id}")

        # 2. Read original data
        with open(original_backup, 'r') as f:
            original_data = json.load(f)

        # 3. Create retried data
        retry_data = original_data.copy()
        if retry_data and isinstance(retry_data, list) and len(retry_data) > 0:
            # Update the first prediction's model_patch
            retry_data[0]["model_patch"] = retry_diff

            # Add retry flags
            retry_data[0]["retry_success"] = True
            retry_data[0]["retry_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            retry_data[0]["original_patch"] = original_data[0].get("model_patch", "")

            # 4. Save retried predictions.json
            with open(predictions_file, 'w') as f:
                json.dump(retry_data, f, indent=2)

            # 5. Save retry record
            retry_record = {
                "instance_id": instance_id,
                "retry_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "original_patch": original_data[0].get("model_patch", ""),
                "retry_patch": retry_diff,
                "patch_changed": original_data[0].get("model_patch", "") != retry_diff
            }

            retry_record_file = output_dir / "predictions_retry.json"
            with open(retry_record_file, 'w') as f:
                json.dump(retry_record, f, indent=2)

            return True
        else:
            console.print(f"⚠️ Invalid predictions.json format: {instance_id}")
            return False

    except Exception as e:
        console.print(f"❌ Update failed {instance_id}: {e}")
        # If update fails, try to restore the original file
        try:
            if 'original_backup' in locals() and original_backup.exists() and not predictions_file.exists():
                original_backup.rename(predictions_file)
                console.print(f"🔄 Restored original file: {instance_id}")
        except:
            pass
        return False


def _execute_classified_retry(
    test_gen_failed: List[str],
    test_validation_failed: List[str],
    empty: List[str],
    successful: List[str],
    all_tasks: List[Tuple],
    workspace_base_path: Path,
    lock,
    semaphore,
    pool
) -> int:
    """
    Execute classified retry with judgment and conditional retry strategies.

    Args:
        test_gen_failed: List of test generation failed instances
        test_validation_failed: List of test validation failed instances
        empty: List of empty patch instances
        successful: List of successful instances
        all_tasks: List of all tasks
        workspace_base_path: Base path of workspace
        lock: Thread lock
        semaphore: Semaphore
        pool: Process pool

    Returns:
        Number of successful retries
    """
    from utils.dynamic_agent import execute_basic_retry_agent, execute_experience_retry_agent
    from utils.file_backup import backup_and_replace_instance_files
    from validate_retry_necessity import judge_failure_root_cause
    from search_zip.flow import find_similar_case

    console = Console()
    success_count = 0

    # 1. Handle empty patches and test generation failures with basic retry
    basic_retry_instances = empty + test_gen_failed
    if basic_retry_instances:
        console.print(f"🔄 Basic retry for {len(basic_retry_instances)} instances (empty patch + test gen failed)")

        # Execute basic retry in parallel
        retry_tasks = []
        for instance_id in basic_retry_instances:
            problem_statement = _get_problem_statement_for_retry(instance_id, all_tasks)
            retry_tasks.append((instance_id, problem_statement, workspace_base_path, lock, semaphore))

        # Use pool.starmap for parallel execution
        results = pool.starmap(execute_basic_retry_agent, retry_tasks)

        # Process results
        for instance_id, (new_diff, logs_path) in zip(basic_retry_instances, results):
            if new_diff:
                retry_type = "empty_patch" if instance_id in empty else "test_gen_failed"
                if backup_and_replace_instance_files(instance_id, new_diff, logs_path, f"basic_{retry_type}"):
                    success_count += 1
                    console.print(f"✅ Basic retry success: {instance_id}")
                else:
                    console.print(f"❌ Basic retry file update failed: {instance_id}")
            else:
                console.print(f"❌ Basic retry failed: {instance_id}")

    # 2. Handle test validation failures with judgment and conditional retry
    if test_validation_failed:
        console.print(f"🔍 Analyzing {len(test_validation_failed)} test validation failures...")

        # Judge each validation failure
        judgment_results = {}
        for instance_id in test_validation_failed:
            judgment = judge_failure_root_cause(instance_id, "output_files")
            judgment_results[instance_id] = judgment
            console.print(f"⚖️ {instance_id}: {judgment['root_cause']} (confidence: {judgment['confidence']:.2f})")

        # Separate by judgment
        test_problem_instances = [id for id, j in judgment_results.items() if j['root_cause'] == 'TEST']
        patch_problem_instances = [id for id, j in judgment_results.items() if j['root_cause'] == 'PATCH']

        # Handle TEST problems with basic retry
        if test_problem_instances:
            console.print(f"🔄 Basic retry for {len(test_problem_instances)} TEST problem instances")
            test_retry_tasks = []
            for instance_id in test_problem_instances:
                problem_statement = _get_problem_statement_for_retry(instance_id, all_tasks)
                test_retry_tasks.append((instance_id, problem_statement, workspace_base_path, lock, semaphore))

            test_results = pool.starmap(execute_basic_retry_agent, test_retry_tasks)

            for instance_id, (new_diff, logs_path) in zip(test_problem_instances, test_results):
                if new_diff:
                    if backup_and_replace_instance_files(instance_id, new_diff, logs_path, "test_problem"):
                        success_count += 1
                        console.print(f"✅ TEST problem retry success: {instance_id}")
                    else:
                        console.print(f"❌ TEST problem file update failed: {instance_id}")
                else:
                    console.print(f"❌ TEST problem retry failed: {instance_id}")

        # Handle PATCH problems with experience retry
        if patch_problem_instances:
            console.print(f"🧠 Experience retry for {len(patch_problem_instances)} PATCH problem instances")

            # Find similar cases and execute experience retry
            patch_retry_tasks = []
            patch_metadata = []

            for instance_id in patch_problem_instances:
                problem_statement = _get_problem_statement_for_retry(instance_id, all_tasks)

                # Find similar case
                similar_case = find_similar_case(instance_id, successful) if successful else None

                # Load current trajectory
                current_trajectory = _load_compressed_trajectory(instance_id)

                # Debug info
                console.print(f"🔍 {instance_id}: similar_case={'Yes' if similar_case else 'No'}, trajectory={'Yes' if current_trajectory else 'No'}")

                # Prepare task data
                if similar_case and current_trajectory:
                    patch_retry_tasks.append((
                        instance_id, problem_statement, current_trajectory,
                        similar_case, workspace_base_path, lock, semaphore
                    ))
                    patch_metadata.append({"instance_id": instance_id, "has_similar": True})
                else:
                    # Fallback to basic retry if no similar case or trajectory
                    patch_retry_tasks.append((instance_id, problem_statement, workspace_base_path, lock, semaphore))
                    patch_metadata.append({"instance_id": instance_id, "has_similar": False})

            # Execute experience retry for those with similar cases
            experience_tasks = [task for task, meta in zip(patch_retry_tasks, patch_metadata) if meta["has_similar"]]
            basic_fallback_tasks = [task for task, meta in zip(patch_retry_tasks, patch_metadata) if not meta["has_similar"]]

            if experience_tasks:
                experience_results = pool.starmap(execute_experience_retry_agent, experience_tasks)
                for i, (new_diff, logs_path) in enumerate(experience_results):
                    instance_id = experience_tasks[i][0]
                    if new_diff:
                        if backup_and_replace_instance_files(instance_id, new_diff, logs_path, "patch_experience"):
                            success_count += 1
                            console.print(f"✅ Experience retry success: {instance_id}")
                        else:
                            console.print(f"❌ Experience retry file update failed: {instance_id}")
                    else:
                        console.print(f"❌ Experience retry failed: {instance_id}")

            if basic_fallback_tasks:
                fallback_results = pool.starmap(execute_basic_retry_agent, basic_fallback_tasks)
                for i, (new_diff, logs_path) in enumerate(fallback_results):
                    instance_id = basic_fallback_tasks[i][0]
                    if new_diff:
                        if backup_and_replace_instance_files(instance_id, new_diff, logs_path, "patch_fallback"):
                            success_count += 1
                            console.print(f"✅ Fallback retry success: {instance_id}")
                        else:
                            console.print(f"❌ Fallback retry file update failed: {instance_id}")
                    else:
                        console.print(f"❌ Fallback retry failed: {instance_id}")

    return success_count


def _load_compressed_trajectory(instance_id: str) -> Optional[str]:
    """Load compressed trajectory for an instance."""
    try:
        trajectory_file = Path("output_files") / instance_id / "compressed_trajectory.txt"
        if trajectory_file.exists():
            with open(trajectory_file, 'r') as f:
                return f.read().strip()
    except Exception:
        pass
    return None


def _retry_with_concurrency(
    retry_instances: List[str],
    successful: List[str],
    all_tasks: List[Tuple],
    workspace_base_path: Path,
    lock,
    semaphore,
    pool
) -> int:
    """
    Retry using pipeline mode (similarity matching and retries run concurrently)

    Args:
        retry_instances: List of instance IDs to retry
        successful: List of successful case IDs
        all_tasks: List of all tasks
        workspace_base_path: Base path of workspace
        lock: Thread lock
        semaphore: Semaphore
        pool: Process pool

    Returns:
        Number of successfully retried cases
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    console = Console()
    console.print(f"🚀 Starting concurrent retries for {len(retry_instances)} cases...")
    console.print("💡 Concurrency mode: similarity matching and retries both run fully concurrently for maximum efficiency")

    success_count = 0

    if not successful:
        console.print("⚠️ No successful cases to reference; proceeding with no-experience retries")
        # No-experience retry: run concurrently with a process pool
        console.print(f"🔄 Using {pool._processes} processes for concurrent no-experience retries...")

        retry_tasks = []
        for instance_id in retry_instances:
            original_problem_statement = _get_problem_statement_for_retry(instance_id, all_tasks)
            experience_prompt = ""  # No-experience prompt
            retry_tasks.append((instance_id, original_problem_statement, experience_prompt, workspace_base_path, lock, semaphore))

        retry_results_list = pool.starmap(_retry_generate_diff_with_experience, retry_tasks)

        # Immediately save and replace
        for instance_id, (retry_diff, retry_logs_path) in zip(retry_instances, retry_results_list):
            # Save match record for no-experience retry (similar_case is None)
            _save_similar_case_match_record(instance_id, None)

            # Record retry stage to trajectory
            try:
                from utils.trajectory_recorder import load_trajectory_record
                trajectory = load_trajectory_record(instance_id)
                if trajectory:
                    recorder = create_execution_path_tracker(instance_id, trajectory.problem_statement)
                    recorder.trajectory = trajectory
                    original_diff = None  # No original diff reference for no-experience retry
                    recorder.record_retry_attempt("no_experience", original_diff, retry_diff, None)
            except Exception as e:
                console.print(f"⚠️ Failed to record retry trajectory {instance_id}: {e}")

            if retry_diff:
                # Copy retry logs
                logs_copied = False
                if retry_logs_path:
                    logs_copied = _copy_retry_logs_to_output(instance_id, retry_logs_path)

                if _update_single_retry_result(instance_id, retry_diff):
                    success_count += 1
                    logs_status = "✅" if logs_copied else "⚠️"
                    console.print(f"✅ Retry succeeded and saved: {instance_id} ({success_count}/{len(retry_instances)}) {logs_status}")
                else:
                    console.print(f"❌ Retry succeeded but saving failed: {instance_id}")
            else:
                console.print(f"❌ Retry failed: {instance_id}")

        console.print(f"🎯 No-experience retries completed: {success_count}/{len(retry_instances)} successful")

        # Generate summary report for similar-case matching (no-experience branch)
        try:
            _generate_similar_case_summary_report(retry_instances)
        except Exception as e:
            console.print(f"⚠️ Failed to generate similar-case summary report: {e}")

        return success_count

    # Experienced retry: use fully concurrent mode
    console.print(f"🔍 Starting fully concurrent processing: similarity matching and retries both concurrent...")
    console.print(f"📊 Concurrency config: similarity matching {pool._processes} threads, retries {pool._processes} processes")

    # Phase 1: run similarity matching concurrently
    console.print("🔍 Phase 1: run similarity matching concurrently...")
    similarity_results = {}

    max_similarity_workers = min(len(retry_instances), pool._processes)
    with ThreadPoolExecutor(max_workers=max_similarity_workers) as similarity_executor:
        # Submit all similarity matching tasks
        similarity_futures = {
            similarity_executor.submit(_find_similar_case_for_retry, instance_id, successful): instance_id
            for instance_id in retry_instances
        }

        # Collect all similarity matching results
        for future in as_completed(similarity_futures):
            instance_id = similarity_futures[future]
            try:
                similar_case = future.result()
                similarity_results[instance_id] = similar_case

                if similar_case:
                    similarity_score = similar_case.get('similarity_score', -1)
                    if similarity_score >= 80:
                        console.print(f"✅ Found high-quality similar case: {instance_id} -> {similar_case['instance']} (similarity: {similarity_score})")
                    else:
                        console.print(f"⚠️ Similar case quality insufficient: {instance_id} -> {similar_case['instance']} (similarity: {similarity_score} < 80)")
                else:
                    console.print(f"⚠️ No similar case found: {instance_id}")
            except Exception as e:
                console.print(f"❌ Similarity matching error {instance_id}: {e}")
                similarity_results[instance_id] = None

    console.print(f"✅ Similarity matching completed: {len(similarity_results)}/{len(retry_instances)} cases")

    # Phase 2: run all retry tasks concurrently based on similar-case results
    console.print("🔄 Phase 2: concurrently execute all retry tasks...")
    retry_tasks = []
    retry_metadata = []  # Store metadata for each task

    for instance_id in retry_instances:
        similar_case = similarity_results.get(instance_id)

        # Check whether to use experience
        use_experience = False
        retry_type = "no_experience"
        experience_prompt = ""

        if similar_case:
            similarity_score = similar_case.get('similarity_score', -1)
            if similarity_score >= 80:
                use_experience = True
                retry_type = "experienced"
                experience_prompt = _build_experience_prompt(similar_case)

        # Prepare retry task parameters
        original_problem_statement = _get_problem_statement_for_retry(instance_id, all_tasks)
        retry_tasks.append((instance_id, original_problem_statement, experience_prompt, workspace_base_path, lock, semaphore))

        # Save metadata for later processing
        retry_metadata.append({
            "instance_id": instance_id,
            "similar_case": similar_case if use_experience else None,
            "retry_type": retry_type,
            "use_experience": use_experience
        })

    # Concurrently execute all retry tasks
    console.print(f"🚀 Using {pool._processes} processes to run {len(retry_tasks)} retry tasks concurrently...")
    retry_results_list = pool.starmap(_retry_generate_diff_with_experience, retry_tasks)

    # Phase 3: process and save all retry results
    console.print("💾 Phase 3: process and save all retry results...")
    for (retry_diff, retry_logs_path), metadata in zip(retry_results_list, retry_metadata):
        instance_id = metadata["instance_id"]
        similar_case = metadata["similar_case"]
        retry_type = metadata["retry_type"]

        # Save similar-case matching record
        _save_similar_case_match_record(instance_id, similar_case)

        # Record retry stage to trajectory
        try:
            from utils.trajectory_recorder import load_trajectory_record
            trajectory = load_trajectory_record(instance_id)
            if trajectory:
                recorder = create_execution_path_tracker(instance_id, trajectory.problem_statement)
                recorder.trajectory = trajectory
                original_diff = None  # Obtain from original predictions
                try:
                    predictions_file = Path(f"output_files/{instance_id}/predictions_original.json")
                    if predictions_file.exists():
                        with open(predictions_file, 'r') as f:
                            original_data = json.load(f)
                            if original_data and len(original_data) > 0:
                                original_diff = original_data[0].get("model_patch", "")
                except:
                    pass
                recorder.record_retry_attempt(retry_type, original_diff, retry_diff, similar_case)
        except Exception as e:
            console.print(f"⚠️ Failed to record retry trajectory {instance_id}: {e}")

        # Save results
        if retry_diff:
            # Copy retry logs
            logs_copied = False
            if retry_logs_path:
                logs_copied = _copy_retry_logs_to_output(instance_id, retry_logs_path)

            if _update_single_retry_result(instance_id, retry_diff):
                success_count += 1
                logs_status = "✅" if logs_copied else "⚠️"
                experience_status = "🧠" if metadata["use_experience"] else "🆕"
                console.print(f"✅ Retry succeeded and saved: {instance_id} ({success_count}/{len(retry_instances)}) {experience_status}{logs_status}")
            else:
                console.print(f"❌ Retry succeeded but saving failed: {instance_id}")
        else:
            console.print(f"❌ Retry failed: {instance_id}")

    console.print(f"🎯 Concurrent retries completed: {success_count}/{len(retry_instances)} successful")

    # Generate summary report for similar-case matching
    try:
        _generate_similar_case_summary_report(retry_instances)
    except Exception as e:
        console.print(f"⚠️ Failed to generate similar-case summary report: {e}")

    return success_count


def main():
    """Main entry function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate patches and optionally test them on SWE-bench problems")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Optionally, specify the number of problems to process (useful for testing or partial runs)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to use for parallel processing of different examples",
    )
    parser.add_argument(
        "--generate-tests",
        action="store_true",
        default=True,
        help="Generate test cases before patch generation (default: True)",
    )
    parser.add_argument(
        "--no-generate-tests",
        action="store_false",
        dest="generate_tests",
        help="Skip test case generation",
    )
    parser.add_argument(
        "--validate-with-tests",
        action="store_true",
        default=True,
        help="Validate patches with generated test cases (default: True)",
    )
    parser.add_argument(
        "--no-validate-with-tests",
        action="store_false",
        dest="validate_with_tests",
        help="Skip test validation of patches",
    )
    parser.add_argument(
        "--problem-id",
        type=str,
        default=None,
        help="Generate patch for a specific problem ID only",
    )
    parser.add_argument(
        "--simple-mode",
        action="store_true",
        default=False,
        help="Simple mode: only generate patches without test generation and post-processing (trajectory compression, similarity matching, retry)",
    )
    parser.add_argument(
        "--enable-post-processing",
        action="store_true",
        default=False,
        help="Enable post-processing workflow: trajectory compression, similarity matching, and intelligent retry for failed cases",
    )

    args = parser.parse_args()

    # Handle mode logic
    if args.simple_mode and args.enable_post_processing:
        print("❌ Error: --simple-mode and --enable-post-processing cannot be used together")
        sys.exit(1)

    # Simple mode: only generate patch, no tests, no post-processing
    if args.simple_mode:
        args.generate_tests = False
        args.validate_with_tests = False
        args.enable_post_processing = False
        print("🚀 Simple mode: generate patch only; skip test generation and post-processing")

    # Full mode: generate tests and patch, then post-process
    elif args.enable_post_processing:
        args.generate_tests = True
        args.validate_with_tests = True
        print("🔥 Full mode: generate tests and patch; run full post-processing")

    # Default mode: decide based on current arguments
    else:
        args.enable_post_processing = False
        print("📝 Default mode: execute according to parameter configuration; no post-processing")

    # If tests are not generated, automatically disable test validation
    if not args.generate_tests:
        if args.validate_with_tests:  # Show notice only if user did not explicitly set --no-validate-with-tests
            print("ℹ️ Auto-disabled test validation since test generation is disabled")
        args.validate_with_tests = False

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize console
    console = Console()

    # Determine run mode description
    if args.simple_mode:
        mode_desc = "Simple Mode (patch only)"
        mode_color = "yellow"
    elif args.enable_post_processing:
        mode_desc = "Complete Mode (tests + patch + post-processing)"
        mode_color = "green"
    else:
        mode_desc = "Default Mode (based on parameters)"
        mode_color = "blue"

    # Display configuration
    console.print(Panel(
        f"""
Configuration:
- Mode: {mode_desc}
- Generate tests: {args.generate_tests}
- Validate with tests: {args.validate_with_tests}
- Enable post-processing: {args.enable_post_processing}
- Parallel processes: {args.num_processes}
        """,
        title="Patch Generation Configuration",
        border_style=mode_color
    ))

    if args.problem_id:
        # Generate patch for a single problem
        console.print(f"Generating patch for problem: {args.problem_id}")

        # Load dataset to find the problem
        console.print("Loading SWE-bench dataset...")
        path = "princeton-nlp___swe-bench_verified"
        swebench_dataset = load_dataset(path)["test"].to_pandas()

        # Locate the specified problem
        problem_data = swebench_dataset[swebench_dataset['instance_id'] == args.problem_id]
        if problem_data.empty:
            console.print(f"❌ Problem {args.problem_id} not found in dataset")
            return

        problem = problem_data.iloc[0]
        problem_statement = problem["problem_statement"]

        workspace_base_path = Path(f"/tmp/workspace/{uuid.uuid4().hex[:8]}").resolve()
        console.print(f"Workspace base path: {workspace_base_path}")

        # Generate patch
        diff, duration = run_patch_generation_with_test(
            args.problem_id,
            problem_statement,
            0,  # rollout_idx
            workspace_base_path,
            threading.Lock(),
            threading.Semaphore(MAX_RUNTIME_PARALLELISM),
            args.generate_tests,
            args.validate_with_tests
        )

        if diff:
            console.print(f"✅ Patch generation completed successfully for {args.problem_id}")
            console.print(f"⏱️ Duration: {duration:.2f}s")
        else:
            console.print(f"❌ Patch generation failed for {args.problem_id}")

        return

    # Batch generate patches (original logic)
    console.print("Loading SWE-bench dataset...")
    path = "princeton-nlp___swe-bench_verified"
    with open('instance_id.txt', 'r', encoding='utf-8') as f:
        test_instance_id_lst = [line.strip() for line in f]

    swebench_dataset = load_dataset(path)["test"].to_pandas()
    if test_instance_id_lst:
        swebench_dataset = swebench_dataset[swebench_dataset['instance_id'].isin(test_instance_id_lst)]

    # Get number of examples to run
    examples = swebench_dataset
    num_examples = args.num_examples if args.num_examples is not None else len(examples)
    console.print(
        f"Running on {num_examples} examples from the dataset."
    )
    console.print(
        f"We will generate 1 solution for each example with parallelism of {args.num_processes}."
    )

    # Print all example IDs to process
    console.print(
        "Selected examples:",
        "\n - " + "\n - ".join(examples.iloc[:num_examples]["instance_id"].tolist()),
    )

    # Get workspace base directory
    workspace_base_path = Path(f"/tmp/workspace/{uuid.uuid4().hex[:8]}").resolve()
    console.print(f"Workspace base path: {workspace_base_path}")

    output_path = f"pre-ensemble_results.jsonl"

    # Scan the output_files directory to find already generated data
    existing_problems = set()
    output_files_dir = Path("output_files")
    if output_files_dir.exists():
        for problem_dir in output_files_dir.iterdir():
            if problem_dir.is_dir():
                # Check for predictions.json file
                predictions_file = problem_dir / "predictions.json"
                if predictions_file.exists():
                    existing_problems.add(problem_dir.name)
        console.print(f"Found {len(existing_problems)} existing problems in output_files")

    # Filter the dataset to process only those not yet generated
    if existing_problems:
        original_count = len(examples)
        examples = examples[~examples["instance_id"].isin(existing_problems)]
        filtered_count = len(examples)
        console.print(f"Filtered dataset: {original_count} -> {filtered_count} problems (skipped {len(existing_problems)} existing)")

        # Update num_examples to the filtered count
        num_examples = min(filtered_count, num_examples) if args.num_examples else filtered_count

        if filtered_count == 0:
            console.print("All problems have already been processed!")
            return

    # Prepare all tasks
    all_tasks = []
    console.print(f"🔍 Creating tasks for {num_examples} examples...")
    for i in range(num_examples):
        problem = examples.iloc[i]
        problem_id = problem["instance_id"]
        problem_statement = problem["problem_statement"]
        console.print(f"   Example {i}: {problem_id}")
        all_tasks.append((i, problem_id, problem_statement, 0))

    console.print(f"📊 Created {len(all_tasks)} total tasks for {num_examples} examples")

    # Process all tasks in parallel
    all_results = {}

    with Manager() as manager:
        lock = manager.Lock()
        semaphore = manager.Semaphore(MAX_RUNTIME_PARALLELISM)

        with Pool(processes=args.num_processes) as pool:
            # Run all tasks in parallel
            results = pool.starmap(
                partial(
                    run_patch_generation_with_test,
                    workspace_base_path=workspace_base_path,
                    lock=lock,
                    semaphore=semaphore,
                    generate_tests=args.generate_tests,
                    validate_with_tests=args.validate_with_tests,
                ),
                [(task[1], task[2], task[3]) for task in all_tasks]
            )

            # Collect results
            for task, result in zip(all_tasks, results):
                example_idx, problem_id, problem_statement, _ = task
                diff, agent_duration = result

                all_results[example_idx] = {
                    "problem_id": problem_id,
                    "problem_statement": problem_statement,
                    "diff": diff,
                    "agent_duration": agent_duration,
                }

            # Process results and save
            all_diff_data = []
            for i in range(num_examples):
                if i in all_results:
                    result = all_results[i]

                    diff_data = {
                        "id": result["problem_id"],
                        "instruction": result["problem_statement"],
                        "diff": result["diff"],
                        "agent_duration": result["agent_duration"],
                    }
                    all_diff_data.append(diff_data)
                    console.print(f"Completed example {i + 1}/{num_examples}: {result['problem_id']}")
                else:
                    console.print(f"Error: No results for example {i + 1}")

            # Save results
            with open(output_path, "w") as f:
                for diff_data in all_diff_data:
                    f.write(json.dumps(diff_data) + "\n")

            all_durations = [d["agent_duration"] for d in all_diff_data]
            # Print latency statistics
            if len(all_durations) > 0:
                console.print(f"Agent latency min: {np.min(all_durations)}")
                console.print(f"Agent latency at 25perc: {np.percentile(all_durations, 25)}")
                console.print(f"Agent latency at median: {np.median(all_durations)}")
                console.print(f"Agent latency at 75perc: {np.percentile(all_durations, 75)}")
                console.print(f"Agent latency max: {np.max(all_durations)}")

            console.print(f"\nAll examples processed. Results saved to {output_path}")

            # 🔥 Post-processing - trajectory compression and retry for failed cases (only when enabled)
            if args.enable_post_processing:
                console.print("\n🚀 Start post-processing: trajectory compression and retries for failed cases...")

                try:
                    # 1. Wait for all files to be saved, then analyze the current results
                    console.print("⏳ Waiting for files to be saved...")
                    time.sleep(2)  # Give the filesystem some time

                    from utils.count import analyze_agent_results
                    successful, test_gen_failed, test_validation_failed, other_failed, empty = analyze_agent_results()

                    # Combine failed cases
                    failed = test_gen_failed + test_validation_failed + other_failed

                    console.print(f"📊 Result analysis: {len(successful)}✅ {len(failed)}❌ {len(empty)}🚫")

                    # Save case ID lists to the current directory
                    _save_case_id_lists(successful, failed, empty)

                    if successful:
                        console.print(f"✅ Successful cases: {', '.join(successful[:5])}{'...' if len(successful) > 5 else ''}")
                    if failed:
                        console.print(f"❌ Failed cases: {', '.join(failed[:5])}{'...' if len(failed) > 5 else ''}")
                    if empty:
                        console.print(f"🚫 Empty diff cases: {', '.join(empty[:5])}{'...' if len(empty) > 5 else ''}")

                    # 2. Compress all trajectories
                    console.print("\n🔍 Start compressing all trajectories...")
                    from search_zip.flow import process_trajectories_and_compress
                    compression_success = process_trajectories_and_compress()

                    if compression_success:
                        console.print("✅ Trajectory compression completed")

                        # Record trajectory compression result for each successful case
                        for success_id in successful:
                            try:
                                from utils.trajectory_recorder import load_trajectory_record
                                trajectory = load_trajectory_record(success_id)
                                if trajectory:
                                    recorder = create_execution_path_tracker(success_id, trajectory.problem_statement)
                                    recorder.trajectory = trajectory
                                    recorder.record_trajectory_compression({"compression_success": True})
                            except Exception as e:
                                console.print(f"⚠️ Failed to record trajectory compression {success_id}: {e}")

                        # 3. New classified retry logic
                        retry_needed = test_gen_failed + test_validation_failed + empty
                        if retry_needed:
                            console.print(f"\n🔄 Starting classified retries for {len(retry_needed)} cases...")

                            # Execute classified retry
                            success_count = _execute_classified_retry(
                                test_gen_failed, test_validation_failed, empty,
                                successful, all_tasks, workspace_base_path,
                                lock, semaphore, pool
                            )

                            console.print(f"💾 Classified retries completed: {success_count}/{len(retry_needed)} cases successfully retried and saved")
                        else:
                            console.print("✅ All cases succeeded; no retries needed")
                    else:
                        console.print("❌ Trajectory compression failed; skipping retry process")

                except Exception as e:
                    console.print(f"💥 Post-processing error: {e}")
                    console.print("⚠️ Continue with the original flow...")

                console.print("\n🎯 Post-processing completed!")
            else:
                console.print("\n📝 Skip post-processing (flag --enable-post-processing not enabled)")

    console.print(Panel(
        f"""
Now you have generated solutions (1 per problem) for {num_examples} problems.

You can manually analyze results by looking into the workspace directory: {workspace_base_path}. You'll be interested to look at files like:
- agent_logs.txt: The logs from the agent
- predictions.json: The diff generated by the agent

Note: Evaluation functionality has been removed to comply with competition requirements.

The user can now analyze the results:
- Review individual problem solutions in the output_files directory
- Examine the generated patches and test results

Results saved to: {output_path}

Configuration used:
- Mode: {mode_desc}
- Test generation: {args.generate_tests}
- Test validation: {args.validate_with_tests}
- Post-processing: {args.enable_post_processing}
        """,
        title="Patch Generation Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
