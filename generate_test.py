"""
Script dedicated to generating unit tests

This script extracts the unit test generation functionality from
run_agent_on_swebench_problem.py and can be run independently to
generate test cases for a specified problem.
"""

import os
import logging
import threading
import sys
import json
import argparse
import shutil
import time
from pathlib import Path
from multiprocessing import Pool, Manager
from functools import partial
from typing import Optional, Dict, List, Tuple, Any

from rich.console import Console
from rich.panel import Panel
from datasets import load_dataset

from utils.docker_utils import MAX_DOCKER_CONCURRENCY, setup_workspace, stop_container
from utils.trajectory_recorder import create_trajectory_recorder
from test_case_generator import (
    generate_test_cases_in_container,
    copy_test_cases_to_output,
    validate_test_cases_against_original_code,
    regenerate_test_cases_with_improved_prompts
)
import uuid


def get_dataset_name(dataset: str) -> str:
    """Get the dataset name for the specified dataset."""
    return {
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
        "lite": "princeton-nlp/SWE-bench_Lite",
    }[dataset]


def generate_test_for_single_problem(
    problem_id: str,
    problem_statement: str,
    rollout_idx: int,
    workspace_base_path: Path,
    lock: threading.Lock,
    semaphore: threading.Semaphore,
) -> tuple[bool, Dict]:
    """
    Generate test cases for a single problem

    Args:
        problem_id: Problem ID
        problem_statement: Problem statement
        rollout_idx: Rollout index
        workspace_base_path: Workspace base path
        lock: Thread lock
        semaphore: Semaphore

    Returns:
        tuple: (success, test generation result)
    """
    console = Console()
    logs_prefix = f"[bold blue]{problem_id}[/bold blue]"

    workspace_path = workspace_base_path / problem_id / f"rollout_{rollout_idx}"

    # Ensure workspace directory exists
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Create trajectory recorder
    trajectory_recorder = create_trajectory_recorder(problem_id, problem_statement)

    # Start Docker container
    container_id = None
    test_generation_success = False
    test_generation_result = {}
    pre_validation_result = {}

    try:
        env, container_id = setup_workspace(workspace_path, problem_id, lock, semaphore)
        console.print(f"{logs_prefix} Docker container started with ID: {container_id}")

        # Set environment variables
        for key, value in env.items():
            os.environ[key] = value

        # Generate test cases
        console.print(f"{logs_prefix} Generating test cases based on original code...")
        try:
            test_generation_result = generate_test_cases_in_container(
                container_id=container_id,
                problem_statement=problem_statement,
                problem_id=problem_id,
                console=console,
                workspace_path=workspace_path / problem_id,
                output_file=workspace_path / "test_generation_logs.txt",
                quiet=True  # Reduce verbose output
            )

            if test_generation_result["test_generation_success"]:
                console.print(f"{logs_prefix} Test cases generated successfully")
                console.print(f"{logs_prefix} Generated {test_generation_result.get('test_cases_generated', 0)} test cases")

                # Pre-execute tests for validation
                console.print(f"{logs_prefix} Pre-executing test cases to validate expected behavior...")
                pre_validation_result = validate_test_cases_against_original_code(container_id, console)

                # Record test generation stage
                trajectory_recorder.record_test_generation(test_generation_result, pre_validation_result)

                if pre_validation_result["validation_success"]:
                    console.print(f"{logs_prefix} ✅ All test cases meet expected behavior!")
                else:
                    console.print(f"{logs_prefix} ⚠️ Some test cases do not meet expected behavior")
                    console.print(f"{logs_prefix} ℹ️ Test cases will be handled by the agent in the next phase")

                # Copy test cases to output directory
                copy_success = copy_test_cases_to_output(container_id, problem_id, console)
                if copy_success:
                    console.print(f"{logs_prefix} ✅ Test cases copied to output directory")
                else:
                    console.print(f"{logs_prefix} ⚠️ Warning: Failed to copy test cases from container")

                # Save test generation logs
                from test_case_generator.integration import save_test_generation_logs
                test_logs_content = f"""
Test Case Generation Logs for {problem_id}
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

Test Generation Result: {test_generation_result}
Pre-Execution Validation Result: {pre_validation_result}
"""

                save_test_generation_logs(
                    test_logs_content,
                    Path(f"output_files/{problem_id}"),
                    console
                )

                test_generation_success = True

            else:
                console.print(f"{logs_prefix} Test case generation failed")
                pre_validation_result = {"validation_success": False}

                # Record failed test generation stage
                trajectory_recorder.record_test_generation(test_generation_result, pre_validation_result)

        except Exception as e:
            console.print(f"{logs_prefix} Test case generation failed: {e}")
            test_generation_result = {"test_generation_success": False, "error": str(e)}
            pre_validation_result = {"validation_success": False}

            # Record exception case for test generation stage
            trajectory_recorder.record_test_generation(test_generation_result, pre_validation_result)

        # Save test generation result to JSON file
        test_result_data = {
            "instance_id": problem_id,
            "test_generation_success": test_generation_success,
            "test_generation_result": test_generation_result,
            "pre_validation_result": pre_validation_result,
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with (workspace_path / "test_generation_result.json").open("w") as f:
            json.dump(test_result_data, f, indent=2)

        # Save important files to output directory
        console.print(f"{logs_prefix} Saving test generation output files...")
        try:
            output_dir = Path(f"output_files/{problem_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save test generation result
            test_result_dest = output_dir / "test_generation_result.json"
            shutil.copy2(workspace_path / "test_generation_result.json", test_result_dest)
            console.print(f"{logs_prefix} Test generation result saved to {test_result_dest}")

            # Save test generation logs
            test_logs_source = workspace_path / "test_generation_logs.txt"
            if test_logs_source.exists():
                test_logs_dest = output_dir / "test_generation_logs.txt"
                shutil.copy2(test_logs_source, test_logs_dest)
                console.print(f"{logs_prefix} Test generation logs saved to {test_logs_dest}")

            # Set final result to trajectory recorder
            final_result = {
                "test_generation_success": test_generation_success,
                "test_result_file_saved": test_result_dest.exists(),
                "test_logs_saved": (output_dir / "test_generation_logs.txt").exists()
            }
            trajectory_recorder.set_final_result(final_result)

            console.print(f"{logs_prefix} Test generation trajectory saved to: {trajectory_recorder.trajectory_file}")

        except Exception as e:
            console.print(f"{logs_prefix} Warning: Failed to save some test generation files: {e}")

    finally:
        # Stop and clean up Docker container
        if container_id is not None:
            console.print(f"{logs_prefix} Stopping Docker container...")
            stop_container(container_id)
            console.print(f"{logs_prefix} Docker container stopped")

    return test_generation_success, {
        "test_generation_result": test_generation_result,
        "pre_validation_result": pre_validation_result
    }


def main():
    """Main entry function"""
    parser = argparse.ArgumentParser(description="Generate test cases for SWE-bench problems")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Optionally, specify the number of examples to run on",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to use for parallel processing of different examples",
    )
    parser.add_argument(
        "--problem-id",
        type=str,
        default=None,
        help="Generate test for a specific problem ID only",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    console = Console()

    if args.problem_id:
        # Generate tests for a single problem
        console.print(f"Generating test cases for problem: {args.problem_id}")

        # Load dataset and locate the target problem
        console.print("Loading SWE-bench dataset...")
        path = "princeton-nlp___swe-bench_verified"
        swebench_dataset = load_dataset(path)["test"].to_pandas()

        # Find specified problem
        problem_data = swebench_dataset[swebench_dataset['instance_id'] == args.problem_id]
        if problem_data.empty:
            console.print(f"❌ Problem {args.problem_id} not found in dataset")
            return

        problem = problem_data.iloc[0]
        problem_statement = problem["problem_statement"]

        workspace_base_path = Path(f"/tmp/workspace/{uuid.uuid4().hex[:8]}").resolve()
        console.print(f"Workspace base path: {workspace_base_path}")

        # Generate tests
        success, result = generate_test_for_single_problem(
            args.problem_id,
            problem_statement,
            0,  # rollout_idx
            workspace_base_path,
            threading.Lock(),
            threading.Semaphore(MAX_DOCKER_CONCURRENCY)
        )

        if success:
            console.print(f"✅ Test generation completed successfully for {args.problem_id}")
        else:
            console.print(f"❌ Test generation failed for {args.problem_id}")

        return

    # Batch test generation (legacy path)
    console.print("Loading SWE-bench dataset...")
    path = "princeton-nlp___swe-bench_verified"
    with open('instance_id.txt', 'r', encoding='utf-8') as f:
        test_instance_id_lst = [line.strip() for line in f]

    swebench_dataset = load_dataset(path)["test"].to_pandas()
    if test_instance_id_lst:
        swebench_dataset = swebench_dataset[swebench_dataset['instance_id'].isin(test_instance_id_lst)]

    # Use full dataset (sharding removed)
    examples = swebench_dataset

    # Determine number of examples to run
    num_examples = args.num_examples if args.num_examples is not None else len(examples)
    console.print(
        f"Generating test cases for {num_examples} examples."
    )

    # List example IDs to process
    console.print(
        "Selected examples:",
        "\n - " + "\n - ".join(examples.iloc[:num_examples]["instance_id"].tolist()),
    )

    # Get workspace base directory
    workspace_base_path = Path(f"/tmp/workspace/{uuid.uuid4().hex[:8]}").resolve()
    console.print(f"Workspace base path: {workspace_base_path}")

    # Prepare all tasks
    all_tasks = []
    console.print(f"🔍 Creating test generation tasks for {num_examples} examples...")
    for i in range(num_examples):
        problem = examples.iloc[i]
        problem_id = problem["instance_id"]
        problem_statement = problem["problem_statement"]
        console.print(f"   Example {i}: {problem_id}")
        all_tasks.append((i, problem_id, problem_statement, 0))

    console.print(f"📊 Created {len(all_tasks)} test generation tasks")

    # Process all tasks in parallel
    all_results = {}

    with Manager() as manager:
        lock = manager.Lock()
        semaphore = manager.Semaphore(MAX_DOCKER_CONCURRENCY)

        with Pool(processes=args.num_processes) as pool:
            # Run all tasks in parallel
            results = pool.starmap(
                partial(
                    generate_test_for_single_problem,
                    workspace_base_path=workspace_base_path,
                    lock=lock,
                    semaphore=semaphore,
                ),
                [(task[1], task[2], task[3]) for task in all_tasks]
            )

            # Collect results
            for task, result in zip(all_tasks, results):
                example_idx, problem_id, problem_statement, _ = task
                success, test_data = result

                all_results[example_idx] = {
                    "problem_id": problem_id,
                    "problem_statement": problem_statement,
                    "test_generation_success": success,
                    "test_data": test_data
                }

            # Process results and save
            output_path = "test_generation_results.jsonl"
            all_test_data = []

            for i in range(num_examples):
                if i in all_results:
                    result = all_results[i]

                    test_data = {
                        "id": result["problem_id"],
                        "instruction": result["problem_statement"],
                        "test_generation_success": result["test_generation_success"],
                        "test_data": result["test_data"]
                    }
                    all_test_data.append(test_data)
                    console.print(f"Completed test generation {i + 1}/{num_examples}: {result['problem_id']}")
                else:
                    console.print(f"Error: No test generation results for example {i + 1}")

            # Save results
            with open(output_path, "w") as f:
                for test_data in all_test_data:
                    f.write(json.dumps(test_data) + "\n")

            console.print(f"\nAll test generation completed. Results saved to {output_path}")

    message_lines = [
        f"Test generation completed for {num_examples} problems.",
        "",
        f"Results saved to: {output_path}",
        "",
        "Individual test cases and logs are saved in the output_files/ directory.",
        "Each problem has its own subdirectory with:",
        "- test_generation_result.json: Generation results and validation status",
        "- test_generation_logs.txt: Detailed generation logs",
        "- test_*.py: Generated test case files (if successful)",
        "",
        "You can now use these test cases with generate_patch_test.py to validate patches.",
    ]
    console.print(Panel(
        "\n".join(message_lines),
        title="Test Generation Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
