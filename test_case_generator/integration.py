"""Test Case Generation Integration Module

This module integrates test case generation into the SWE-bench workflow.
It runs an LLM agent to generate test cases and validates the results.
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict

from rich.console import Console

from prompts.test_case import TEST_CASE_GENERATION_INSTRUCTIONS, TEST_CASE_GENERATION_AGENT_CONFIG

# Docker commands
DOCKER_EXEC_CMD = ["docker", "exec", "-i"]


def generate_test_cases_in_container(
    container_id: str,
    problem_statement: str,
    problem_id: str = None,
    console: Console = None,
    workspace_path: Path = None,
    output_file: Path = None,
    quiet: bool = False
) -> Dict:
    """
    Generate test cases in the same container using a dedicated test case generation agent.

    Args:
        quiet: If True, suppress most console output to reduce verbosity
    """
    console = console or Console()

    # Create a quiet console to reduce output
    if quiet:
        import io
        from rich.console import Console as RichConsole
        quiet_console = RichConsole(file=io.StringIO(), stderr=False)
    else:
        quiet_console = console

    try:
        quiet_console.print("Setting up test generation in container...")

        # Format the prompt with the required parameters
        formatted_prompt = TEST_CASE_GENERATION_INSTRUCTIONS.format(
            location="/testbed",  # Container workspace path
            problem_statement=problem_statement,  # The problem to solve
            hints_text=""  # No additional hints for now
        )

        # Create a dedicated test case generation engine
        from tools.patch_agent import PatchEngine
        from tools.bash_tool import create_docker_bash_tool
        from tools.complete_tool import CompleteTool
        from simple_llm_client import create_simple_client
        from utils.workspace_manager import ProjectSpaceHandler
        import logging

        # Initialize required components
        client = create_simple_client("test_generation")  # Use dedicated configuration for test_generation
        workspace_manager = ProjectSpaceHandler(root=Path("/testbed"), container_workspace="/testbed")
        logger_for_agent_logs = logging.getLogger("test_case_generation")
        # Set log level to WARNING to reduce verbosity
        logger_for_agent_logs.setLevel(logging.WARNING)
        
        # Create patch engine with only bash and complete tools
        test_engine = PatchEngine(
            llm_client=client,
            workspace_manager=workspace_manager,
            console=quiet_console,  # Use quiet console
            activity_logger=logger_for_agent_logs,
            max_iterations=TEST_CASE_GENERATION_AGENT_CONFIG["max_turns"],
            max_response_tokens=TEST_CASE_GENERATION_AGENT_CONFIG["max_output_tokens_per_turn"],
            container_id=container_id
        )

        # Override the tools to only allow bash and complete
        test_engine._tool_registry._tool_instances = [
            create_docker_bash_tool(container=container_id, ask_user_permission=False),
            CompleteTool()
        ]

        try:
            quiet_console.print("Starting test generation agent...")
            start_time = time.time()
            
            # Run the patch engine with our test case generation prompt
            result = test_engine.run_impl({
                "instruction": formatted_prompt
            })
            
            test_generation_duration = time.time() - start_time

        except Exception as e:
            console.print(f"Error during agent execution: {e}")
            raise

        # Check if test cases were generated
        console.print("Checking for generated test cases...")
        test_cases_created = _check_test_cases(container_id, console)

        # Save the complete agent trajectory
        console.print("Saving agent trajectory...")
        try:
            # Get the complete dialog history from the agent
            dialog_messages = test_engine.dialog._message_lists

            # Format the trajectory for human reading
            trajectory_content = f"""Test Case Generation Agent Trajectory for {problem_statement}
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
AGENT CONFIGURATION
{'='*80}
Max Turns: {test_engine.max_iterations}
Max Output Tokens Per Turn: {test_engine.max_response_tokens}
Tools Available: {[tool.__class__.__name__ for tool in test_engine._tool_registry._tool_instances]}
Container ID: {getattr(test_engine, 'container_id', 'N/A')}

{'='*80}
COMPLETE CONVERSATION HISTORY
{'='*80}
"""
            
            # Process each turn in the conversation
            for turn_idx, turn_messages in enumerate(dialog_messages):
                is_user_turn = turn_idx % 2 == 0
                
                if is_user_turn:
                    trajectory_content += f"\n{'='*50} TURN {turn_idx//2 + 1} - USER {'='*50}\n"
                    for message in turn_messages:
                        if hasattr(message, 'text'):
                            trajectory_content += f"USER PROMPT: {message.text}\n"
                        elif hasattr(message, 'tool_output'):
                            trajectory_content += f"TOOL RESULT [{message.tool_name}]:\n{message.tool_output}\n"
                else:
                    trajectory_content += f"\n{'='*50} TURN {turn_idx//2 + 1} - ASSISTANT {'='*50}\n"
                    for message in turn_messages:
                        if hasattr(message, 'text'):
                            trajectory_content += f"ASSISTANT RESPONSE: {message.text}\n"
                        elif hasattr(message, 'tool_name'):
                            trajectory_content += f"TOOL CALL [{message.tool_name}]:\n"
                            trajectory_content += f"  Tool Call ID: {message.tool_call_id}\n"
                            trajectory_content += f"  Tool Input: {message.tool_input}\n"
                        elif hasattr(message, 'thinking'):
                            trajectory_content += f"THINKING: {message.thinking}\n"
            
            # Add token usage information if available
            if hasattr(test_engine.dialog, 'count_tokens'):
                try:
                    total_tokens = test_engine.dialog.count_tokens()
                    trajectory_content += f"\n{'='*80}\n"
                    trajectory_content += f"TOKEN USAGE\n"
                    trajectory_content += f"{'='*80}\n"
                    trajectory_content += f"Total Tokens Used: {total_tokens}\n"
                except Exception as e:
                    trajectory_content += f"Token counting error: {e}\n"
            
            # Save trajectory to output directory - use problem_id instead of problem_statement
            if problem_id:
                output_dir = Path(f"output_files/{problem_id}")
            else:
                # Fallback: try to extract from problem_statement if it contains __
                output_dir = Path(f"output_files/{problem_statement.split('__')[0] if '__' in problem_statement else 'unknown'}")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            trajectory_file = output_dir / f"{problem_id if problem_id else 'unknown'}_agent_trajectory.txt"
            with open(trajectory_file, "w", encoding="utf-8") as f:
                f.write(trajectory_content)
            
            console.print(f"PASS: Complete agent trajectory saved to {trajectory_file}")
            trajectory_saved = True
        except Exception as e:
            console.print(f"Warning: Failed to save agent trajectory: {e}")
            trajectory_saved = False
            trajectory_content = "Failed to capture trajectory"

        return {
            "test_generation_success": test_cases_created >= 3,
            "test_cases_generated": test_cases_created,
            "test_cases_dir": "/testbed/test_cases" if test_cases_created >= 3 else None,
            "error": None if test_cases_created >= 3 else f"Expected 3 test cases, found {test_cases_created}",
            "agent_trajectory_saved": trajectory_saved,
            "agent_trajectory_content": trajectory_content
        }

    except Exception as e:
        console.print(f"Error during test generation: {e}")
        return {
            "test_generation_success": False,
            "test_cases_generated": 0,
            "error": str(e)
        }


def _check_test_cases(container_id: str, console: Console) -> int:
    """Check how many test cases were generated. We expect 3 test cases."""
    check_cmd = [
        *DOCKER_EXEC_CMD, container_id,
        "bash", "-c", """
        cd /testbed
        if [ -d "test_cases" ]; then
            echo "DIRECTORY_EXISTS"
            ls -la test_cases/*.py 2>/dev/null | wc -l
        else
            echo "DIRECTORY_NOT_FOUND"
        fi
        """
    ]

    result = subprocess.run(check_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) >= 2 and output_lines[0] == "DIRECTORY_EXISTS":
            return int(output_lines[1])

    return 0


def save_test_generation_logs(
    logs_content: str,
    output_dir: Path,
    console: Console = None
) -> bool:
    """
    Save test generation logs to a separate file.
    
    Args:
        logs_content: The log content to save
        output_dir: Directory to save the logs
        console: Rich console for output
        
    Returns:
        True if saved successfully
    """
    console = console or Console()
    
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save test generation logs
        log_file = output_dir / "test_generation_logs.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(logs_content)
        
        console.print(f"PASS: Test generation logs saved to {log_file}")
        return True
        
    except Exception as e:
        console.print(f"Failed to save test generation logs: {e}")
        return False


def copy_test_cases_to_output(
    container_id: str,
    problem_id: str,
    console: Console = None
) -> bool:
    """
    Copy test cases from container to output directory.
    """
    console = console or Console()
    
    try:
        output_dir = Path(f"output_files/{problem_id}/test_cases")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        copy_cmd = [
            "docker", "cp",
            f"{container_id}:/testbed/test_cases/.",
            str(output_dir)
        ]
        
        result = subprocess.run(copy_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print(f"PASS: Test cases copied to {output_dir}")
            return True
        else:
            console.print(f"Failed to copy test cases: {result.stderr}")
            return False
            
    except Exception as e:
        console.print(f"Error copying test cases: {e}")
        return False


def validate_test_cases_against_original_code(
    container_id: str,
    console: Console = None
) -> Dict:
    """
    Validate that generated test cases have expected behavior against original code.
    """
    console = console or Console()
    
    try:
        console.print("Validating test cases against original code...")
        
        # Run each test individually and capture results
        test_results = {}

        for test_file in ["test_failure_scenario.py", "test_happy_path.py", "test_edge_case.py"]:
            console.print(f"Testing {test_file}...")

            test_cmd = [
                *DOCKER_EXEC_CMD, container_id,
                "bash", "-c", f"""
                cd /testbed
                source /opt/miniconda3/etc/profile.d/conda.sh
                conda activate testbed

                if [ -f "test_cases/{test_file}" ]; then
                    echo "=== Running {test_file} with pytest ==="
                    pytest "test_cases/{test_file}" -v -s 2>&1
                    exit_code=$?
                    echo "=== Exit code: $exit_code ==="
                    exit $exit_code
                else
                    echo "Test file not found: {test_file}"
                    exit 1
                fi
                """
            ]

            result = subprocess.run(test_cmd, capture_output=True, text=True)

            test_results[test_file] = {
                "status": "PASSED" if result.returncode == 0 else "FAILED",
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.returncode
            }

            console.print(f"  {test_file}: {test_results[test_file]['status']} (exit code: {result.returncode})")

            # Log detailed error information for failed tests
            if result.returncode != 0:
                console.print(f"    Error output: {result.stderr[:200]}...")
                console.print(f"    Stdout: {result.stdout[:200]}...")
        
        # Validate expected behavior
        validation_success = True
        validation_details = {}
        
        # Check failure scenario test
        if test_results["test_failure_scenario.py"]["status"] == "FAILED":
            console.print("PASS: test_failure_scenario.py: Correctly FAILED on original code")
            validation_details["failure_test"] = "PASS"
        else:
            console.print("test_failure_scenario.py: Should FAIL on original code but PASSED")
            validation_success = False
            validation_details["failure_test"] = "FAIL"

        # Check happy path test
        if test_results["test_happy_path.py"]["status"] == "PASSED":
            console.print("PASS: test_happy_path.py: Correctly PASSED on original code")
            validation_details["happy_path_test"] = "PASS"
        else:
            console.print("test_happy_path.py: Should PASS on original code but FAILED")
            validation_success = False
            validation_details["happy_path_test"] = "FAIL"

        # Check edge case test
        if test_results["test_edge_case.py"]["status"] == "PASSED":
            console.print("PASS: test_edge_case.py: Correctly PASSED on original code")
            validation_details["edge_case_test"] = "PASS"
        else:
            console.print("test_edge_case.py: Should PASS on original code but FAILED")
            validation_success = False
            validation_details["edge_case_test"] = "FAIL"

        if validation_success:
            console.print("All test cases meet expected behavior!")
        else:
            console.print("Some test cases do not meet expected behavior")
        
        return {
            "validation_success": validation_success,
            "test_results": test_results,
            "validation_details": validation_details,
            "test_output": result.stdout if 'result' in locals() else "",
            "error": None
        }
            
    except Exception as e:
        console.print(f"Error during test validation: {e}")
        return {
            "validation_success": False,
            "test_results": {},
            "validation_details": {},
            "test_output": "",
            "error": str(e)
        }


def regenerate_test_cases_with_improved_prompts(
    container_id: str,
    problem_statement: str,
    console: Console = None,
    workspace_path: Path = None,
    output_file: Path = None
) -> Dict:
    """
    Regenerate test cases with improved prompts based on validation results.
    """
    console = console or Console()
    
    console.print("Regenerating test cases with improved prompts...")
    
    try:
        # Call the original generation function
        result = generate_test_cases_in_container(
            container_id=container_id,
            problem_statement=problem_statement,
            console=console,
            workspace_path=workspace_path,
            output_file=output_file
        )
        
        # Ensure the result has the expected format
        if "test_generation_success" in result:
            return {
                "regeneration_success": result["test_generation_success"],
                "test_cases_generated": result.get("test_cases_generated", 0),
                "details": result
            }
        else:
            return {
                "regeneration_success": False,
                "test_cases_generated": 0,
                "details": result
            }
            
    except Exception as e:
        console.print(f"Error during test case regeneration: {e}")
        return {
            "regeneration_success": False,
            "test_cases_generated": 0,
            "details": {"error": str(e)}
        }
