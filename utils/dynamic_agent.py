"""
Dynamic Agent Executor for different retry strategies.

This module provides a unified interface for running agents with different
prompt templates in Docker containers.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from rich.console import Console

from utils.docker_utils import configure_project_space, terminate_runtime_pod
from utils.common import create_modification
from cli import main as cli_main


class PromptTemplate:
    """Prompt template definitions"""
    
    @staticmethod
    def get_basic_retry_prompt(workspace_root: str, problem_statement: str) -> str:
        """Basic retry using instruction.py + system_prompt.py"""
        from prompts.system_prompt import SYSTEM_PROMPT
        from prompts.instruction import INSTRUCTION_PROMPT
        
        base_prompt = SYSTEM_PROMPT.format(workspace_root=workspace_root) + "\n\n" + \
                     INSTRUCTION_PROMPT.format(location=workspace_root, pr_description=problem_statement)
        
        return base_prompt 
    
    @staticmethod
    def get_experience_retry_prompt(workspace_root: str, problem_statement: str, 
                                   current_trajectory: str, similar_case: Dict) -> str:
        """Experience-based retry using diff_retry.py + system_prompt.py"""
        from prompts.system_prompt import SYSTEM_PROMPT
        from prompts.diff_retry import INSTRUCTION_PROMPT
        
        # Build the complete compressed trajectory content of similar cases
        similar_trajectory_content = PromptTemplate._build_similar_trajectory_content(similar_case)
        
        base_prompt = SYSTEM_PROMPT.format(workspace_root=workspace_root) + "\n\n" + \
                     INSTRUCTION_PROMPT.format(
                         location=workspace_root,
                         pr_description=problem_statement,
                         current_compressed_trajectory=current_trajectory,
                         similar_instance_id=similar_case.get("instance", "unknown"),
                         similarity_score=similar_case.get("similarity_score", 0),
                         similarity_reasoning=similar_case.get("similarity_reasoning", "N/A"),
                         similar_case_strategy=similar_case.get("strategy", "N/A"),
                         similar_case_key_changes=similar_case.get("key_changes", "N/A"),
                         similar_compressed_trajectory=similar_trajectory_content
                     )
        
        return base_prompt
    
    @staticmethod
    def _build_similar_trajectory_content(similar_case: Dict) -> str:
        """Build the compressed trajectory content of similar cases, excluding the meta fields."""
        try:
            # Excluded meta fields
            meta_fields = {'instance', 'similarity_score', 'similarity_reasoning'}
            
            # Extract trajectory-related fields
            trajectory_fields = {}
            for key, value in similar_case.items():
                if key not in meta_fields and value not in [None, "", "N/A"]:
                    trajectory_fields[key] = value
            
            # If there is compressed trajectory data, format the output
            if trajectory_fields:
                import json
                return json.dumps(trajectory_fields, indent=2, ensure_ascii=False)
            else:
                return "No detailed trajectory information available"
                
        except Exception as e:
            return f"Error building trajectory content: {e}"


class DynamicAgentExecutor:
    """Unified agent executor with dynamic prompt switching"""
    
    def __init__(self, instance_id: str, workspace_base_path: Path, lock, semaphore):
        self.instance_id = instance_id
        self.workspace_base_path = workspace_base_path
        self.lock = lock
        self.semaphore = semaphore
        self.console = Console()
        
    def execute_agent(self, prompt_type: str, problem_statement: str, 
                     extra_data: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Execute agent with specified prompt type.
        
        Args:
            prompt_type: "basic_retry" or "experience_retry"
            problem_statement: The problem description
            extra_data: Extra data for experience_retry (trajectory, similar_case)
            
        Returns:
            Tuple of (generated_diff, logs_file_path)
        """
        logs_prefix = f"[bold blue]{self.instance_id}[/bold blue]"
        
        # Create workspace
        workspace_path = self.workspace_base_path / self.instance_id / f"retry_{prompt_type}"
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        container_id = None
        try:
            # Start Docker container (using new version's function)
            env, container_id = configure_project_space(workspace_path, self.instance_id, self.lock, self.semaphore)
            self.console.print(f"{logs_prefix} Docker container started: {container_id}")
            
            # Set environment variables
            for key, value in env.items():
                os.environ[key] = value
            
            # Generate prompt based on type
            workspace_root = str(workspace_path / self.instance_id)
            if prompt_type == "basic_retry":
                prompt = PromptTemplate.get_basic_retry_prompt(workspace_root, problem_statement)
            elif prompt_type == "experience_retry":
                if not extra_data or 'current_trajectory' not in extra_data or 'similar_case' not in extra_data:
                    raise ValueError("experience_retry requires current_trajectory and similar_case in extra_data")
                prompt = PromptTemplate.get_experience_retry_prompt(
                    workspace_root, problem_statement, 
                    extra_data['current_trajectory'], extra_data['similar_case']
                )
            else:
                raise ValueError(f"Unknown prompt_type: {prompt_type}")
            
            # Execute agent
            success, diff, logs_path = self._run_agent_in_container(
                container_id, workspace_root, prompt, prompt_type
            )
            
            if success and diff:
                self.console.print(f"{logs_prefix} ✅ Agent execution successful ({prompt_type})")
                return diff, logs_path
            else:
                self.console.print(f"{logs_prefix} ❌ Agent execution failed ({prompt_type})")
                return None, logs_path
                
        finally:
            # Clean up container (using new version's function)
            if container_id is not None:
                self.console.print(f"{logs_prefix} Stopping Docker container...")
                terminate_runtime_pod(container_id)
                self.console.print(f"{logs_prefix} Container stopped")
    
    def _run_agent_in_container(self, container_id: str, workspace_root: str, 
                               prompt: str, prompt_type: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Run agent in container with given prompt"""
        
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # Create CLI arguments
            output_file = Path(workspace_root).parent / f"agent_logs_{prompt_type}.txt"
            cli_args = [
                "cli.py",
                "--workspace", workspace_root,
                "--problem-statement", prompt,
                "--docker-container-id", container_id,
                "--use-container-workspace", "/testbed",
                "--minimize-stdout-logs",
                "--logs-path", str(output_file)
            ]
            
            # Replace sys.argv and run agent
            sys.argv = cli_args
            start_time = time.time()
            cli_main()
            duration = time.time() - start_time
            
            self.console.print(f"Agent completed in {duration:.2f}s ({prompt_type})")
            
            # Generate patch (using new version's function)
            diff = create_modification(None, container_id=container_id)
            
            return bool(diff and diff.strip()), diff, str(output_file) if output_file.exists() else None
            
        finally:
            # Restore sys.argv
            sys.argv = original_argv


def execute_basic_retry_agent(instance_id: str, problem_statement: str, 
                             workspace_base_path: Path, lock, semaphore) -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience function for basic retry (empty patch / no test cases).
    
    Returns:
        Tuple of (generated_diff, logs_file_path)
    """
    executor = DynamicAgentExecutor(instance_id, workspace_base_path, lock, semaphore)
    return executor.execute_agent("basic_retry", problem_statement)


def execute_experience_retry_agent(instance_id: str, problem_statement: str,
                                  current_trajectory: str, similar_case: Dict,
                                  workspace_base_path: Path, lock, semaphore) -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience function for experience-based retry (patch problem with similar case).
    
    Returns:
        Tuple of (generated_diff, logs_file_path)
    """
    executor = DynamicAgentExecutor(instance_id, workspace_base_path, lock, semaphore)
    extra_data = {
        'current_trajectory': current_trajectory,
        'similar_case': similar_case
    }
    return executor.execute_agent("experience_retry", problem_statement, extra_data)
