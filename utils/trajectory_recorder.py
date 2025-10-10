"""
Trajectory Recorder - Manage the complete execution flow records for each instance

This module records the complete execution trajectory for each SWE-bench instance, including:
- Test case generation stage
- Diff generation stage
- Trajectory compression stage
- Retry stage (if needed)

It excludes timestamps and similar metadata; only core trajectory data is retained.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from rich.console import Console


@dataclass
class StageRecord:
    """Record for a single stage"""
    stage_name: str
    success: bool
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class TrajectoryRecord:
    """Complete trajectory record"""
    instance_id: str
    problem_statement: str
    stages: List[StageRecord]
    final_result: Optional[Dict[str, Any]] = None


class TrajectoryRecorder:
    """Trajectory Recorder"""
    
    def __init__(self, instance_id: str, problem_statement: str, output_dir: Optional[Path] = None):
        """
        Initialize the trajectory recorder
        
        Args:
            instance_id: Instance ID
            problem_statement: Problem statement
            output_dir: Output directory, defaults to output_files/{instance_id}
        """
        self.instance_id = instance_id
        self.problem_statement = problem_statement
        self.output_dir = output_dir or Path(f"output_files/{instance_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trajectory record
        self.trajectory = TrajectoryRecord(
            instance_id=instance_id,
            problem_statement=problem_statement,
            stages=[]
        )
        
        self.console = Console()
        self.logger = logging.getLogger(f"trajectory.{instance_id}")
        
        # Trajectory file path
        self.trajectory_file = self.output_dir / "complete_trajectory.json"
        
    def add_stage_record(self, stage_name: str, success: bool, details: Dict[str, Any], 
                        error_message: Optional[str] = None):
        """
        Add a stage record
        
        Args:
            stage_name: Stage name
            success: Whether it succeeded
            details: Details
            error_message: Error message (if failed)
        """
        stage_record = StageRecord(
            stage_name=stage_name,
            success=success,
            details=self._clean_details(details),
            error_message=error_message
        )
        
        self.trajectory.stages.append(stage_record)
        self._save_trajectory()

        # Print record information
        status_icon = "✅" if success else "❌"
        self.console.print(f"{status_icon} [{self.instance_id}] {stage_name}: {'Success' if success else 'Failed'}")

    def _add_stage_record_skipped(self, stage_name: str, details: Dict[str, Any]):
        """
        Add a skipped stage record (do not show as failure)

        Args:
            stage_name: Stage name
            details: Details
        """
        stage_record = StageRecord(
            stage_name=stage_name,
            success=False,  # Technically false, but skipped
            details=self._clean_details(details),
            error_message=None
        )

        self.trajectory.stages.append(stage_record)
        self._save_trajectory()

        # For skipped stages, show as skipped rather than failed
        self.console.print(f"⏭️ [{self.instance_id}] {stage_name}: Skipped")
        
    def set_final_result(self, result: Dict[str, Any]):
        """
        Set final result
        
        Args:
            result: Final result data
        """
        self.trajectory.final_result = self._clean_details(result)
        self._save_trajectory()
        
    def _clean_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean details and remove timestamps and other unnecessary fields
        
        Args:
            details: Original details
            
        Returns:
            Cleaned details
        """
        if not isinstance(details, dict):
            return details
            
        cleaned = {}
        
        # Keys to exclude (timestamps and similar)
        exclude_keys = {
            'timestamp', 'time', 'generated_at', 'created_at', 'updated_at',
            'duration', 'start_time', 'end_time', 'execution_time'
        }
        
        for key, value in details.items():
            # Skip time-related keys
            if any(time_key in key.lower() for time_key in exclude_keys):
                continue
                
            # Recursively clean nested dicts
            if isinstance(value, dict):
                cleaned[key] = self._clean_details(value)
            elif isinstance(value, list):
                cleaned[key] = [self._clean_details(item) if isinstance(item, dict) else item for item in value]
            else:
                cleaned[key] = value
                
        return cleaned
        
    def _save_trajectory(self):
        """Save trajectory record to file"""
        try:
            trajectory_data = asdict(self.trajectory)
            
            with open(self.trajectory_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save trajectory: {e}")
            
    def record_test_generation(self, test_generation_result: Dict[str, Any],
                             pre_validation_result: Dict[str, Any]):
        """
        Record the test case generation stage

        Args:
            test_generation_result: Test generation result
            pre_validation_result: Pre-validation result
        """
        success = test_generation_result.get("test_generation_success", False)
        skipped = test_generation_result.get("skipped", False)

        details = {
            "test_generation_result": test_generation_result,
            "pre_validation_result": pre_validation_result,
            "test_cases_generated": test_generation_result.get("test_cases_generated", 0),
            "validation_success": pre_validation_result.get("validation_success", False),
            "skipped": skipped
        }

        error_message = None
        if not success and not skipped:
            error_message = test_generation_result.get("error", "Unknown test generation error")

        # If test generation was skipped, use a special record to avoid showing failure
        if skipped:
            self._add_stage_record_skipped("test_generation", details)
        else:
            self.add_stage_record("test_generation", success, details, error_message)
        
    def record_diff_generation(self, diff: Optional[str], agent_duration: float,
                             agent_logs_path: Optional[Path] = None):
        """
        Record the diff generation stage

        Args:
            diff: Generated diff content
            agent_duration: Agent execution duration
            agent_logs_path: Agent log file path
        """
        # Fix: consider success when diff is not None; even an empty string is a valid diff (means no changes)
        success = diff is not None
        has_content = bool(diff and diff.strip())

        details = {
            "diff_generated": success,
            "diff_length": len(diff) if diff else 0,
            "has_diff_content": has_content,
            "agent_duration": agent_duration,
            "agent_logs_available": agent_logs_path is not None and agent_logs_path.exists() if agent_logs_path else False
        }

        error_message = None
        if not success:
            error_message = "Failed to generate diff - returned None"
        elif not has_content:
            # An empty diff is not an error; record as informational
            details["note"] = "Empty diff generated - no code changes detected"

        self.add_stage_record("diff_generation", success, details, error_message)
        
    def record_test_validation(self, execution_success: bool, test_results: Dict[str, Any], 
                             validation_result: Dict[str, Any]):
        """
        Record the test validation stage
        
        Args:
            execution_success: Whether test execution succeeded
            test_results: Test results
            validation_result: Validation result
        """
        details = {
            "execution_success": execution_success,
            "test_results": test_results,
            "validation_result": validation_result
        }
        
        error_message = None
        if not execution_success:
            error_message = "Test execution failed"
            
        self.add_stage_record("test_validation", execution_success, details, error_message)
        
    def record_trajectory_compression(self, compression_result: Optional[Dict[str, Any]]):
        """
        Record the trajectory compression stage
        
        Args:
            compression_result: Compression result
        """
        success = compression_result is not None
        
        details = {
            "compression_performed": success,
            "compression_result": compression_result if success else None
        }
        
        error_message = None
        if not success:
            error_message = "Trajectory compression failed"
            
        self.add_stage_record("trajectory_compression", success, details, error_message)
        
    def record_retry_attempt(self, retry_type: str, original_diff: Optional[str], 
                           retry_diff: Optional[str], similar_case: Optional[Dict[str, Any]] = None):
        """
        Record the retry stage
        
        Args:
            retry_type: Retry type ("experienced" or "no_experience")
            original_diff: Original diff
            retry_diff: Retry-generated diff
            similar_case: Similar case info (if any)
        """
        success = retry_diff is not None and len(retry_diff.strip()) > 0
        
        details = {
            "retry_type": retry_type,
            "retry_success": success,
            "original_diff_length": len(original_diff) if original_diff else 0,
            "retry_diff_length": len(retry_diff) if retry_diff else 0,
            "similar_case_found": similar_case is not None,
            "similar_case": similar_case if similar_case else None,
            "diff_changed": original_diff != retry_diff if original_diff and retry_diff else False
        }
        
        error_message = None
        if not success:
            error_message = f"Retry attempt ({retry_type}) failed"
            
        self.add_stage_record("retry_attempt", success, details, error_message)
        
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        Get trajectory summary
        
        Returns:
            Trajectory summary
        """
        total_stages = len(self.trajectory.stages)
        successful_stages = sum(1 for stage in self.trajectory.stages if stage.success)
        
        stage_summary = {}
        for stage in self.trajectory.stages:
            stage_summary[stage.stage_name] = {
                "success": stage.success,
                "error": stage.error_message
            }
            
        return {
            "instance_id": self.instance_id,
            "total_stages": total_stages,
            "successful_stages": successful_stages,
            "success_rate": successful_stages / total_stages if total_stages > 0 else 0,
            "stage_summary": stage_summary,
            "final_result_available": self.trajectory.final_result is not None
        }
        
    def export_compact_trajectory(self) -> Dict[str, Any]:
        """
        Export a compact trajectory record (for compression and analysis)
        
        Returns:
            Compact trajectory data
        """
        # Extract key information
        test_gen_success = False
        diff_gen_success = False
        retry_performed = False
        final_diff_available = False
        
        for stage in self.trajectory.stages:
            if stage.stage_name == "test_generation":
                test_gen_success = stage.success
            elif stage.stage_name == "diff_generation":
                diff_gen_success = stage.success
            elif stage.stage_name == "retry_attempt":
                retry_performed = True
                
        if self.trajectory.final_result:
            final_diff_available = bool(self.trajectory.final_result.get("model_patch"))
            
        return {
            "instance": self.instance_id,
            "problem_statement": self.problem_statement,
            "test_generation_success": test_gen_success,
            "diff_generation_success": diff_gen_success,
            "retry_performed": retry_performed,
            "final_diff_available": final_diff_available,
            "stage_count": len(self.trajectory.stages),
            "trajectory_file": str(self.trajectory_file)
        }


def create_execution_path_tracker(instance_id: str, problem_statement: str,
                             output_dir: Optional[Path] = None) -> TrajectoryRecorder:
    """
    Factory function to create a trajectory recorder
    
    Args:
        instance_id: Instance ID
        problem_statement: Problem statement
        output_dir: Output directory
        
    Returns:
        Trajectory recorder instance
    """
    return TrajectoryRecorder(instance_id, problem_statement, output_dir)


def load_trajectory_record(instance_id: str, output_dir: Optional[Path] = None) -> Optional[TrajectoryRecord]:
    """
    Load an existing trajectory record
    
    Args:
        instance_id: Instance ID
        output_dir: Output directory
        
    Returns:
        Trajectory record object, or None if it does not exist
    """
    output_dir = output_dir or Path(f"output_files/{instance_id}")
    trajectory_file = output_dir / "complete_trajectory.json"
    
    if not trajectory_file.exists():
        return None
        
    try:
        with open(trajectory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Reconstruct StageRecord objects
        stages = []
        for stage_data in data.get("stages", []):
            stage = StageRecord(
                stage_name=stage_data["stage_name"],
                success=stage_data["success"],
                details=stage_data["details"],
                error_message=stage_data.get("error_message")
            )
            stages.append(stage)
            
        return TrajectoryRecord(
            instance_id=data["instance_id"],
            problem_statement=data["problem_statement"],
            stages=stages,
            final_result=data.get("final_result")
        )
        
    except Exception as e:
        logging.error(f"Failed to load trajectory record for {instance_id}: {e}")
        return None



