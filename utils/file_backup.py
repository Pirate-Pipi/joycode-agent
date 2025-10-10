"""
Unified file backup and replacement system.

This module provides consistent file backup and replacement operations
for different retry scenarios.
"""

import json
import time
import shutil
from pathlib import Path
from typing import Optional
from rich.console import Console


class FileBackupManager:
    """Manages file backup and replacement operations"""
    
    def __init__(self, instance_id: str, output_files_dir: str = "output_files"):
        self.instance_id = instance_id
        self.output_dir = Path(output_files_dir) / instance_id
        self.console = Console()
        
    def backup_and_replace_files(self, new_diff: str, new_logs_path: Optional[str], 
                                retry_type: str) -> bool:
        """
        Backup original files and replace with new ones.
        
        Args:
            new_diff: New generated diff
            new_logs_path: Path to new agent logs
            retry_type: Type of retry for record keeping
            
        Returns:
            True if successful, False otherwise
        """
        if not self.output_dir.exists():
            self.console.print(f"⚠️ Output directory not found: {self.output_dir}")
            return False
        
        try:
            # Backup and replace predictions.json
            success_predictions = self._backup_and_replace_predictions(new_diff, retry_type)
            
            # Backup and replace agent logs
            success_logs = self._backup_and_replace_logs(new_logs_path, retry_type)
            
            # Create retry record
            self._create_retry_record(new_diff, retry_type)
            
            if success_predictions:
                self.console.print(f"✅ Files updated successfully: {self.instance_id} ({retry_type})")
                return True
            else:
                self.console.print(f"❌ Failed to update files: {self.instance_id}")
                return False
                
        except Exception as e:
            self.console.print(f"❌ File backup/replace failed {self.instance_id}: {e}")
            return False
    
    def _backup_and_replace_predictions(self, new_diff: str, retry_type: str) -> bool:
        """Backup and replace predictions.json"""
        predictions_file = self.output_dir / "predictions.json"
        if not predictions_file.exists():
            self.console.print(f"⚠️ predictions.json not found: {self.instance_id}")
            return False
        
        # Backup original file
        original_backup = self.output_dir / "predictions_original.json"
        if not original_backup.exists():
            shutil.copy2(predictions_file, original_backup)
            self.console.print(f"📁 Backed up original predictions.json: {self.instance_id}")
        
        # Load and update predictions data
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        if isinstance(predictions_data, list) and len(predictions_data) > 0:
            # Update with new diff
            predictions_data[0]["model_patch"] = new_diff
            predictions_data[0]["retry_type"] = retry_type
            predictions_data[0]["retry_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Save updated file
            with open(predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            
            return True
        else:
            self.console.print(f"⚠️ Invalid predictions.json format: {self.instance_id}")
            return False
    
    def _backup_and_replace_logs(self, new_logs_path: Optional[str], retry_type: str) -> bool:
        """Backup and replace agent logs"""
        agent_logs_file = self.output_dir / "agent_logs.txt"
        
        # Backup original logs if exists
        if agent_logs_file.exists():
            agent_logs_original = self.output_dir / "agent_logs_original.txt"
            if not agent_logs_original.exists():
                shutil.copy2(agent_logs_file, agent_logs_original)
                self.console.print(f"📁 Backed up original agent_logs.txt: {self.instance_id}")
        
        # Copy new logs if available
        if new_logs_path and Path(new_logs_path).exists():
            # Copy as retry-specific logs
            retry_logs_dest = self.output_dir / f"agent_logs_{retry_type}.txt"
            shutil.copy2(new_logs_path, retry_logs_dest)
            
            # Replace current logs
            shutil.copy2(new_logs_path, agent_logs_file)
            self.console.print(f"📁 Updated logs: {self.instance_id} -> {retry_type}")
            return True
        else:
            self.console.print(f"⚠️ No new logs to copy: {self.instance_id}")
            return False
    
    def _create_retry_record(self, new_diff: str, retry_type: str) -> None:
        """Create retry record for tracking"""
        retry_record = {
            "instance_id": self.instance_id,
            "retry_type": retry_type,
            "retry_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "new_patch_length": len(new_diff) if new_diff else 0,
            "success": bool(new_diff and new_diff.strip())
        }
        
        retry_record_file = self.output_dir / f"retry_record_{retry_type}.json"
        with open(retry_record_file, 'w') as f:
            json.dump(retry_record, f, indent=2)


def backup_and_replace_instance_files(instance_id: str, new_diff: str, 
                                     new_logs_path: Optional[str], retry_type: str,
                                     output_files_dir: str = "output_files") -> bool:
    """
    Convenience function for backing up and replacing instance files.
    
    Args:
        instance_id: Instance ID
        new_diff: New generated diff
        new_logs_path: Path to new agent logs
        retry_type: Type of retry (for record keeping)
        output_files_dir: Base output directory
        
    Returns:
        True if successful, False otherwise
    """
    backup_manager = FileBackupManager(instance_id, output_files_dir)
    return backup_manager.backup_and_replace_files(new_diff, new_logs_path, retry_type)
