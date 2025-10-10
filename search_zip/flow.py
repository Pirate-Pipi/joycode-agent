# search_zip/flow.py
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rich.console import Console

# Import existing modules
from .zip_traj import zip_traj
from .search import search


class TrajectoryProcessor:
    """Trajectory processing module: handles file parsing, content extraction, and trajectory compression"""
    
    def __init__(self, output_files_dir: str = "output_files"):
        self.output_files_dir = Path(output_files_dir)
        self.console = Console()
        
    def extract_and_compress_trajectories(self, successful_ids: List[str] = None) -> List[Dict]:
        """
        Extract and compress trajectories (optionally only compress successful ones)

        Args:
            successful_ids: List of successful case IDs; if None, compress all trajectories

        Returns:
            List of compressed trajectories
        """
        try:
            self.console.print("🔍 Start extracting and compressing trajectories...")
            
            # 1. Build input data
            input_data = self._build_input_data()
            
            if not input_data:
                self.console.print("⚠️ No valid trajectory data found")
                return []
            
            self.console.print(f"📁 Found {len(input_data)} trajectory files")
            
            # 2. Call zip_traj to compress
            compressed_trajectories = zip_traj(input_data)
            
            if compressed_trajectories:
                self.console.print(f"✅ Successfully compressed {len(compressed_trajectories)} trajectories")
                
                # 3. Save compressed results
                self._save_compressed_trajectories(compressed_trajectories)
                
                return compressed_trajectories
            else:
                self.console.print("❌ Trajectory compression failed")
                return []
                
        except Exception as e:
            self.console.print(f"💥 Trajectory processing error: {e}")
            return []
    
    def _build_input_data(self) -> List[Dict]:
        """Build input data for zip_traj"""
        input_data = []
        
        try:
            # Traverse the output_files directory
            if not self.output_files_dir.exists():
                self.console.print(f"❌ Directory does not exist: {self.output_files_dir}")
                return []
            
            # Traverse each subdirectory (each corresponds to a problem_id)
            for problem_dir in self.output_files_dir.iterdir():
                if not problem_dir.is_dir():
                    continue
                
                problem_id = problem_dir.name
                
                # Check required files exist
                trajectory_file = problem_dir / "agent_logs.txt"
                predictions_file = problem_dir / "predictions.json"
                
                if not trajectory_file.exists():
                    self.console.print(f"⚠️ Trajectory file not found: {trajectory_file}")
                    continue
                    
                if not predictions_file.exists():
                    self.console.print(f"⚠️ Predictions file not found: {predictions_file}")
                    continue
                
                try:
                    # Read file contents
                    trajectory_content = trajectory_file.read_text(encoding='utf-8')
                    patch_content = predictions_file.read_text(encoding='utf-8')
                    
                    # Build input data
                    input_item = {
                        "trajectory": trajectory_content,
                        "patch": patch_content,
                        "instance": problem_id
                    }
                    
                    input_data.append(input_item)
                    self.console.print(f"✅ Loaded: {problem_id}")
                    
                except Exception as e:
                    self.console.print(f"⚠️ Failed to read files {problem_id}: {e}")
                    continue
            
            return input_data
            
        except Exception as e:
            self.console.print(f"💥 Error building input data: {e}")
            return []
    
    def _save_compressed_trajectories(self, compressed_trajectories: List[Dict]):
        """Save compressed trajectories (global and per-instance)"""
        try:
            # 1. Save global compressed trajectories (existing logic)
            global_output_file = Path("./compressed_trajectories.json")
            global_output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(global_output_file, 'w', encoding='utf-8') as f:
                json.dump(compressed_trajectories, f, indent=2, ensure_ascii=False)
            
            self.console.print(f"💾 Global compressed trajectories saved to: {global_output_file}")
            
            # 2. Save compressed trajectories for each instance
            saved_count = 0
            for trajectory in compressed_trajectories:
                try:
                    # Handle different types of trajectory data
                    if isinstance(trajectory, dict):
                        instance_id = trajectory.get("instance")
                    elif isinstance(trajectory, str):
                        # If it's a string, skip per-instance saving
                        continue
                    else:
                        continue
                        
                    if instance_id:
                        instance_dir = self.output_files_dir / instance_id
                        instance_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save compressed trajectory for a single instance
                        instance_compressed_file = instance_dir / "compressed_trajectory.json"
                        with open(instance_compressed_file, 'w', encoding='utf-8') as f:
                            json.dump(trajectory, f, indent=2, ensure_ascii=False)
                        
                        saved_count += 1
                        self.console.print(f"💾 {instance_id} compressed trajectory saved to: {instance_compressed_file}")
                except Exception as e:
                    self.console.print(f"⚠️ Skipping saving single instance trajectory: {e}")
                    continue
            
            self.console.print(f"✅ Saved compressed trajectories for {saved_count} instances in total")
            
        except Exception as e:
            self.console.print(f"❌ Failed to save compressed trajectories: {e}")


class SimilarCaseFinder:
    """Similar case finder module: searches similar cases and gets compressed trajectory info"""
    
    def __init__(self, compressed_trajectories_file: str = "./compressed_trajectories.json"):
        self.compressed_trajectories_file = Path(compressed_trajectories_file)
        self.console = Console()
        self.compressed_trajectories = self._load_compressed_trajectories()
        
    def _load_compressed_trajectories(self) -> List[Dict]:
        """Load compressed trajectory data"""
        try:
            if not self.compressed_trajectories_file.exists():
                self.console.print(f"⚠️ Compressed trajectories file does not exist: {self.compressed_trajectories_file}")
                return []
            
            with open(self.compressed_trajectories_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.console.print(f"📚 Loaded {len(data)} compressed trajectories")
            return data
            
        except Exception as e:
            self.console.print(f"💥 Failed to load compressed trajectories: {e}")
            return []
    
    def find_similar_case(self, failed_instance_id: str, 
                         available_success_instances: List[str], 
                         save_search_result: bool = True) -> Optional[Dict]:
        """
        Find the most similar case for a failed instance and save the search result

        Args:
            failed_instance_id: Failed instance ID
            available_success_instances: List of available successful instances
            save_search_result: Whether to save the search result

        Returns:
            Compressed trajectory info of the similar case, or None if not found
        """
        try:
            self.console.print(f"🔍 Searching similar cases for failed instance {failed_instance_id}...")
            
            if not available_success_instances:
                self.console.print("⚠️ No available successful instances")
                if save_search_result:
                    self._save_search_result(failed_instance_id, None)
                return None
            
            # 1. Call search() to find the most similar instance
            search_result = search(
                [failed_instance_id],  # failed instance
                available_success_instances,  # successful instances
                n=10 # number of splits
            )
            
            if not search_result or not search_result.get('instance_id'):
                self.console.print("❌ No similar case found")
                if save_search_result:
                    self._save_search_result(failed_instance_id, None)
                return None
            
            most_similar_id = search_result['instance_id']
            similarity_score = search_result.get('similarity_score', -1)
            reasoning = search_result.get('reasoning', 'N/A')
            
            self.console.print(f"✅ Most similar case found: {most_similar_id} (similarity: {similarity_score})")
            
            # 2. Get this case's info from compressed trajectories
            similar_case_info = self._get_case_trajectory_info(most_similar_id)
            
            if similar_case_info:
                self.console.print(f"📋 Retrieved trajectory info for case {most_similar_id}")
                # Attach similarity score and reasoning to the returned dict
                similar_case_info['similarity_score'] = similarity_score
                similar_case_info['similarity_reasoning'] = reasoning
                
                # 3. Save search result
                if save_search_result:
                    self._save_search_result(failed_instance_id, {
                        'search_result': search_result,
                        'similar_case_info': similar_case_info,
                        'available_candidates': available_success_instances
                    })
                
                return similar_case_info
            else:
                self.console.print(f"⚠️ No trajectory info found for case {most_similar_id}")
                if save_search_result:
                    self._save_search_result(failed_instance_id, {
                        'search_result': search_result,
                        'similar_case_info': None,
                        'available_candidates': available_success_instances
                    })
                return None
                
        except Exception as e:
            self.console.print(f"💥 Error during similar case search: {e}")
            if save_search_result:
                self._save_search_result(failed_instance_id, None)
            return None
    
    def _get_case_trajectory_info(self, instance_id: str) -> Optional[Dict]:
        """Get trajectory info for a specific instance"""
        try:
            # Search in the compressed trajectories list
            for case_info in self.compressed_trajectories:
                # Type check to ensure case_info is a dict
                if not isinstance(case_info, dict):
                    continue
                    
                if case_info.get("instance") == instance_id:
                    return case_info
            
            self.console.print(f"⚠️ Instance not found in compressed trajectories: {instance_id}")
            return None
            
        except Exception as e:
            self.console.print(f"💥 Error getting trajectory info: {e}")
            return None
    
    def _save_search_result(self, failed_instance_id: str, search_data: Optional[Dict]):
        """Save similarity search results to the instance directory"""
        try:
            import time
            from pathlib import Path
            
            # Determine the save directory
            instance_dir = Path("output_files") / failed_instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            
            # Build search result data
            if search_data is None:
                # Case: no similar case found
                result_data = {
                    "failed_instance_id": failed_instance_id,
                    "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "search_success": False,
                    "similar_case_found": False,
                    "reason": "No similar case found or search failed"
                }
            else:
                # Case: similar case found
                search_result = search_data.get('search_result', {})
                similar_case_info = search_data.get('similar_case_info')
                
                result_data = {
                    "failed_instance_id": failed_instance_id,
                    "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "search_success": True,
                    "similar_case_found": similar_case_info is not None,
                    "most_similar_instance_id": search_result.get('instance_id'),
                    "similarity_score": search_result.get('similarity_score', -1),
                    "similarity_reasoning": search_result.get('reasoning', 'N/A'),
                    "available_candidates_count": len(search_data.get('available_candidates', [])),
                    "similar_case_strategy": similar_case_info.get('strategy') if similar_case_info else None,
                    "similar_case_key_changes": similar_case_info.get('key_changes') if similar_case_info else None
                }
            
            # Save to file
            search_result_file = instance_dir / "similarity_search_result.json"
            with open(search_result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            self.console.print(f"💾 Search results saved to: {search_result_file}")
            
        except Exception as e:
            self.console.print(f"❌ Failed to save search results: {e}")
    
    def get_available_success_instances(self, successful_ids: List[str] = None) -> List[str]:
        """Get all available successful instance IDs"""
        try:
            # Extract successful instance IDs from compressed trajectories
            available_instances = []
            
            for case_info in self.compressed_trajectories:
                # Type check to ensure case_info is a dict
                if not isinstance(case_info, dict):
                    continue
                    
                instance_id = case_info.get("instance")
                # Only include instances that are in the successful_ids list
                if successful_ids is not None:
                    if instance_id and instance_id in successful_ids:
                        available_instances.append(instance_id)
                else:
                    # If successful_ids is not provided, return an empty list
                    self.console.print("⚠️ No successful case list provided, returning empty list")
                    return []
            
            self.console.print(f"✅ Found {len(available_instances)} successful instances")
            return available_instances
            
        except Exception as e:
            self.console.print(f"💥 Error getting available instance list: {e}")
            return []


class FlowManager:
    """Flow manager: coordinates the two modules"""
    
    def __init__(self, output_files_dir: str = "output_files"):
        self.trajectory_processor = TrajectoryProcessor(output_files_dir)
        self.similar_case_finder = SimilarCaseFinder()
        self.console = Console()
    
    def process_all_trajectories(self) -> bool:
        """Process all trajectories: extract, compress, and save"""
        try:
            self.console.print("🚀 Start processing all trajectories...")
            
            # Call trajectory processing module
            compressed_trajectories = self.trajectory_processor.extract_and_compress_trajectories()
            
            if compressed_trajectories:
                self.console.print("✅ All trajectories processed")
                return True
            else:
                self.console.print("❌ Trajectory processing failed")
                return False
                
        except Exception as e:
            self.console.print(f"💥 Error in trajectory processing flow: {e}")
            return False
    
    def find_similar_case_for_failure(self, failed_instance_id: str, successful_ids: List[str]) -> Optional[Dict]:
        """Find a similar case for a failed instance"""
        try:
            self.console.print(f"🔍 Finding similar case for failed instance {failed_instance_id}...")
            
            # Get available successful instances
            available_success_instances = self.similar_case_finder.get_available_success_instances(successful_ids)
            
            if not available_success_instances:
                self.console.print("⚠️ No available successful instances")
                return None
            
            # Find similar case
            similar_case = self.similar_case_finder.find_similar_case(
                failed_instance_id, 
                available_success_instances
            )
            
            return similar_case
            
        except Exception as e:
            self.console.print(f"💥 Error in similar case search flow: {e}")
            return None


# Main functions
def process_trajectories_and_compress():
    """Process all trajectories and compress - Part 1"""
    flow_manager = FlowManager()
    return flow_manager.process_all_trajectories()


# Global FlowManager instance to avoid recreation
_global_flow_manager = None

def find_similar_case(failed_instance_id: str, successful_ids: List[str]) -> Optional[Dict]:
    """Find similar case - Part 2 functionality"""
    global _global_flow_manager
    if _global_flow_manager is None:
        _global_flow_manager = FlowManager()
    return _global_flow_manager.find_similar_case_for_failure(failed_instance_id, successful_ids)

