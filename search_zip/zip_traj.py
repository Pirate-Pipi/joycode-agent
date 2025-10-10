from openai import OpenAI
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


system_prompt = '''
You are an AI assistant specialized in analyzing software engineering trajectories. Your task is to analyze execution trajectories from SWE-agent runs and provide structured insights about the solution approach.

You will be provided with:
1. A trajectory file (.txt) in JSON format containing the agent's step-by-step execution
2. A prediction file (.pred) containing the final result

Your goal is to extract and summarize the core solution strategy, techniques, and approaches used in this trajectory.

Return your analysis in JSON format with the following fields:
- approach_summary: A concise summary of the main approach used in this solution
- modified_files: List of files that were modified during execution  
- key_changes: Description of the most important code changes made
- strategy: The core solution strategy at an abstract level
- specific_techniques: Specific techniques or methods used in this solution
- tools_used: Tools and commands heavily utilized during execution
- reasoning_pattern: The problem-solving pattern observed in the trajectory
- assumptions_made: Key assumptions made during the solution process
- components_touched: Main components, functions, or modules that were modified

Focus on extracting actionable insights about the solution methodology rather than implementation details.
'''

user_prompt = '''
Please analyze the following SWE-agent trajectory and provide insights about the solution approach.

Trajectory Data (.txt file):
{trajectory_content}

Prediction Result (.json file):
{patch_content}

Please provide your analysis in the JSON format specified in the system prompt.
'''

def open_ai_sdk(trajectory_content, patch_content, model='gpt-4.1', max_retries=10):
    """
    Use the unified client manager to call the trajectory compression model
    """
    from get_client import compress_trajectory

    return compress_trajectory(
        trajectory_content=trajectory_content,
        patch_content=patch_content,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_retries=max_retries
    )


def zip_traj(data): 
    trajectory_content_lst = [item["trajectory"] for item in data]
    patch_content_lst = [item["patch"] for item in data]
    instance_lst = [item["instance"] for item in data]
    query_lst = []
    task_args_lst = [
        (instance_id, trajectory_content, patch_content)
        for instance_id, trajectory_content, patch_content in zip(instance_lst, trajectory_content_lst, patch_content_lst)
    ]

    MAX_PARSE_RETRIES = 3

    # Use instance_id as the key to build the result dict
    result_dict = {instance_id: ' ' for instance_id in instance_lst}

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(open_ai_sdk, trajectory_content, patch_content): instance_id
            for instance_id, trajectory_content, patch_content in task_args_lst
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Trajectory compression progress"):
            instance_id = futures[future]
            parse_success = False
            for parse_attempt in range(MAX_PARSE_RETRIES):
                try:
                    result = future.result()
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        parsed['instance'] = instance_id   # Use instance_id as identifier
                        result_dict[instance_id] = parsed
                        parse_success = True
                        break
                    else:
                        print(f"Parsed result is not a dict, retry attempt {parse_attempt+1}: {result}")
                except Exception as exc:
                    print(f"Parse exception, retry attempt {parse_attempt+1}, error: {exc}")

            if not parse_success:
                result_dict[instance_id] = ' '
                idx = instance_lst.index(instance_id)
                print(f"Processing {trajectory_content_lst[idx]}, {patch_content_lst[idx]} failed after multiple retries")
    # Return result list (preserving input order)
    result_lst = [result_dict[instance_id] for instance_id in instance_lst]
    return result_lst
