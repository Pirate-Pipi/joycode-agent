import logging
from llm_server import call_llm_simple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pandas as pd
# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def open_ai_sdk(messages: str, max_retries: int = 10):
    # Use unified LLM interface from llm_server; model/purpose configured in llm_server/model_config.json
    for attempt in range(max_retries):
        try:
            response_text = call_llm_simple(
                purpose="patch_generation",
                prompt=messages,
                max_tokens=32000,
                temperature=1,
            )
            # Check whether the response is valid
            if response_text and isinstance(response_text, str) and response_text.strip():
                return response_text
            else:
                logging.warning(f"Attempt {attempt+1}: Empty response text.")
        except Exception as e:
            logging.warning(f"Attempt {attempt+1}: Exception occurred - {e}")
    return None


prompt_template = '''
You are a senior software engineer serving as a code reviewer. Your task is to evaluate two candidate code solutions (diffs) for a specific coding task. Please follow the steps below:

**Task description:**
<problem_statement>
{problem_statement}
</problem_statement>

Below, you will find 2 diffs and related information. Please follow these steps:

1. **Comprehension**: Carefully read and understand each candidate solution and its explanation. Make sure you fully grasp what each solution does and the reasoning behind it.
2. **Analysis**: Identify the strengths, weaknesses, and potential trade-offs of each solution. Consider aspects such as correctness, efficiency, readability, maintainability, edge case handling, and adherence to best practices.
3. **Comparison**: Directly compare the two solutions. Highlight any significant differences in their approaches or implementations, and discuss the impact of these differences.
4. **Optimal Solution Selection**: Choose the better solution (the one you consider most appropriate or commonly preferred based on your analysis).
5. **Justification**: Briefly explain the reason for your choice, describing why this solution is superior to the other. Your reasoning should reflect a rigorous and meticulous analysis process.

**IMPORTANT:**
Your final output must be a single JSON object in the following format:
```json
{{
  "solution_index": "1 or 2",
  "Basis of reasoning": ""
}}
```
- `solution_index`: The index (either "1" or "2") of the selected solution.
- `Basis of reasoning`: A concise summary of your reasoning and justification for choosing this solution, reflecting a meticulous analytical process.

Do not output anything except this JSON object as your final result.

Below are the two different candidate solutions. Each solution is enclosed within a `<candidate_solution>` tag, immediately followed by a `<candidate_explanation>` tag, which explains the reasoning and context for generating the diff.

Diff 1:
<candidate_solution index 1>
{diff_1}
</candidate_solution index 1>

Diff 2:
<candidate_solution index 2>
{diff_2}
</candidate_solution index 2>
'''

# 1. Load JSON and extract mapping: instance_id -> model_patch
def read_json_to_dict(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Map instance_id to model_patch
    return {v['instance_id']: v['model_patch'] for v in data.values()}

patch_dict_1 = read_json_to_dict('patch_1.json')
patch_dict_2   = read_json_to_dict('patch_2.json')

# 2. Read df_issue and build mapping: instance_id -> problem_statement
df_issue = pd.read_parquet('test-00000-of-00001.parquet')
instance2problem = dict(zip(df_issue['instance_id'], df_issue['problem_statement']))

# 3. Iterate over instance_ids from JSON and build prompts
query_lst = []
for idx, instance_id in enumerate(patch_dict_1.keys()):
    problem_statement = instance2problem.get(instance_id, "")
    diff_1 = patch_dict_1.get(instance_id, "")
    diff_2 = patch_dict_2.get(instance_id, "")

    prompt = prompt_template.format(
        problem_statement=problem_statement,
        diff_1=diff_1,
        diff_2=diff_2
    )
    query_lst.append((idx, prompt, instance_id))  # attach instance_id


# Multi-threaded processing
MAX_PARSE_RETRIES = 32
result_lst = [' '] * len(query_lst)

def call_and_parse(idx, prompt, instance_id):
    for parse_attempt in range(MAX_PARSE_RETRIES):
        try:
            result = open_ai_sdk(prompt)
            parsed = json.loads(result.split("```json")[1].split("```")[0].strip())
            if isinstance(parsed, dict):
                parsed['id'] = idx
                parsed['instance_id'] = instance_id  # attach instance_id
                return parsed
            else:
                logging.warning(f"Parsed result is not a dict, retry attempt {parse_attempt+1}: {result}")
        except Exception as exc:
            logging.warning(f"Parsing exception, retry attempt {parse_attempt+1}, error: {exc}")
    return ' '

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(call_and_parse, idx, prompt, instance_id): idx for idx, prompt, instance_id in query_lst}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Progress"):
        idx = futures[future]
        try:
            result = future.result()
            result_lst[idx] = result
        except Exception as exc:
            result_lst[idx] = ' '
            logging.error(f"Processing {idx} failed after multiple retries, error: {exc}")

# Optionally save results
with open('result.json', 'w', encoding='utf-8') as f:
    json.dump(result_lst, f, ensure_ascii=False, indent=2)