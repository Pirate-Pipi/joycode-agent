from openai import OpenAI
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def open_ai_sdk(messages, model='gpt-4.1', max_retries=10):
    """
    Use the unified client manager to call the similarity search model
    """
    from get_client import call_similarity_search_model

    return call_similarity_search_model(
        messages=messages,
        max_retries=max_retries
    )


Similarity_Matching = '''
You are a professional issue similarity analyst. Your task is to select the most similar candidate problem from the provided list to the target problem.

Similarity Evaluation Criteria (in order of importance):
1. Core Issue Similarity (Weight: 40%): Whether the root cause and technical nature of the issue are the same.
2. Affected Modules/Components Overlap (Weight: 25%): Whether the same or related software modules are involved.
3. Technical Keyword Match (Weight: 20%): The extent of overlap in technical terms, API names, function names, class names, etc.
4. Issue Type Consistency (Weight: 10%): Whether the types (bug, feature request, etc.) match.
5. Error Symptom Similarity (Weight: 5%): Whether the specific manifestations and symptoms are similar.

Analysis Requirements:
1. Compare the structured information of the target problem with each candidate problem, one by one.
2. Focus on the matching degree of core_issue, affected_modules, and technical_keywords.
3. Consider whether the technical depth and complexity of the problems are comparable.
4. Give priority to problems with fundamentally similar nature rather than superficial description similarities.
5. IMPORTANT: Do not return the target problem itself as the most similar candidate - only select from the provided candidate problems.

Output Format:
Please output your analysis in the following JSON format:

<json>
{{
  "most_similar_instance_id": "instacne_id number of the most similar candidate problem",
  "similarity_score": "Similarity score (0-100)",
  "reasoning": "Detailed reasoning explaining why this candidate was chosen as the most similar"
}}
</json>


Target Problem Structured Information:
{target_problem_structure}

Candidate Problem List:
{candidate_problems_with_structures}

Please begin the analysis and provide the most similar candidate problem.

'''


# Get the path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
structured_problem_path = os.path.join(current_dir, 'structured_problem.json')

with open(structured_problem_path, 'r', encoding='utf-8') as f:
    structured_problem_lst = json.load(f)
    

def search(instance_fail_lst, instance_success_lst, n=10):
    candidate_pools = [item for item in structured_problem_lst if item['instance_id'] in instance_success_lst]
    fail = [item for item in structured_problem_lst if item['instance_id'] in instance_fail_lst]
    
    # Split the success list into n parts; sending too many at once can reduce result quality
    def split_list(lst, n_splits):
        k, m = divmod(len(lst), n_splits)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_splits)]

    success_splits = split_list(candidate_pools, n) # split into n parts
    inputs = [
        Similarity_Matching.format(
            target_problem_structure=fail,
            candidate_problems_with_structures=split
        )
        for split in success_splits
    ]


    def try_get_result(future, max_retry=3, delay=1):
        for attempt in range(max_retry):
            try:
                res = future.result()
                json_str = res.split('<json>', 1)[1].split('</json>', 1)[0]
                data = json.loads(json_str)
                # Force similarity_score to float
                try:
                    data['similarity_score'] = float(data.get('similarity_score', -1))
                except Exception:
                    data['similarity_score'] = -1
                return data
            except Exception as exc:
                print(f"Processing error on attempt {attempt+1}: {exc}")
                if attempt < max_retry - 1:
                    time.sleep(delay)  # retry
        return {'similarity_score': -1, 'most_similar_instance_id': None, 'reasoning': 'N/A'}  # final failure

    results = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(open_ai_sdk, inp): i for i, inp in enumerate(inputs)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Similarity matching progress"):
            data = try_get_result(future, max_retry=3, delay=1)
            results.append(data)

    def safe_score(x):
        try:
            return float(x.get('similarity_score', -1))
        except (ValueError, TypeError):
            return -1

    ## dict type
    best_result = max(results, key=safe_score)
    print(best_result)
    hit_id = best_result.get('most_similar_instance_id')
    similarity_score = best_result.get('similarity_score', -1)

    # Return a dict containing the similarity score
    return {
        'instance_id': hit_id,
        'similarity_score': similarity_score,
        'reasoning': best_result.get('reasoning', 'N/A')
    }
