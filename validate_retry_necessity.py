"""
Minimal LLM-based validation for retry necessity.
"""

import json
import time
import re
from pathlib import Path
from llm_server import call_llm_simple

# In-memory cache for dataset lookups
_PROBLEM_STATEMENT_CACHE = {}


def _get_problem_statement_from_dataset(instance_id: str) -> str:
    """Try to load problem_statement from the HF dataset; fallback to empty string."""
    if instance_id in _PROBLEM_STATEMENT_CACHE:
        return _PROBLEM_STATEMENT_CACHE[instance_id]
    try:
        # Lazy import with ignore to avoid static analysis errors if not installed
        from datasets import load_dataset  # type: ignore[import-not-found]
        ds = load_dataset("princeton-nlp___swe-bench_verified", split="test")
        # Iterate to avoid pandas dependency
        for row in ds:
            if row.get("instance_id") == instance_id:
                ps = row.get("problem_statement") or ""
                _PROBLEM_STATEMENT_CACHE[instance_id] = ps
                return ps
    except Exception:
        pass
    _PROBLEM_STATEMENT_CACHE[instance_id] = ""
    return ""


def _load_inputs(instance_id: str, output_files_dir: str) -> tuple:
    """Load issue, patch, and test_case with minimal assumptions."""
    instance_dir = Path(output_files_dir) / instance_id
    if not instance_dir.exists():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

    # Prefer dataset problem statement
    issue = _get_problem_statement_from_dataset(instance_id) or f"Problem statement for {instance_id}"
    patch = ""
    test_case = ""

    # predictions.json -> model_patch (+ optional problem_statement)
    predictions_file = instance_dir / "predictions.json"
    if predictions_file.exists():
        try:
            with open(predictions_file, 'r') as f:
                predictions_data = json.load(f)
            if isinstance(predictions_data, list) and predictions_data:
                patch = predictions_data[0].get("model_patch", "")
                # Only use predictions problem_statement if dataset is empty
                if not issue:
                    issue = predictions_data[0].get("problem_statement", issue)
            elif isinstance(predictions_data, dict):
                patch = predictions_data.get("model_patch", "")
                if not issue:
                    issue = predictions_data.get("problem_statement", issue)
        except Exception:
            pass

    # test_cases/*.py -> first file content
    test_cases_dir = instance_dir / "test_cases"
    if test_cases_dir.exists():
        for test_file in sorted(test_cases_dir.glob("*.py")):
            try:
                with open(test_file, 'r') as f:
                    test_case = f.read()
                break
            except Exception:
                continue

    return issue, patch, test_case


def _call_llm_for_judgment(issue: str, patch: str, test_case: str) -> str:
    """Call LLM with judgment prompt and return raw text (expected JSON)."""
    from prompts.notpass_judgement import NOTPASS_JUDGEMENT_PROMPT
    prompt = NOTPASS_JUDGEMENT_PROMPT.format(
        pr_description=issue,
        patch=patch,
        test_case=test_case,
    )
    response_text = call_llm_simple(
        purpose="patch_generation",
        prompt=prompt,
        max_tokens=400,
        temperature=0.2,
    ) or ""
    return response_text.strip()


def _parse_response(response: str) -> dict:
    """Parse JSON from LLM response simply."""
    if not response:
        return {"root_cause": "PATCH", "confidence": 0.5, "one_sentence": "Empty response"}
    try:
        return json.loads(response)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", response)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"root_cause": "PATCH", "confidence": 0.5, "one_sentence": "Unparseable response"}


def judge_failure_root_cause(instance_id: str, output_files_dir: str = "output_files") -> dict:
    """Return LLM-based root cause judgment for a given instance and save it."""
    instance_dir = Path(output_files_dir) / instance_id
    judgment_file = instance_dir / "judgment_result.json"

    try:
        issue, patch, test_case = _load_inputs(instance_id, output_files_dir)
        response = _call_llm_for_judgment(issue, patch, test_case)
        data = _parse_response(response)
        result = {
            "instance_id": instance_id,
            "root_cause": data.get("root_cause", "PATCH"),
            "confidence": data.get("confidence", 0.5),
            "reasoning": data.get("one_sentence", ""),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_response": response,
        }
        instance_dir.mkdir(parents=True, exist_ok=True)
        with open(judgment_file, 'w') as f:
            json.dump(result, f, indent=2)
        return result
    except Exception as e:
        result = {
            "instance_id": instance_id,
            "root_cause": "PATCH",
            "confidence": 0.5,
            "reasoning": f"Failed to analyze: {e}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_response": None,
        }
        try:
            instance_dir.mkdir(parents=True, exist_ok=True)
            with open(judgment_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        return result
