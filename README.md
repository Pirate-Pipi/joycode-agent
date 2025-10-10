# JoyCode SWE-bench Agent Pipeline

An end-to-end pipeline that lets LLMs fix real-world OSS: generate patch → generate/verify tests → intelligent retries.

This repository provides a ready-to-run SWE-bench pipeline. The default entry point is `run_patch_pipeline.py`.

---

## Features

- Patch generation: run agents in a container to produce code diffs
- Test generation (optional) and pre-validation on the original codebase
- Validation and root-cause analysis for failures (test issue vs. patch issue)
- Intelligent retries powered by failure attribution and similar-case retrieval
- Full execution trajectory with optional compression for easy review

## Repository Layout (Key Files)

- `run_patch_pipeline.py`: main entry-point script
- `cli.py`: core agent CLI invoked by the entry point
- `test_case_generator/`: logic for test generation and pre-validation
- `test/`: execute tests inside the container and judge results
- `utils/docker_utils.py`: container management, image pulling, workspace setup
- `llm_server/`: unified LLM invocation and configuration (purpose-driven)
- `princeton-nlp___swe-bench_verified/`: local copy of SWE-bench Verified dataset

---

## Prerequisites

### 1) Python Environment

```bash
conda create -n joycode python=3.11
conda activate joycode
pip install -r requirements.txt
```

### 2) Docker Environment

- Install and start Docker (Linux/Mac/Windows)
- Ensure you can pull from `docker.1ms.run`

The pipeline automatically pulls per-instance images, e.g.:
- `swebench/sweb.eval.x86_64.<issue_key>:latest` (fetched from `docker.1ms.run/` and re-tagged)

If your network is restricted, you can pre-pull:

```bash
docker pull docker.1ms.run/swebench/sweb.eval.x86_64.<issue_key>:latest
```

### 3) LLM Configuration (Important)

Centralized model config lives at `llm_server/model_config.json`.
Replace `api_key` / `base_url` / `model_name` with your own. Different purposes can use different models.

### 4) Dataset & Instance List

- The repo includes a local SWE-bench Verified structure under `princeton-nlp___swe-bench_verified` (no extra download needed)
- Specify instances to run in `instance_id.txt` (one per line)
- You can also run a single instance from CLI flags (see below)

---

## Quickstart (Recommended)

```bash
python run_patch_pipeline.py --num-processes 1 --enable-post-processing
```

Meaning:
- `--num-processes 1`: start with 1 for easier log inspection
- `--enable-post-processing`: enable trajectory compression, similar-case matching, and intelligent retries

> Minimal reproducible run: ensure `instance_id.txt` contains at least one `instance_id`.

### Common Run Modes & Flags

- Simple mode (only generate patch; no test gen; no post-processing)

```bash
python run_patch_pipeline.py --simple-mode
```

- Single instance

```bash
python run_patch_pipeline.py --problem-id <INSTANCE_ID> --num-processes 1 --enable-post-processing
```

- Limit number of cases (from `instance_id.txt` or dataset)

```bash
python run_patch_pipeline.py --num-examples 10 --num-processes 4
```

- Toggle test generation and validation (both enabled by default)

```bash
python run_patch_pipeline.py --no-generate-tests --no-validate-with-tests
```

> Note: `--simple-mode` and `--enable-post-processing` are mutually exclusive.

---

## Outputs

Main output directory: `output_files/<instance_id>/`

You will find:
- `predictions.json`: model-generated patch (diff) and metadata
- `agent_logs.txt`: main agent run logs
- `test_generation_result.json`: test generation + pre-validation results (if enabled)
- `test_generation_logs.txt`: test generation logs (if enabled)
- `agent_logs_retry.txt`: retry agent logs (if retries occurred)

Summary files:
- `output_files/successful_cases.txt`, `failed_cases.txt`, `empty_diff_cases.txt`
- `output_files/similar_case_matches_summary.json`: similar case matching summary

Trajectory (if post-processing enabled):
- `output_files/<instance_id>/compressed_trajectory.txt`

---

## Voting Tool (vote.py)

We provide a simple voting script `vote.py` that lets the LLM choose between two candidate patches.

Important notes:
- Uses the unified `llm_server` interface (no direct OpenAI SDK usage)
- Logging via Python `logging` (INFO/WARNING/ERROR). Make sure `llm_server/model_config.json` is configured.

### 1) Prepare Inputs for vote.py

Three input files (place them at the repo root):
- `correct_patch.json`: candidate A (your preferred/current version)
- `error_patch.json`: candidate B (alternative for comparison)
- `test-00000-of-00001.parquet`: must contain at least `instance_id` and `problem_statement`

Recommended way to produce the two JSONs from `output_files/<instance_id>/`:
- If both `predictions_original.json` and `predictions.json` exist, use the latter as correct and the former as error
- If only `predictions.json` exists, obtain error from another run or skip cases without two candidates

Example script (builds dict of arbitrary keys -> {instance_id, model_patch}):

```python
import json, os
base = "output_files"
correct, error = {}, {}
for i, d in enumerate(os.listdir(base)):
    p = os.path.join(base, d)
    cur = os.path.join(p, "predictions.json")
    orig = os.path.join(p, "predictions_original.json")
    def read_patch(f):
        return json.load(open(f))[0]["model_patch"] if os.path.exists(f) else None
    c, e = read_patch(cur), read_patch(orig)
    if c and e:
        correct[str(i)] = {"instance_id": d, "model_patch": c}
        error[str(i)]   = {"instance_id": d, "model_patch": e}
json.dump(correct, open("correct_patch.json", "w"), indent=2)
json.dump(error,   open("error_patch.json",   "w"), indent=2)
```

Generate the Parquet with problem statements (if missing):

```python
from datasets import load_dataset
import pandas as pd
ds = load_dataset("princeton-nlp___swe-bench_verified")["test"].to_pandas()
ds[["instance_id","problem_statement"]].to_parquet("test-00000-of-00001.parquet")
```

### 2) Run the Voting Tool

From the repo root:

```bash
python vote.py
```

This will produce `result.json` with the model’s JSON output, including `solution_index` and a brief rationale.

Tips:
- vote.py uses `llm_server.call_llm_simple(purpose="patch_generation")`; models and hyperparams come from `llm_server/model_config.json`
- To configure a dedicated model for voting, define a separate purpose (e.g., `retry_agent`) in the config and switch the purpose in the script

---

## Pipeline Stages (Overview)

1) Start container: pull and start the SWE-bench image for the instance; run the repair agent inside
2) Generate tests (optional): validate that tests pass on the original code; copy to outputs
3) Run agent: use `cli.py` to produce the diff
4) Pre/Post validation: run tests inside the container and judge patch quality
5) Post-processing (optional): trajectory compression, similar-case retrieval, intelligent retries, and write-back

---

## Results

```
Submission summary for 20250909_JoyCode on SWE-bench verified split
==================================================
Resolved 373 instances (74.6%)
==================================================
Resolved by Repository
- astropy/astropy: 13/22 (59.09%)
- django/django: 178/231 (77.06%)
- matplotlib/matplotlib: 25/34 (73.53%)
- mwaskom/seaborn: 1/2 (50.0%)
- pallets/flask: 1/1 (100.0%)
- psf/requests: 3/8 (37.5%)
- pydata/xarray: 19/22 (86.36%)
- pylint-dev/pylint: 2/10 (20.0%)
- pytest-dev/pytest: 17/19 (89.47%)
- scikit-learn/scikit-learn: 28/32 (87.5%)
- sphinx-doc/sphinx: 29/44 (65.91%)
- sympy/sympy: 57/75 (76.0%)
==================================================
Resolved by Time
- 2013: 1/3 (33.33%)
- 2014: 0/2 (0.0%)
- 2015: 0/1 (0.0%)
- 2016: 2/2 (100.0%)
- 2017: 12/16 (75.0%)
- 2018: 18/24 (75.0%)
- 2019: 74/98 (75.51%)
- 2020: 88/108 (81.48%)
- 2021: 61/86 (70.93%)
- 2022: 72/102 (70.59%)
- 2023: 45/58 (77.59%)
```

---

## References & Acknowledgements

- SWE-bench: https://www.swe-bench.com/
