# JoyCode SWE-bench Agent Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

**JoyCode** is an end-to-end LLM-powered pipeline for fixing real-world open-source software issues. It generates patches, creates and verifies tests, and employs intelligent retry mechanisms to achieve high success rates on the SWE-bench dataset.

**Project Status:** JoyCode has achieved **74.6% resolution rate** on SWE-bench Verified split, demonstrating state-of-the-art performance in automated software engineering.

**Key Innovation:** Our pipeline combines patch generation with intelligent test creation and failure attribution, enabling robust automated code repair with comprehensive validation and smart retry mechanisms.

<image src="sources/Code_Agent.png" alt="Code Agent" width="800"/>

## ✨ Features

### 🏆 **High Performance & Cost Efficiency**
- **74.6% Success Rate** on SWE-bench Verified, ranking 2nd globally
- **30-50% Lower Resource Consumption** than top competitors
- Exceptional cost-performance ratio with near state-of-the-art results

### 🔄 **Patch-Test Co-generation**
- **Smart Test Generation**: Automatic Fail2Pass and Pass2Pass test creation with pre-validation
- **Collaborative Verification**: Patches and tests generated together for comprehensive validation
- **Closed-loop Iteration**: "Generate → Validate → Refine" cycle replacing one-shot approaches

### 🧠 **Intelligent Failure Attribution**
- **Root Cause Analysis**: Precise failure attribution to patch vs. test issues
- **Targeted Retry Strategy**: Experience-driven retries based on failure analysis
- **CSR-Powered Learning**: Historical success pattern retrieval for optimization

### 🏗️ **Multi-Agent Architecture**
- **Specialized Agents**: Testing, Patch, CSR, and Decision agents with distinct roles
- **React-based Workflow**: "Observe-Think-Act" loop mimicking human developers
- **Smart Decision Making**: LLM-powered voting for optimal patch selection

### 💡 **Smart Resource Management**
- **Token-Efficient Design**: Targeted LLM calls avoiding wasteful parallel sampling
- **Early Failure Detection**: Pre-validation to filter invalid paths
- **Quality-First Generation**: Fewer, higher-quality patches over massive sampling

### 🐳 **Production-Ready Engineering**
- **Containerized Execution**: Isolated Docker environments with SWE-bench images
- **Repository-Level Understanding**: Multi-file coordination and cross-module reasoning
- **Comprehensive Logging**: Full trajectory recording with optional compression
- **Multi-LLM Support**: Flexible model configuration for different pipeline stages

## 🚀 Installation

### Requirements
- Python 3.11+
- Docker with access to `docker.1ms.run`
- LLM API keys (OpenAI, Anthropic, etc.)

### Setup

```bash
# Clone repository
git clone https://github.com/jd-opensource/joycode-agent.git
cd joycode

# Create conda environment
conda create -n joycode python=3.11
conda activate joycode

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

### LLM Configuration

Configure your models in `llm_server/model_config.json`:

```json
{
  "patch_generation": {
    "api_key": "your_api_key_here",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-5",
    "max_tokens": 4000,
    "temperature": 1
  }
}
```

### Docker Setup

Ensure Docker is running and you can access the registry:

```bash
# Test Docker connectivity
docker pull docker.1ms.run/swebench/sweb.eval.x86_64.django__django-11099:latest
```

### Instance Configuration

Specify instances to process in `instance_id.txt`:
```
django__django-11099
matplotlib__matplotlib-23562
sympy__sympy-18189
...
```

## 📖 Usage

### Quick Start

```bash
# Run with default settings
python run_patch_pipeline.py --num-processes 1 --enable-post-processing
```

### Common Usage Patterns

```bash
# Simple mode (patch only, no tests)
python run_patch_pipeline.py --simple-mode

# Single instance processing
python run_patch_pipeline.py --problem-id django__django-11099 --num-processes 1

# Batch processing with limits
python run_patch_pipeline.py --num-examples 10 --num-processes 4

# Disable test generation
python run_patch_pipeline.py --no-generate-tests --no-validate-with-tests

# Custom configuration
python run_patch_pipeline.py --enable-post-processing --num-processes 2
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num-processes` | Number of parallel processes | 1 |
| `--enable-post-processing` | Enable trajectory compression and retries | False |
| `--simple-mode` | Patch generation only | False |
| `--problem-id` | Process single instance | None |
| `--num-examples` | Limit number of instances | All |
| `--no-generate-tests` | Skip test generation | False |
| `--no-validate-with-tests` | Skip test validation | False |

## 🛠️ Advanced Features

### Patch Voting System

Compare and select between multiple patch candidates:

```bash
# Prepare voting inputs
python scripts/prepare_voting_data.py

# Run voting
python vote.py
```

**Input Requirements:**
- `correct_patch.json`: Primary patch candidates
- `error_patch.json`: Alternative patch candidates  
- `test-00000-of-00001.parquet`: Instance metadata with problem statements

### Pipeline Stages

1. **Container Setup**: Pull and start SWE-bench Docker images
2. **Test Generation** (optional): Create and validate tests on original code
3. **Agent Execution**: Generate patches using LLM agents via `cli.py`
4. **Validation**: Run tests and evaluate patch quality
5. **Post-processing** (optional): Trajectory compression, similarity matching, intelligent retries

### Output Structure

```
output_files/
├── <instance_id>/
│   ├── predictions.json              # Generated patch and metadata
│   ├── agent_logs.txt               # Main agent execution logs
│   ├── test_generation_result.json  # Test generation results
│   ├── test_generation_logs.txt     # Test generation logs
│   ├── agent_logs_retry.txt         # Retry attempt logs
│   └── compressed_trajectory.txt    # Compressed execution trajectory
├── successful_cases.txt             # Summary of successful instances
├── failed_cases.txt                 # Summary of failed instances
├── empty_diff_cases.txt            # Cases with no generated patches
└── similar_case_matches_summary.json # Similar case analysis
```

## 📊 Performance Results

```
Submission summary for 20250909_JoyCode on SWE-bench verified split
==================================================
Resolved 373 instances (74.6%)
==================================================
Resolved by Repository:
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
```

## 🔧 Development

### Repository Structure

```
joycode/
├── run_patch_pipeline.py           # Main entry point
├── cli.py                         # Core agent CLI
├── test_case_generator/           # Test generation logic
├── test/                         # Test execution and validation
├── utils/docker_utils.py         # Container management
├── llm_server/                   # LLM integration layer
├── princeton-nlp___swe-bench_verified/ # Local SWE-bench dataset
└── vote.py                       # Patch voting system
```

### Troubleshooting

**Docker Issues:**
```bash
# Check Docker connectivity
docker info
docker pull docker.1ms.run/hello-world
```

**LLM Configuration:**
```bash
# Verify model config
python -c "import json; print(json.load(open('llm_server/model_config.json')))"
```

**Memory Issues:**
```bash
# Reduce parallel processes
python run_patch_pipeline.py --num-processes 1
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SWE-bench](https://www.swe-bench.com/) for providing the benchmark dataset
