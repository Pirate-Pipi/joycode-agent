"""Test Case Generation Prompt

This prompt instructs the LLM to generate high-quality test cases that can validate diff quality.
"""

TEST_CASE_GENERATION_INSTRUCTIONS = f"""
### **Generate Test Cases for Diff Validation**

**WARNING: CRITICAL: NEVER use shebang lines (#!/usr/bin/env python3) in bash commands!**

<uploaded_files>
{{location}}
</uploaded_files>

I have uploaded a Python code repository to the directory `{{location}}`. Please consider the following problem statement and any associated hints:

<problem_statement>
{{problem_statement}}
</problem_statement>

<hints_text>
{{hints_text}}
</hints_text>

---

**TASK**: Generate three test cases to validate that a diff fixes the reported problem.

**IMPORTANT**: Before generating test cases, ensure all required dependencies are available. If any Python package is missing, install it using the JD mirror source.

**ENVIRONMENT SETUP** (Execute First):
```bash
# Check and install pytest if needed
python -c "import pytest" 2>/dev/null || pip install pytest

# Verify pytest installation
python -c "import pytest; print('pytest ready')"
```

**DEPENDENCY MANAGEMENT RULE**:
- When generating test case code, ALWAYS check if imported packages exist
- If a package is missing, install it BEFORE creating the test file

**EXECUTION STEPS**:

1. **Problem Analysis** (2-3 rounds):
   - Round 1: Understand the problem statement and identify key components
   - Round 2: Explore repository structure and find relevant code
   - Round 3: Locate existing tests and understand the bug context

2. **Test Case Generation** (2-3 rounds):
   - Round 4: Generate initial test cases based on analysis
   - Round 5: Test and validate each test case individually
   - Round 6: Refine any test cases that don't meet requirements

3. **Final Implementation** (1 round):
   - Round 7: Create final test case files using bash commands

**TEST CASE REQUIREMENTS**:

1. **test_failure_scenario.py**: MUST FAIL on original code, PASS on fixed code
   - Reproduce the exact bug/problem described
   - This validates the diff actually fixes the issue
   - **CRITICAL**: Use direct assertions, NO try-catch blocks that mask failures

2. **test_happy_path.py**: MUST PASS on both original and fixed code
   - Test normal functionality that should work
   - Ensures the fix doesn't break existing features

3. **test_edge_case.py**: MUST PASS on both original and fixed code
   - Test edge cases close to the problem but don't trigger the bug
   - Verifies fix maintains stability

**QUALITY STANDARDS**:
- Clear docstrings explaining test purpose
- Concise but comprehensive test logic
- Meaningful assertions with context
- Print statements for test progress and results

**MANDATORY**: You MUST use bash commands to create the actual files on disk.

**REQUIRED BASH COMMANDS TO EXECUTE:**
```bash
mkdir -p test_cases
```

```bash
cat > test_cases/test_failure_scenario.py << 'EOF'
import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix, is_separable

def test_reported_bug():
    '''Reproduce the reported bug: nested compound models separability issue'''
    # This should fail on original code due to the reported bug
    cm = m.Linear1D(1) & m.Linear1D(1)
    nested = m.Linear1D(2) & cm

    sep = separability_matrix(nested)
    expected = np.array([
        [True,  False, False],
        [False, True,  False],
        [False, False, True]
    ], dtype=bool)

    # This assertion should fail on original code
    assert (sep == expected).all(), "Bug reproduction: Expected expected_value, got sep_value"
    print("PASS: Bug reproduction test completed")
EOF
```

```bash
cat > test_cases/test_happy_path.py << 'EOF'
import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix, is_separable

def test_basic_functionality():
    '''Test basic separability functionality that should work'''
    # Simple parallel models should work correctly
    cm = m.Linear1D(1) & m.Linear1D(1)
    sep = separability_matrix(cm)

    # Basic validation
    assert sep.shape == (2, 2)
    assert sep.dtype == bool

    print("PASS: Basic functionality test PASSED")
EOF
```

```bash
cat > test_cases/test_edge_case.py << 'EOF'
import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix, is_separable

def test_edge_case():
    '''Test edge case that should work but is close to the problem'''
    # Test a scenario close to the bug but doesn't trigger it
    model = m.Linear1D(1) & m.Linear1D(1)
    sep = separability_matrix(model)

    # Verify basic structure
    assert sep.shape == (2, 2)
    assert sep.dtype == bool

    print("PASS: Edge case test PASSED")
EOF
```

```bash
ls test_cases/
```

**IMPORTANT**:
- Execute each step separately to avoid timeouts
- Focus on quality over quantity
- Ensure test_failure_scenario.py reproduces the exact reported problem
"""

# Global configuration for test case generation agent
TEST_CASE_GENERATION_AGENT_CONFIG = {
    "max_turns": 30,
    "max_output_tokens_per_turn": 8192,
    "ask_user_permission": False
}
