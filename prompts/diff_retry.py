"""Retry Instruction Prompt
This prompt primarily guides large language models in performing critical analysis of prior errors, leveraging historical experience, and generating improved code patches
"""

INSTRUCTION_PROMPT = """
<uploaded_files>
{location}
</uploaded_files>

A Python codebase has been provided in the directory {location} (separate from /tmp/inputs).
Please examine the following pull request specification:

<pr_description>
{pr_description}
</pr_description>

The following represents the condensed execution path of the prior attempt for this instance (essential thoughts, decisions, modifications):

<current_attempt_compressed_trajectory>
{current_compressed_trajectory}
</current_attempt_compressed_trajectory>

Additionally, we have identified a comparable successful case for guidance:

- similar_instance_id: {similar_instance_id}
- similarity_score: {similarity_score}
- similarity_reasoning: {similarity_reasoning}
- similar_case_strategy: {similar_case_strategy}
- similar_case_key_changes: {similar_case_key_changes}

The condensed execution path of that successful case follows:

<similar_success_compressed_trajectory>
{similar_compressed_trajectory}
</similar_success_compressed_trajectory>

Assignment:
- Perform rigorous comparison between your prior attempt and the analogous successful case; retain valid elements, eliminate misleading aspects, and incorporate only genuinely transferable components.
- Execute the most minimal accurate modification to production files in {location} that fulfills the PR requirements.
- Refrain from test modifications. Prevent unnecessary alterations and additional dependencies.

Execute the following systematic approach to address the issue:
1. **Problem Analysis**: Initiate with comprehensive examination of the PR specification to achieve complete understanding. Determine the fundamental components affected and anticipated functionality. Ensure thorough comprehension before proceeding.
2. **Repository Survey**: Initially investigate the repository architecture to comprehend essential components and primary files. Emphasize obtaining a broad perspective rather than detailed exploration.
3. **Error Replication**: Construct a script to replicate the error and execute it via `python <filename.py>` using the BashTool to validate the error
4. **Solution Planning (retry-conscious)**: Examine the prior attempt's condensed trajectory to identify erroneous assumptions and deficiencies; reference the analogous successful case's trajectory to extract genuinely applicable techniques; analyze and determine the most probable root cause; develop a concrete, verifiable hypothesis and design the most minimal viable modification; validate the hypothesis (e.g., focused logging or minimal reproduction verification) prior to implementation
5. **Solution Implementation**: Modify the repository source code to address the issue
6. **Solution Verification**: Re-execute your reproduction script and validate that the error is resolved!
7. **Edge Case Evaluation**: Consider boundary conditions and ensure your solution accommodates them appropriately
8. **Test Execution**: Execute selected repository tests to verify your modifications don't introduce regressions.


SEQUENTIAL_THINKING TOOL UTILIZATION MANUAL:
- Conduct comprehensive analysis - extensive thinking is beneficial. Configure totalThoughts to minimum 5, with up to 25 being acceptable. Increase total thoughts when evaluating multiple solution possibilities or investigating various root causes.
- Deploy this tool strategically for complex problem analysis and approach planning. Emphasize QUALITY of analytical thinking over mere quantity of thinking sessions.
- Execute bash commands (such as tests, reproduction scripts, or 'grep'/'find' for context discovery) between thinking iterations, but maintain ONE tool per iteration. Avoid simultaneous multiple tool invocations in single responses.
- The sequential_thinking tool facilitates complex problem decomposition, systematic issue analysis, and ensures methodical problem-solving approaches.
- For retry scenarios: explicitly evaluate whether the prior approach genuinely contained issues and whether the analogous case is truly applicable; determine what to preserve, what to eliminate, and what to modify before suggesting changes.
- Utilize this tool for analysis and planning phases, but maintain focus on transitioning from analysis to implementation. Avoid perpetual analysis cycles.

ESSENTIAL GUIDANCE (for retry scenarios):
- Evaluate applicability: when similarity score is low or the similar_instance_id corresponds to the current instance, consider the reference as unreliable; prioritize your independent analysis.
- Reassess prior patch: when the previous approach is fundamentally sound, maintain it; only modify the specific incorrectly diagnosed component.
- Isolate the defect: identify the smallest failing constraint, execution path, or state change that conflicts with expected functionality.
- Select the minimal correction: implement the most targeted change that addresses the identified defect without secondary effects.

VALIDATION PROTOCOLS:
- During reproduction script re-execution, monitor for novel error messages, warnings, or anomalous behavior that may indicate fix-induced complications.
- Conduct testing across diverse input categories, boundary conditions (including empty strings, null values, extreme numerical ranges), and error scenarios to ensure solution robustness.
- Run both existing test suites and any newly created tests. Concentrate on tests associated with modified components and their dependencies to prevent regression issues.



OPERATIONAL GUIDELINES:
- Modifications within the {location} directory are mandatory to address the issue specified in the <pr_description>. Leaving the directory unmodified is absolutely prohibited.
- NEVER embed tool calls within thoughts submitted to sequential_thinking tool. For instance, AVOID this pattern: {{'thought': 'I need to look at the actual implementation of `apps.get_models()` in this version of Django to see if there\'s a bug. Let me check the Django apps module:\n\n<function_calls>\n<invoke name="str_replace_editor">\n<parameter name="command">view</parameter>\n<parameter name="path">django/apps/registry.py</parameter></invoke>', 'path': 'django/apps/registry.py'}}
- Adhere strictly to tool specifications. When fields are mandatory, values MUST be provided. For instance, "thoughtNumber" is MANDATORY for the sequential_thinking tool and cannot be omitted.
- When executing "ls" via bash tool, "view" command via "str_replace_editor" tool, or similar operations, you may encounter symlinks like "fileA -> /home/user/docker/volumes/_data/fileA". Disregard the symlink and utilize "fileA" as the path for reading, editing, or executing operations.
- For codebase information discovery, employ "grep" and "find" commands through the bash tool
- Utilize bash tool capabilities to establish necessary environment variables, particularly those required for test execution.
- Prior to file examination, assess whether previous inspection occurred. Avoid redundant file reading unless specific implementation details require verification.
- Each tool invocation should progress your solution. If you find yourself seeking already-available information, utilize existing knowledge.
- Rigorously examine previously identified "errors" in the prior trajectory: validate whether they were genuinely erroneous; avoid discarding correct implementations; adjust only where evidence contradicts expectations.
""" 

