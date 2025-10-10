"""Task Execution Prompt

This prompt provides comprehensive guidance for agents working on SWE-bench challenges.
"""

INSTRUCTION_PROMPT = """
<uploaded_files>
{location}
</uploaded_files>
A Python codebase has been made available in the directory {location} (distinct from /tmp/inputs). Please review the following pull request specification:

<pr_description>
{pr_description}
</pr_description>

Your assistance is requested to execute the required modifications within the repository to fulfill the specifications outlined in the <pr_description>.
All test-related modifications mentioned in the <pr_description> have been completed. Therefore, you are NOT required to alter any testing frameworks, test cases, or validation logic!

Your objective is to apply minimal, targeted modifications to production code files within the {location} directory to satisfy the <pr_description> requirements.

Execute the following systematic approach to address the issue:
1. **Problem Comprehension**: Commence with thorough analysis of the PR specification to achieve complete understanding. Determine the fundamental components affected and the desired functionality. Ensure comprehensive problem understanding before advancement.
2. **Repository Investigation**: Initially survey the repository architecture to comprehend essential components and primary files. Prioritize obtaining a comprehensive overview rather than detailed examination.
3. **Issue Replication**: Develop a reproduction script and execute it via `python <filename.py>` using the BashTool to validate the issue
4. **Solution Strategy**: Employ the sequential_thinking tool for fix planning. Examine 5-7 potential problem origins, narrow these to 1-2 most probable causes, then incorporate logging to verify hypotheses before proceeding to actual code implementation
5. **Solution Implementation**: Modify the repository source code to address the identified issue
6. **Solution Validation**: Re-execute your reproduction script to confirm issue resolution!
7. **Edge Case Analysis**: Evaluate boundary conditions and ensure your solution accommodates them appropriately
8. **Test Execution**: Execute relevant repository tests to verify your modifications don't introduce regressions.


SEQUENTIAL_THINKING TOOL UTILIZATION GUIDELINES:
- Conduct comprehensive analysis - extensive thinking is encouraged. Configure totalThoughts to minimum 5, with up to 25 being acceptable. Increase total thoughts when evaluating multiple solution approaches or investigating various root causes.
- Deploy this tool tactically for complex problem analysis and approach planning. Emphasize QUALITY of analytical thinking over mere quantity of thinking sessions.
- Execute bash commands (such as tests, reproduction scripts, or 'grep'/'find' for context discovery) between thinking iterations, but maintain ONE tool per iteration. Avoid simultaneous multiple tool invocations in single responses.
- The sequential_thinking tool facilitates complex problem decomposition, systematic issue analysis, and ensures methodical problem-solving approaches.
- Utilize this tool for analysis and planning phases, but maintain focus on transitioning from analysis to implementation. Avoid perpetual analysis cycles.

VALIDATION PROTOCOLS:
- During reproduction script re-execution, monitor for novel error messages, warnings, or anomalous behavior that may signal fix-induced complications.
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

"""
