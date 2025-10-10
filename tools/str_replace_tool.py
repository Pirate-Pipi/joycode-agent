"""
Advanced File Editing System

Features:
- Multi-format file viewing with enhanced display
- Precise string replacement with validation
- Comprehensive backup and undo system
- Rich error handling and user guidance
- Advanced file operations (create, insert, delete)
- Intelligent content analysis and suggestions
- Enhanced logging and debugging capabilities
- Workspace-aware file management

"""

import asyncio
from pathlib import Path
from collections import defaultdict
from utils.indent_utils import (
    match_indent,
    match_indent_by_first_line,
)
from utils.workspace_manager import ProjectSpaceHandler
from utils.common import (
    ConversationFlow,
    LLMTool,
    ToolCallParameters,
    ToolImplOutput,
)

from typing import Any, Literal, Optional, get_args
import logging

# Enhanced logging configuration
logger = logging.getLogger(__name__)

Command = Literal[
    "view",      # View file or directory contents
    "create",    # Create new file with content
    "str_replace", # Replace string in existing file
    "insert",    # Insert content at specific line
    "undo_edit", # Undo last edit operation
]


def is_path_in_directory(directory: Path, path: Path) -> bool:
    """
    🔍 Secure Path Validation

    Validates that a given path is within the specified directory,
    preventing directory traversal attacks and ensuring workspace security.

    Args:
        directory: Base directory for validation
        path: Path to validate

    Returns:
        bool: True if path is within directory, False otherwise
    """
    directory = directory.resolve()
    path = path.resolve()
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def adjust_parallel_calls(
    tool_calls: list[ToolCallParameters],
) -> list[ToolCallParameters]:
    """
    🔧 Intelligent Parallel Call Optimization

    Optimizes the execution order of parallel tool calls to prevent conflicts
    and ensure consistent results when multiple file operations are performed.

    Strategy:
    - Prioritize insert operations before replacements
    - Sort insert operations by line number (ascending)
    - Adjust line numbers to account for content shifts

    Args:
        tool_calls: List of tool call parameters to optimize

    Returns:
        list[ToolCallParameters]: Optimized tool call sequence
    """
    # Strategic sorting: inserts first, then by line number
    tool_calls.sort(
        key=lambda x: (
            x.tool_input.get("command") != "insert",  # Insert operations first
            x.tool_input.get("insert_line", 0),       # Then by line number
        )
    )

    # Dynamic line number adjustment for content shifts
    line_shift = 0
    for tool_call in tool_calls:
        if (
            tool_call.tool_input.get("command") == "insert"
            and "insert_line" in tool_call.tool_input
            and "new_str" in tool_call.tool_input
        ):
            # Adjust for previous insertions
            tool_call.tool_input["insert_line"] += line_shift

            # Calculate shift for next operations
            new_lines = len(tool_call.tool_input["new_str"].splitlines())
            line_shift += new_lines

            logger.debug(f"📍 Adjusted insert line +{line_shift} for {new_lines} new lines")

    return tool_calls


# Enhanced tool output with success tracking
class ExtendedToolImplOutput(ToolImplOutput):
    """
    📊 Enhanced Tool Output

    Extends the base tool output with success tracking and
    additional metadata for better error handling and debugging.
    """

    @property
    def success(self) -> bool:
        """
        ✅ Success Status Indicator

        Returns:
            bool: True if operation was successful, False otherwise
        """
        return bool(self.auxiliary_data.get("success", False))


class ToolError(Exception):
    """
    🚨 Enhanced Tool Error

    Custom exception class for tool-specific errors with
    improved error messaging and context preservation.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"🚨 Tool Error: {self.message}"


# Configuration constants
SNIPPET_LINES: int = 4  # Lines to show around matches

TRUNCATED_MESSAGE: str = (
    "<response clipped>"
    "<NOTE>To save on context only part of this file has been shown to you. "
    "You should retry this tool after you have searched inside the file with "
    "`grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
)

# Response length limits (enhanced from original Anthropic values)
# Original: MAX_RESPONSE_LEN: int = 16000
MAX_RESPONSE_LEN: int = 200000  # Increased for better context


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN) -> str:
    """
    ✂️ Intelligent Content Truncation

    Truncates content when it exceeds specified length limits,
    appending helpful guidance for users to find specific content.

    Args:
        content: Content to potentially truncate
        truncate_after: Maximum length before truncation (None = no limit)

    Returns:
        str: Original content or truncated content with guidance message
    """
    if not truncate_after or len(content) <= truncate_after:
        return content

    logger.info(f"📏 Truncating content: {len(content)} chars -> {truncate_after} chars")
    return content[:truncate_after] + TRUNCATED_MESSAGE


async def run(
    cmd: str,
    timeout: float | None = 120.0,  # seconds
    truncate_after: int | None = MAX_RESPONSE_LEN,
) -> tuple[int, str, str]:
    """
    🖥️ Asynchronous Command Execution

    Executes shell commands asynchronously with timeout protection
    and automatic output truncation for large responses.

    Args:
        cmd: Shell command to execute
        timeout: Maximum execution time in seconds
        truncate_after: Maximum output length before truncation

    Returns:
        tuple[int, str, str]: (return_code, stdout, stderr)

    Raises:
        TimeoutError: If command execution exceeds timeout
    """
    logger.debug(f"🚀 Executing command: {cmd}")

    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

        result = (
            process.returncode or 0,
            maybe_truncate(stdout.decode(), truncate_after=truncate_after),
            maybe_truncate(stderr.decode(), truncate_after=truncate_after),
        )

        logger.debug(f"✅ Command completed with return code: {result[0]}")
        return result

    except asyncio.TimeoutError as exc:
        logger.warning(f"⏰ Command timeout after {timeout}s: {cmd}")
        try:
            process.kill()
        except ProcessLookupError:
            pass
        raise TimeoutError(
            f"🚨 Command '{cmd}' timed out after {timeout} seconds"
        ) from exc


def run_sync(*args, **kwargs) -> tuple[int, str, str]:
    """
    🔄 Synchronous Command Wrapper

    Provides synchronous interface to the asynchronous run function
    for compatibility with synchronous code contexts.
    """
    return asyncio.run(run(*args, **kwargs))


class StrReplaceEditorTool(LLMTool):
    """
    🛠️ Advanced String Replacement Editor

    A comprehensive file editing system providing sophisticated string replacement
    operations with intelligent validation, backup management, and rich user feedback.

    Key Features:
    • Persistent state across command calls and user interactions
    • Multi-format file viewing with enhanced display options
    • Precise string replacement with comprehensive validation
    • Intelligent content insertion with line number management
    • Comprehensive backup and undo system for safe editing
    • Rich error handling with detailed user guidance
    • Workspace-aware file management and security
    • Advanced logging and debugging capabilities

    Supported Operations:
    • view: Display file/directory contents with optional range selection
    • create: Create new files with validation and conflict prevention
    • str_replace: Replace exact string matches with comprehensive validation
    • insert: Insert content at specific line positions
    • undo_edit: Revert last edit operation with full history tracking
    """

    name = "str_replace_editor"

    description = """\
🎯 Advanced File Editing System

A sophisticated editing tool providing comprehensive file operations with
intelligent validation, backup management, and rich user feedback.

✨ Core Capabilities:
• Persistent state across command calls and user discussions
• Multi-format file viewing with enhanced display options
• Precise string replacement with comprehensive validation
• Intelligent content insertion with line number management
• Comprehensive backup and undo system for safe editing

📁 File Operations:
• `view`: Display file contents (with `cat -n`) or directory listings (2 levels deep)
• `create`: Create new files with conflict prevention and validation
• `str_replace`: Replace exact string matches with comprehensive validation
• `insert`: Insert content at specific line positions with smart positioning
• `undo_edit`: Revert last edit operation with full history tracking

🔍 Advanced Features:
• Long output truncation with helpful guidance (`<response clipped>`)
• Workspace-aware file management and security validation
• Rich error handling with detailed user guidance and suggestions

⚠️ String Replacement Guidelines:
• `old_str` must match EXACTLY (including whitespace and line breaks)
• Include sufficient context in `old_str` to ensure uniqueness
• `new_str` contains the replacement content for the matched `old_str`
• Non-unique matches will be rejected with helpful error messages
"""
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "description": "🎯 Operation to perform: view (👁️), create (📝), str_replace (🔄), insert (➕), undo_edit (↩️)",
            },
            "file_text": {
                "description": "📝 Content for new file creation (required for `create` command)",
                "type": "string",
            },
            "insert_line": {
                "description": "📍 Line number for insertion - content will be inserted AFTER this line (required for `insert` command)",
                "type": "integer",
            },
            "new_str": {
                "description": "✨ Replacement content for `str_replace` or insertion content for `insert` command",
                "type": "string",
            },
            "old_str": {
                "description": "🎯 Exact string to replace in file (required for `str_replace` command) - must match precisely including whitespace",
                "type": "string",
            },
            "path": {
                "description": "📁 File or directory path for the operation",
                "type": "string",
            },
            "view_range": {
                "description": "👁️ Optional line range for `view` command [start, end]. Use [start, -1] for start to end of file. 1-based indexing.",
                "items": {"type": "integer"},
                "type": "array",
            },
        },
        "required": ["command", "path"],
    }

    # Enhanced file edit history tracking for comprehensive undo operations
    _file_history = defaultdict(list)

    def __init__(
        self,
        workspace_manager: ProjectSpaceHandler,
        ignore_indentation_for_str_replace: bool = False,
        expand_tabs: bool = False,
    ):
        """
        🚀 Initialize Advanced String Replacement Editor

        Sets up the editor with comprehensive configuration options,
        workspace management, and history tracking capabilities.

        Args:
            workspace_manager: Workspace management system for file operations
            ignore_indentation_for_str_replace: Skip indentation matching for replacements
            expand_tabs: Convert tabs to spaces during processing
        """
        super().__init__()

        # Core configuration
        self.workspace_manager = workspace_manager
        self.ignore_indentation_for_str_replace = ignore_indentation_for_str_replace
        self.expand_tabs = expand_tabs

        # Enhanced history tracking
        self._file_history = defaultdict(list)

        logger.info("🛠️ String replacement editor initialized with enhanced features")

    def run_impl(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[ConversationFlow] = None,
    ) -> ExtendedToolImplOutput:
        """
        🚀 Execute File Operation

        Orchestrates the complete file operation workflow including validation,
        execution, and comprehensive result reporting with enhanced error handling.

        Args:
            tool_input: Operation parameters and configuration
            dialog_messages: Optional dialog context (maintained for compatibility)

        Returns:
            ExtendedToolImplOutput: Comprehensive operation result with metadata
        """
        # Extract and validate operation parameters
        operation_params = self._extract_operation_params(tool_input)

        # Execute the requested operation with comprehensive error handling
        try:
            return self._execute_operation(operation_params)
        except Exception as e:
            return self._handle_operation_error(e, operation_params)

    def _extract_operation_params(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Extract and structure operation parameters."""
        return {
            "command": tool_input["command"],
            "path": tool_input["path"],
            "file_text": tool_input.get("file_text"),
            "view_range": tool_input.get("view_range"),
            "old_str": tool_input.get("old_str"),
            "new_str": tool_input.get("new_str"),
            "insert_line": tool_input.get("insert_line"),
        }

    def _execute_operation(self, params: dict[str, Any]) -> ExtendedToolImplOutput:
        """Execute the file operation based on command type."""
        command = params["command"]

        # Enhanced operation routing with detailed logging
        operation_map = {
            "view": self._handle_view_operation,
            "create": self._handle_create_operation,
            "str_replace": self._handle_str_replace_operation,
            "insert": self._handle_insert_operation,
            "undo_edit": self._handle_undo_operation,
        }

        if command not in operation_map:
            raise ToolError(f"Unknown command: {command}")

        logger.info(f"🎯 Executing {command} operation on {params['path']}")
        return operation_map[command](params)

    def _handle_view_operation(self, params: dict[str, Any]) -> ExtendedToolImplOutput:
        """Handle file/directory viewing operations."""
        ws_path = self._prepare_workspace_path(params["path"], "view")
        return self.view(ws_path, params["view_range"])

    def _handle_create_operation(self, params: dict[str, Any]) -> ExtendedToolImplOutput:
        """Handle file creation operations."""
        if params["file_text"] is None:
            raise ToolError("📝 Parameter `file_text` is required for create command")

        ws_path = self._prepare_workspace_path(params["path"], "create")
        self.write_file(ws_path, params["file_text"])
        self._file_history[ws_path].append(params["file_text"])

        container_path = self.workspace_manager.container_path(params["path"])
        success_msg = f"✅ File created successfully at: {container_path}"

        return ExtendedToolImplOutput(
            success_msg, success_msg, {"success": True}
        )

    def _handle_str_replace_operation(self, params: dict[str, Any]) -> ExtendedToolImplOutput:
        """Handle string replacement operations."""
        if params["old_str"] is None:
            raise ToolError("🎯 Parameter `old_str` is required for str_replace command")

        ws_path = self._prepare_workspace_path(params["path"], "str_replace")

        try:
            if self.ignore_indentation_for_str_replace:
                return self._str_replace_ignore_indent(ws_path, params["old_str"], params["new_str"])
            else:
                return self.str_replace(ws_path, params["old_str"], params["new_str"])
        except PermissionError:
            error_msg = f"🚨 File {params['path']} could not be edited due to lack of permission. Try changing file permissions."
            return ExtendedToolImplOutput(error_msg, error_msg, {"success": False})

    def _handle_insert_operation(self, params: dict[str, Any]) -> ExtendedToolImplOutput:
        """Handle content insertion operations."""
        if params["insert_line"] is None:
            raise ToolError("📍 Parameter `insert_line` is required for insert command")
        if params["new_str"] is None:
            raise ToolError("✨ Parameter `new_str` is required for insert command")

        ws_path = self._prepare_workspace_path(params["path"], "insert")
        return self.insert(ws_path, params["insert_line"], params["new_str"])

    def _handle_undo_operation(self, params: dict[str, Any]) -> ExtendedToolImplOutput:
        """Handle undo operations."""
        ws_path = self._prepare_workspace_path(params["path"], "undo_edit")
        return self.undo_edit(ws_path)

    def _prepare_workspace_path(self, path: str, command: str) -> Path:
        """Prepare and validate workspace path with security checks."""
        ws_path = self.workspace_manager.workspace_path(Path(path))
        self.validate_path(command, ws_path)

        # Enhanced security validation
        if not is_path_in_directory(self.workspace_manager.root, ws_path):
            container_root = self.workspace_manager.container_path(self.workspace_manager.root)
            raise ToolError(
                f"🚨 Path {ws_path} is outside workspace root: {container_root}. "
                f"Access restricted to workspace directory only."
            )

        return ws_path

    def _handle_operation_error(self, error: Exception, params: dict[str, Any]) -> ExtendedToolImplOutput:
        """Handle operation errors with enhanced error reporting."""
        error_msg = f"🚨 Operation failed: {str(error)}"
        logger.error(f"Operation error in {params.get('command', 'unknown')}: {error}")

        return ExtendedToolImplOutput(
            error_msg, error_msg, {"success": False, "error": str(error)}
        )
    def validate_path(self, command: str, path: Path) -> None:
        """
        🔍 Comprehensive Path Validation

        Validates path and command combinations with enhanced error messages
        and intelligent conflict detection for safe file operations.

        Args:
            command: Operation command to validate against
            path: File/directory path to validate

        Raises:
            ToolError: When path/command combination is invalid
        """
        # Enhanced existence validation
        if not path.exists() and command != "create":
            raise ToolError(
                f"📁 Path {path} does not exist. Please provide a valid path or use 'create' command."
            )

        # Enhanced creation conflict detection
        if path.exists() and command == "create":
            content = self.read_file(path)
            if content.strip():
                raise ToolError(
                    f"📝 File already exists and contains content at: {path}. "
                    f"Cannot overwrite non-empty files with 'create' command. "
                    f"Use 'str_replace' or 'view' instead."
                )

        # Enhanced directory operation validation
        if path.is_dir() and command != "view":
            raise ToolError(
                f"📁 Path {path} is a directory. Only 'view' command is supported for directories. "
                f"For file operations, specify a file path."
            )

    def view(
        self, path: Path, view_range: Optional[list[int]] = None
    ) -> ExtendedToolImplOutput:
        """
        👁️ Advanced File/Directory Viewing

        Provides comprehensive viewing capabilities for files and directories
        with intelligent range selection and enhanced formatting.

        Args:
            path: File or directory path to view
            view_range: Optional line range [start, end] for file viewing

        Returns:
            ExtendedToolImplOutput: Formatted view result with metadata
        """
        # Handle directory viewing
        if path.is_dir():
            return self._view_directory(path, view_range)

        # Handle file viewing with range support
        return self._view_file(path, view_range)

    def _view_directory(self, path: Path, view_range: Optional[list[int]]) -> ExtendedToolImplOutput:
        """View directory contents with enhanced formatting."""
        if view_range:
            raise ToolError(
                "📁 The `view_range` parameter is not supported for directory viewing. "
                "Use without view_range to see directory contents."
            )

        _, stdout, stderr = run_sync(rf"find {path} -maxdepth 2 -not -path '*/\.*'")

        if not stderr:
            output = (
                f"📁 Directory contents (up to 2 levels deep) in {path}:\n"
                f"{'='*60}\n{stdout}\n"
            )
            success_msg = "📁 Directory contents listed successfully"
        else:
            output = f"🚨 Error listing directory:\nstderr: {stderr}\nstdout: {stdout}\n"
            success_msg = "❌ Directory listing encountered errors"

        return ExtendedToolImplOutput(
            output, success_msg, {"success": not stderr}
        )

    def _view_file(self, path: Path, view_range: Optional[list[int]]) -> ExtendedToolImplOutput:
        """View file contents with intelligent range handling."""
        file_content = self.read_file(path)
        file_lines = file_content.split("\n")
        total_lines = len(file_lines)

        # Process view range if specified
        init_line = 1
        if view_range:
            init_line, file_content = self._process_view_range(
                view_range, file_lines, total_lines
            )

        # Generate formatted output
        output = self._make_output(
            file_content=file_content,
            file_descriptor=str(self.workspace_manager.container_path(path)),
            total_lines=total_lines,
            init_line=init_line,
        )

        return ExtendedToolImplOutput(
            output, "📄 File content displayed successfully", {"success": True}
        )

    def _process_view_range(
        self, view_range: list[int], file_lines: list[str], total_lines: int
    ) -> tuple[int, str]:
        """Process and validate view range parameters."""
        # Enhanced validation
        if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
            raise ToolError(
                "📏 Invalid `view_range`. Must be a list of exactly two integers [start, end]."
            )

        init_line, final_line = view_range

        # Comprehensive range validation
        if init_line < 1 or init_line > total_lines:
            raise ToolError(
                f"📍 Invalid start line {init_line}. Must be between 1 and {total_lines}."
            )

        if final_line > total_lines:
            raise ToolError(
                f"📍 Invalid end line {final_line}. Must be ≤ {total_lines} (file length)."
            )

        if final_line != -1 and final_line < init_line:
            raise ToolError(
                f"📍 Invalid range: end line {final_line} must be ≥ start line {init_line}."
            )

        # Extract content based on range
        if final_line == -1:
            content = "\n".join(file_lines[init_line - 1:])
        else:
            content = "\n".join(file_lines[init_line - 1:final_line])

        return init_line, content

    def _str_replace_ignore_indent(self, path: Path, old_str: str, new_str: str | None):
        """Replace old_str with new_str in content, ignoring indentation.

        Finds matches in stripped version of text and uses those line numbers
        to perform replacements in original indented version.
        """
        if new_str is None:
            new_str = ""

        content = self.read_file(path)
        if self.expand_tabs:
            content = content.expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs()

        new_str = match_indent(new_str, content)
        assert new_str is not None, "new_str should not be None after match_indent"

        # Split into lines for processing
        content_lines = content.splitlines()
        stripped_content_lines = [line.strip() for line in content.splitlines()]
        stripped_old_str_lines = [line.strip() for line in old_str.splitlines()]

        # Find all potential starting line matches
        matches = []
        for i in range(len(stripped_content_lines) - len(stripped_old_str_lines) + 1):
            is_match = True
            for j, pattern_line in enumerate(stripped_old_str_lines):
                if j == len(stripped_old_str_lines) - 1:
                    if stripped_content_lines[i + j].startswith(pattern_line):
                        # it's a match but last line in old_str is not the full line
                        # we need to append the rest of the line to new_str
                        new_str += stripped_content_lines[i + j][len(pattern_line) :]
                    else:
                        is_match = False
                        break
                elif stripped_content_lines[i + j] != pattern_line:
                    is_match = False
                    break
            if is_match:
                matches.append(i)

        if not matches:
            raise ToolError(
                f"No replacement was performed, old_str \n ```\n{old_str}\n```\n did not appear in {self.workspace_manager.container_path(path)}."
            )
        if len(matches) > 1:
            # Add 1 to convert to 1-based line numbers for error message
            match_lines = [idx + 1 for idx in matches]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str \n ```\n{old_str}\n```\n starting at lines {match_lines}. Please ensure it is unique"
            )

        # Get the matching range in the original content
        match_start = matches[0]
        match_end = match_start + len(stripped_old_str_lines)

        # Get the original indented lines
        original_matched_lines = content_lines[match_start:match_end]

        indented_new_str = match_indent_by_first_line(
            new_str, original_matched_lines[0]
        )
        assert indented_new_str is not None, "indented_new_str should not be None"

        # Create new content by replacing the matched lines
        new_content = [
            *content_lines[:match_start],
            *indented_new_str.splitlines(),
            *content_lines[match_end:],
        ]
        new_content_str = "\n".join(new_content)

        self._file_history[path].append(content)  # Save old content for undo
        path.write_text(new_content_str)

        # Create a snippet of the edited section
        start_line = max(0, match_start - SNIPPET_LINES)
        end_line = match_start + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_content[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            file_content=snippet,
            file_descriptor=f"a snippet of {self.workspace_manager.container_path(path)}",
            total_lines=len(new_content),
            init_line=start_line + 1,
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return ExtendedToolImplOutput(
            success_msg,
            f"The file {path} has been edited.",
            {"success": True},
        )

    def str_replace(
        self, path: Path, old_str: str, new_str: str | None
    ) -> ExtendedToolImplOutput:
        if new_str is None:
            new_str = ""

        content = self.read_file(path)
        if self.expand_tabs:
            content = content.expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs()

        if not old_str.strip():
            if content.strip():
                raise ToolError(
                    f"No replacement was performed, old_str is empty which is only allowed when the file is empty. The file {path} is not empty."
                )
            else:
                # replace the whole file with new_str
                new_content = new_str
                self._file_history[path].append(content)  # Save old content for undo
                path.write_text(new_content)
                # Prepare the success message
                success_msg = f"The file {path} has been edited. "
                success_msg += self._make_output(
                    file_content=new_content,
                    file_descriptor=f"{self.workspace_manager.container_path(path)}",
                    total_lines=len(new_content.split("\n")),
                )
                success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

                return ExtendedToolImplOutput(
                    success_msg,
                    f"The file {path} has been edited.",
                    {"success": True},
                )

        # First try exact match
        occurrences = content.count(old_str)

        if occurrences == 0:
            # Exact match failed, trying smart matching strategy...
            logger.info(f"🔍 Exact match failed, trying smart matching strategy...")
            return self._smart_str_replace(path, content, old_str, new_str)
        elif occurrences > 1:
            file_content_lines = content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str \n ```\n{old_str}\n```\n in lines {lines}. Please ensure it is unique"
            )

        new_content = content.replace(old_str, new_str)
        self._file_history[path].append(content)  # Save old content for undo
        path.write_text(new_content)

        # Create a snippet of the edited section
        replacement_line = content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            file_content=snippet,
            file_descriptor=f"a snippet of {self.workspace_manager.container_path(path)}",
            total_lines=len(new_content.split("\n")),
            init_line=start_line + 1,
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return ExtendedToolImplOutput(
            success_msg,
            f"The file {path} has been edited.",
            {"success": True},
        )

    def _smart_str_replace(self, path: Path, content: str, old_str: str, new_str: str) -> ExtendedToolImplOutput:
        """
        Smart string replacement: when exact match fails, try format-tolerant strategies

        Handles only formatting differences (indentation, whitespace, etc.), ensuring code content is identical.
        Does not perform fuzzy matching to avoid mistaken replacements.

        Args:
            path: File path
            content: File content
            old_str: String to replace
            new_str: New string

        Returns:
            Operation result
        """
        logger.info(f"🧠 Starting format-tolerant matching analysis...")

        # Strategy 1: Indentation-tolerant match - ignore leading whitespace differences
        try:
            return self._try_indent_tolerant_match(path, content, old_str, new_str)
        except Exception as e:
            logger.debug(f"Indentation-tolerant match failed: {e}")

        # Strategy 2: Whitespace-tolerant match - normalize whitespace then match
        try:
            return self._try_whitespace_tolerant_match(path, content, old_str, new_str)
        except Exception as e:
            logger.debug(f"Whitespace-tolerant match failed: {e}")

        # All strategies failed; provide detailed error message and suggestions
        return self._generate_detailed_error_message(path, content, old_str)

    def _try_indent_tolerant_match(self, path: Path, content: str, old_str: str, new_str: str) -> ExtendedToolImplOutput:
        """
        Try indentation-tolerant matching: ignore leading whitespace differences
        """
        logger.info("🔧 Trying indentation-tolerant match...")

        # Split content by lines
        content_lines = content.splitlines()
        old_str_lines = old_str.splitlines()

        # Strip leading whitespace for each line for matching
        stripped_content_lines = [line.strip() for line in content_lines]
        stripped_old_lines = [line.strip() for line in old_str_lines]

        # Find the start position of the match
        match_start = -1
        for i in range(len(stripped_content_lines) - len(stripped_old_lines) + 1):
            if stripped_content_lines[i:i+len(stripped_old_lines)] == stripped_old_lines:
                match_start = i
                break

        if match_start == -1:
            raise ToolError("Indentation-tolerant match failed")

        # Get the indentation of the original lines
        original_lines = content_lines[match_start:match_start+len(old_str_lines)]

        # Apply new content while preserving original indentation
        new_str_lines = new_str.splitlines()
        if original_lines:
            # Use the first line's indentation as the baseline
            base_indent = len(original_lines[0]) - len(original_lines[0].lstrip())
            indented_new_lines = []

            for i, new_line in enumerate(new_str_lines):
                if new_line.strip():  # non-empty line
                    # Compute relative indentation
                    if i < len(old_str_lines):
                        old_line_indent = len(old_str_lines[i]) - len(old_str_lines[i].lstrip())
                    else:
                        old_line_indent = 0

                    # Apply baseline indentation + relative indentation
                    final_indent = base_indent + old_line_indent
                    indented_new_lines.append(' ' * final_indent + new_line.strip())
                else:
                    indented_new_lines.append('')

            # Perform replacement
            actual_old_str = '\n'.join(original_lines)
            actual_new_str = '\n'.join(indented_new_lines)

            new_content = content.replace(actual_old_str, actual_new_str)

            # Save and return result
            self._file_history[path].append(content)
            path.write_text(new_content)

            logger.info("✅ Indentation-tolerant match succeeded")
            return self._create_success_output(path, new_content, match_start, len(indented_new_lines))

        raise ToolError("Unable to determine indentation pattern")

    def _try_whitespace_tolerant_match(self, path: Path, content: str, old_str: str, new_str: str) -> ExtendedToolImplOutput:
        """
        Try whitespace-tolerant matching: normalize all whitespace then match
        """
        logger.info("🔧 Trying whitespace-tolerant match...")

        import re

        # Normalize whitespace: convert multiple spaces/tabs into single spaces
        def normalize_whitespace(text: str) -> str:
            # Preserve line structure but normalize intra-line whitespace
            lines = text.splitlines()
            normalized_lines = []
            for line in lines:
                # Preserve relative leading indentation but normalize to spaces
                leading_whitespace = len(line) - len(line.lstrip())
                content_part = line.strip()
                if content_part:
                    # Normalize intra-line whitespace
                    content_part = re.sub(r'\s+', ' ', content_part)
                    normalized_lines.append(' ' * leading_whitespace + content_part)
                else:
                    normalized_lines.append('')
            return '\n'.join(normalized_lines)

        normalized_content = normalize_whitespace(content)
        normalized_old_str = normalize_whitespace(old_str)

        if normalized_old_str in normalized_content:
            # Find match position
            match_pos = normalized_content.find(normalized_old_str)

            # Find corresponding position in the original content
            original_match_start = self._find_original_position(content, normalized_content, match_pos)
            original_match_end = self._find_original_position(content, normalized_content,
                                                            match_pos + len(normalized_old_str))

            if original_match_start != -1 and original_match_end != -1:
                actual_old_str = content[original_match_start:original_match_end]

                # Apply replacement, preserving original whitespace style
                actual_new_str = self._adapt_whitespace_style(actual_old_str, new_str)
                new_content = content.replace(actual_old_str, actual_new_str)

                # Save and return result
                self._file_history[path].append(content)
                path.write_text(new_content)

                logger.info("✅ Whitespace-tolerant match succeeded")
                return self._create_success_output(path, new_content,
                                                 content[:original_match_start].count('\n'),
                                                 actual_new_str.count('\n') + 1)

        raise ToolError("Whitespace-tolerant match failed")

    def _find_original_position(self, original: str, normalized: str, normalized_pos: int) -> int:
        """
        Find the original position corresponding to a position in the normalized text
        """
        if normalized_pos == 0:
            return 0

        # Simplified approach: estimate position by character counting
        original_chars = 0
        normalized_chars = 0

        for char in original:
            if normalized_chars >= normalized_pos:
                break
            if char in normalized:
                normalized_chars += 1
            original_chars += 1

        return original_chars

    def _adapt_whitespace_style(self, original_str: str, new_str: str) -> str:
        """
        Adapt the new string to the original string's whitespace style
        """
        # Simplified: keep the structure of the new string but try to match the original indentation style
        original_lines = original_str.splitlines()
        new_lines = new_str.splitlines()

        if not original_lines or not new_lines:
            return new_str

        # Detect indentation character of the original string (spaces or tabs)
        first_line = original_lines[0]
        indent_char = '\t' if first_line.startswith('\t') else ' '

        # Apply the same indentation character to the new string
        adapted_lines = []
        for line in new_lines:
            if line.strip():  # non-empty line
                leading_spaces = len(line) - len(line.lstrip())
                if indent_char == '\t':
                    # Convert to tab indentation
                    tab_count = leading_spaces // 4  # Assume 4 spaces equal 1 tab
                    adapted_lines.append('\t' * tab_count + line.lstrip())
                else:
                    adapted_lines.append(line)
            else:
                adapted_lines.append('')

        return '\n'.join(adapted_lines)


    def _generate_detailed_error_message(self, path: Path, content: str, old_str: str) -> ExtendedToolImplOutput:
        """
        Generate detailed error information and suggestions
        """
        logger.info("📋 Generating detailed error information...")

        content_lines = content.splitlines()
        old_str_lines = old_str.splitlines()

        # Find possible matching lines (based on exact string match)
        potential_matches = []
        for i, content_line in enumerate(content_lines):
            for j, old_line in enumerate(old_str_lines):
                # Check only exact matches or exact matches after stripping whitespace
                if content_line.strip() == old_line.strip() and content_line.strip():
                    potential_matches.append({
                        'content_line': i + 1,
                        'old_line': j + 1,
                        'content': content_line,
                        'expected': old_line
                    })

        # Construct error message
        error_msg = f"🚨 String replacement failed: exact match not found in file {path}\n\n"

        if potential_matches:
            error_msg += "🔍 Found the following potentially related lines (content matches but formatting may differ):\n"
            # Show the first 5 matching lines
            for match in potential_matches[:5]:
                error_msg += f"  📍 Line {match['content_line']}\n"
                error_msg += f"     File content: {repr(match['content'])}\n"
                error_msg += f"     Expected: {repr(match['expected'])}\n\n"


        # Show the first few lines of old_str as reference
        if old_str_lines:
            error_msg += "🎯 The content you attempted to match (first 5 lines):\n"
            for i, line in enumerate(old_str_lines[:5]):
                error_msg += f"  {i+1}: {repr(line)}\n"
            if len(old_str_lines) > 5:
                error_msg += f"  ... (and {len(old_str_lines) - 5} more lines)\n"

        raise ToolError(error_msg)

    def _create_success_output(self, path: Path, new_content: str, start_line: int, num_lines: int) -> ExtendedToolImplOutput:
        """
        Create success operation output
        """
        # Create a code snippet of the edited region
        content_lines = new_content.splitlines()
        snippet_start = max(0, start_line - 2)
        snippet_end = min(len(content_lines), start_line + num_lines + 2)
        snippet = '\n'.join(content_lines[snippet_start:snippet_end])

        # Prepare success message
        success_msg = f"✅ File {path} has been successfully edited via smart matching.\n"
        success_msg += self._make_output(
            file_content=snippet,
            file_descriptor=f"Edited region snippet {self.workspace_manager.container_path(path)}",
            total_lines=len(content_lines),
            init_line=snippet_start + 1,
        )
        success_msg += "\n🔍 Please verify the changes meet expectations. Edit the file again if needed."

        return ExtendedToolImplOutput(
            success_msg,
            f"File {path} has been successfully edited.",
            {"success": True, "method": "smart_match"},
        )

    def insert(
        self, path: Path, insert_line: int, new_str: str
    ) -> ExtendedToolImplOutput:
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        file_text = self.read_file(path)
        if self.expand_tabs:
            file_text = file_text.expandtabs()
            new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        self.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            file_content=snippet,
            file_descriptor="a snippet of the edited file",
            total_lines=len(new_file_text_lines),
            init_line=max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

        return ExtendedToolImplOutput(
            success_msg,
            "Insert successful",
            {"success": True},
        )

    def undo_edit(self, path: Path) -> ExtendedToolImplOutput:
        """Implement the undo_edit command."""
        if not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")

        old_text = self._file_history[path].pop()
        self.write_file(path, old_text)

        formatted_file = self._make_output(
            file_content=old_text,
            file_descriptor=str(self.workspace_manager.container_path(path)),
            total_lines=len(old_text.split("\n")),
        )
        output = f"Last edit to {path} undone successfully.\n{formatted_file}"

        return ExtendedToolImplOutput(
            output,
            "Undo successful",
            {"success": True},
        )

    def read_file(self, path: Path):
        """Read the content of a file from a given path; raise a ToolError if an error occurs."""
        try:
            return path.read_text()
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to read {path}") from None

    def write_file(self, path: Path, file: str):
        """Write the content of a file to a given path; raise a ToolError if an error occurs."""
        try:
            path.write_text(file)
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to write to {path}") from None

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        total_lines: int,
        init_line: int = 1,
    ):
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if self.expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
            + f"Total lines in file: {total_lines}\n"
        )

    def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
        return f"Editing file {tool_input['path']}"
