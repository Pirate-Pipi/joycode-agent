"""
Advanced Sequential Thinking System

Features:
- Dynamic thought progression with adaptive branching
- Comprehensive thought history and revision tracking
- Rich visual formatting for enhanced readability
- Flexible validation and error handling
- Advanced logging and debugging capabilities
"""

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

from utils.common import ConversationFlow, LLMTool, ToolImplOutput

# Enhanced logging configuration
logger = logging.getLogger(__name__)


class ThoughtData(TypedDict, total=False):
    """
    🧩 Comprehensive Thought Data Structure

    Defines the complete schema for individual thoughts in the sequential
    thinking process, supporting advanced features like branching and revision.

    Attributes:
        thought: The actual thought content/reasoning
        thoughtNumber: Sequential position in the thought chain
        totalThoughts: Expected total number of thoughts in the sequence
        isRevision: Whether this thought revises a previous one
        revisesThought: ID of the thought being revised (if applicable)
        branchFromThought: ID of the thought this branches from
        branchId: Unique identifier for this thought branch
        needsMoreThoughts: Whether additional thoughts are needed
        nextThoughtNeeded: Whether the thinking process should continue
    """

    thought: str
    thoughtNumber: int
    totalThoughts: int
    isRevision: Optional[bool]
    revisesThought: Optional[int]
    branchFromThought: Optional[int]
    branchId: Optional[str]
    needsMoreThoughts: Optional[bool]
    nextThoughtNeeded: bool


class SequentialThinkingTool(LLMTool):
    """
    🧠 Advanced Sequential Thinking Engine

    A sophisticated cognitive tool that enables structured, multi-step reasoning
    through dynamic thought processes. Supports complex problem decomposition,
    iterative refinement, and branching exploration paths.

    Key Capabilities:
    - Dynamic thought progression with adaptive planning
    - Thought revision and refinement mechanisms
    - Branching exploration for alternative approaches
    - Comprehensive history tracking and analysis
    - Rich visual formatting for enhanced comprehension
    - Flexible validation and error recovery
    """

    name = "sequential_thinking"
    description = """
    🎯 Dynamic Sequential Reasoning Tool

    Enables sophisticated multi-step problem analysis through structured thinking.
    Perfect for complex problem decomposition, strategic planning, and iterative refinement.

    ✨ Features:
    • Adaptive thought progression with branching
    • Revision and refinement capabilities
    • Rich visual thought formatting
    • Comprehensive history tracking

    💡 Use when you need to think through complex problems step-by-step!
    """

    input_schema = {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "🧠 Current reasoning step or analytical insight"
            },
            "nextThoughtNeeded": {
                "type": "boolean",
                "description": "🔄 Whether additional thinking is required",
            },
            "thoughtNumber": {
                "type": "integer",
                "description": "📍 Sequential position in thought chain",
                "minimum": 1,
            },
            "totalThoughts": {
                "type": "integer",
                "description": "📊 Current estimate of total thoughts needed",
                "minimum": 1,
            },
            "isRevision": {
                "type": "boolean",
                "description": "🔄 Indicates revision of previous thinking",
            },
            "revisesThought": {
                "type": "integer",
                "description": "🎯 Target thought number for revision",
                "minimum": 1,
            },
            "branchFromThought": {
                "type": "integer",
                "description": "🌿 Origin point for thought branching",
                "minimum": 1,
            },
            "branchId": {
                "type": "string",
                "description": "🏷️ Unique identifier for thought branch"
            },
            "needsMoreThoughts": {
                "type": "boolean",
                "description": "➕ Flag for requiring additional thoughts",
            },
        },
        "required": ["thought", "nextThoughtNeeded", "thoughtNumber", "totalThoughts"],
    }

    def __init__(self, verbose: bool = False):
        """
        🚀 Initialize Advanced Sequential Thinking Engine

        Sets up the thinking tool with comprehensive state management,
        history tracking, and branching capabilities.

        Args:
            verbose: Enable detailed logging and debug output
        """
        super().__init__()

        # Core thinking state management
        self.thought_history: List[ThoughtData] = []
        self.branches: Dict[str, List[ThoughtData]] = {}
        self.verbose = verbose

        # Enhanced logging setup
        if verbose:
            logger.setLevel(logging.DEBUG)
            logger.info("🧠 Sequential thinking tool initialized in verbose mode")

    def _validate_thought_data(self, input_data: Dict[str, Any]) -> ThoughtData:
        """
        🔍 Comprehensive Thought Data Validation

        Performs thorough validation of thought input data, ensuring all
        required fields are present and properly typed for reliable processing.

        Args:
            input_data: Raw input data to validate and structure

        Returns:
            ThoughtData: Validated and structured thought data

        Raises:
            ValueError: When input data fails validation requirements
        """
        # Core field validation with enhanced error messages
        validation_rules = [
            ("thought", str, "🧠 Thought content must be a non-empty string"),
            ("thoughtNumber", int, "📍 Thought number must be a positive integer"),
            ("totalThoughts", int, "📊 Total thoughts must be a positive integer"),
            ("nextThoughtNeeded", bool, "🔄 Next thought needed must be boolean"),
        ]

        for field, expected_type, error_msg in validation_rules:
            value = input_data.get(field)
            if not value and expected_type != bool:
                raise ValueError(f"{error_msg} (missing)")
            if not isinstance(value, expected_type):
                raise ValueError(f"{error_msg} (wrong type: {type(value)})")

        # Construct validated thought data
        return {
            "thought": input_data["thought"],
            "thoughtNumber": input_data["thoughtNumber"],
            "totalThoughts": input_data["totalThoughts"],
            "nextThoughtNeeded": input_data["nextThoughtNeeded"],
            "isRevision": input_data.get("isRevision"),
            "revisesThought": input_data.get("revisesThought"),
            "branchFromThought": input_data.get("branchFromThought"),
            "branchId": input_data.get("branchId"),
            "needsMoreThoughts": input_data.get("needsMoreThoughts"),
        }

    def _format_thought(self, thought_data: ThoughtData) -> str:
        """
        🎨 Advanced Thought Formatting Engine

        Creates visually appealing, structured representations of thoughts
        with contextual indicators for revisions, branches, and progress.

        Args:
            thought_data: Structured thought data to format

        Returns:
            str: Beautifully formatted thought display with visual indicators
        """
        # Extract core thought components
        thought_number = thought_data["thoughtNumber"]
        total_thoughts = thought_data["totalThoughts"]
        thought_content = thought_data["thought"]

        # Determine thought type and context
        thought_type, context_info = self._determine_thought_context(thought_data)

        # Build formatted display
        return self._build_thought_display(
            thought_type, thought_number, total_thoughts,
            context_info, thought_content
        )

    def _determine_thought_context(self, thought_data: ThoughtData) -> tuple[str, str]:
        """Determine thought type and contextual information."""
        is_revision = thought_data.get("isRevision", False)
        revises_thought = thought_data.get("revisesThought")
        branch_from_thought = thought_data.get("branchFromThought")
        branch_id = thought_data.get("branchId")

        if is_revision:
            return "🔄 Revision", f" (revising thought {revises_thought})"
        elif branch_from_thought:
            return "🌿 Branch", f" (from thought {branch_from_thought}, ID: {branch_id})"
        else:
            return "💭 Thought", ""

    def _build_thought_display(
        self, thought_type: str, number: int, total: int,
        context: str, content: str
    ) -> str:
        """Build the visual thought display with borders and formatting."""
        header = f"{thought_type} {number}/{total}{context}"

        # Calculate optimal border width for visual appeal
        content_lines = content.split('\n')
        max_width = max(len(header), max(len(line) for line in content_lines))
        border_width = max_width + 4
        border = "─" * border_width

        # Format multi-line content properly
        formatted_lines = []
        for line in content_lines:
            formatted_lines.append(f"│ {line.ljust(border_width)} │")

        return f"""
┌{border}┐
│ {header.ljust(border_width)} │
├{border}┤
{chr(10).join(formatted_lines)}
└{border}┘"""

    def run_impl(
        self,
        tool_input: Dict[str, Any],
        dialog_messages: Optional[ConversationFlow] = None,
    ) -> ToolImplOutput:
        """
        🚀 Execute Sequential Thinking Process

        Orchestrates the complete thinking workflow including validation,
        history management, branching, formatting, and response generation.

        Args:
            tool_input: Raw thought input data for processing
            dialog_messages: Optional dialog context (maintained for compatibility)

        Returns:
            ToolImplOutput: Comprehensive thinking result with metadata
        """
        try:
            # Validate and structure input data
            validated_thought = self._validate_thought_data(tool_input)

            # Dynamic thought count adjustment
            self._adjust_thought_totals(validated_thought)

            # Update thinking state and history
            self._update_thinking_state(validated_thought)

            # Generate and display formatted thought
            formatted_display = self._format_thought(validated_thought)
            self._log_thought_display(formatted_display)

            # Generate comprehensive response
            return self._generate_response(validated_thought)

        except Exception as e:
            return self._handle_processing_error(e)

    def _adjust_thought_totals(self, thought_data: ThoughtData) -> None:
        """Dynamically adjust total thought count if exceeded."""
        current_number = thought_data["thoughtNumber"]
        current_total = thought_data["totalThoughts"]

        if current_number > current_total:
            thought_data["totalThoughts"] = current_number
            if self.verbose:
                logger.info(f"📊 Adjusted total thoughts to {current_number}")

    def _update_thinking_state(self, thought_data: ThoughtData) -> None:
        """Update comprehensive thinking state including history and branches."""
        # Add to main thought history
        self.thought_history.append(thought_data)

        # Handle branch management
        self._manage_thought_branches(thought_data)

    def _manage_thought_branches(self, thought_data: ThoughtData) -> None:
        """Manage thought branching and branch tracking."""
        branch_from = thought_data.get("branchFromThought")
        branch_id = thought_data.get("branchId")

        if branch_from and branch_id:
            if branch_id not in self.branches:
                self.branches[branch_id] = []
            self.branches[branch_id].append(thought_data)

            if self.verbose:
                logger.info(f"🌿 Created branch '{branch_id}' from thought {branch_from}")

    def _log_thought_display(self, formatted_display: str) -> None:
        """Log the formatted thought display."""
        if self.verbose:
            logger.warning(formatted_display)  # Using warning level for visibility

    def _generate_response(self, thought_data: ThoughtData) -> ToolImplOutput:
        """Generate comprehensive response with thinking metadata."""
        response_data = {
            "thoughtNumber": thought_data["thoughtNumber"],
            "totalThoughts": thought_data["totalThoughts"],
            "nextThoughtNeeded": thought_data["nextThoughtNeeded"],
            "branches": list(self.branches.keys()),
            "thoughtHistoryLength": len(self.thought_history),
        }

        progress_msg = f"🧠 Processed thought {thought_data['thoughtNumber']}/{thought_data['totalThoughts']}"

        return ToolImplOutput(
            tool_output=json.dumps(response_data, indent=2),
            tool_result_message=progress_msg,
            auxiliary_data={"thought_data": thought_data},
        )

    def _handle_processing_error(self, error: Exception) -> ToolImplOutput:
        """Handle processing errors with detailed error information."""
        error_response = {
            "error": str(error),
            "status": "failed",
            "error_type": type(error).__name__
        }

        return ToolImplOutput(
            tool_output=json.dumps(error_response, indent=2),
            tool_result_message=f"🚨 Thinking process error: {str(error)}",
            auxiliary_data={"error": str(error)},
        )

    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """
        🎬 Generate Tool Start Message

        Creates an engaging, informative message when the sequential thinking
        tool is activated, providing context about the thinking process.

        Args:
            tool_input: Input data containing thought information

        Returns:
            str: User-friendly start message with thinking context
        """
        thought_number = tool_input.get("thoughtNumber", "?")
        total_thoughts = tool_input.get("totalThoughts", "?")

        # Determine thought type for enhanced messaging
        is_revision = tool_input.get("isRevision", False)
        is_branch = tool_input.get("branchFromThought") is not None

        if is_revision:
            return f"🔄 Revising thought {thought_number}/{total_thoughts} - refining previous reasoning"
        elif is_branch:
            return f"🌿 Branching thought {thought_number}/{total_thoughts} - exploring alternative path"
        else:
            return f"🧠 Processing thought {thought_number}/{total_thoughts} - sequential reasoning in progress"
