"""Planner for selecting evidence tools."""

import json
from typing import Iterable, List

try:
    from .api import get_json_result
    from .constants import (
        TOOL_CLASSIFICATION,
        TOOL_EXPERT_FEATURES,
        TOOL_EXPERT_LABELS,
        TOOL_NAMES,
        TOOL_NEIGHBORS,
        TOOL_TREND,
    )
except ImportError:  # Support ``python Facies/main.py``.
    from api import get_json_result
    from constants import (
        TOOL_CLASSIFICATION,
        TOOL_EXPERT_FEATURES,
        TOOL_EXPERT_LABELS,
        TOOL_NAMES,
        TOOL_NEIGHBORS,
        TOOL_TREND,
    )


PLANNER_SYSTEM_PROMPT = """You are a planning agent for well-log facies classification.
Return only valid JSON in this form:
{"tools": [{"name": "tool_name", "why": "brief reason"}]}
Use only tool names explicitly listed in the user prompt. Select between one and five tools.
"""

TOOL_DESCRIPTIONS = {
    TOOL_EXPERT_FEATURES: "explains the well-log features used by the classifier",
    TOOL_EXPERT_LABELS: "provides domain descriptions of the lithofacies labels",
    TOOL_CLASSIFICATION: "provides rule-based lithofacies classification heuristics",
    TOOL_TREND: "analyzes vertical trends in preceding, target, and following intervals",
    TOOL_NEIGHBORS: "retrieves similar labeled observations from training wells",
}


def build_tool_select_prompt(table_str: str, available_tools: Iterable[str]) -> str:
    tool_lines = "\n".join(
        f"- {name}: {TOOL_DESCRIPTIONS[name]}"
        for name in available_tools
        if name in TOOL_DESCRIPTIONS
    )
    return f"""Select complementary evidence tools for the following interval.

Available tools:
{tool_lines}

Well-log data (the initial prediction is intentionally hidden):
{table_str}

Return only the requested JSON object.
"""


def get_tool_selection(table_str: str, available_tools: Iterable[str] = TOOL_NAMES):
    allowed = tuple(name for name in available_tools if name in TOOL_NAMES)
    if not allowed:
        raise ValueError("At least one evidence tool must be available.")

    planner_prompt = build_tool_select_prompt(table_str, allowed)
    think, answer, usage = get_json_result(planner_prompt, PLANNER_SYSTEM_PROMPT)
    try:
        tools_json = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Planner returned invalid JSON: {exc}") from exc

    raw_tools = tools_json.get("tools", [])
    if not isinstance(raw_tools, list):
        raise ValueError("Planner field 'tools' must be a list.")

    selected: List[str] = []
    reasons = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name in allowed and name not in selected:
            selected.append(name)
            reasons.append({"name": name, "why": str(item.get("why", ""))})

    if not selected:
        selected = [TOOL_EXPERT_FEATURES]
        reasons = [{"name": TOOL_EXPERT_FEATURES, "why": "Safe planner fallback."}]

    normalized_answer = json.dumps({"tools": reasons}, ensure_ascii=False)
    return planner_prompt, think, normalized_answer, selected, usage
