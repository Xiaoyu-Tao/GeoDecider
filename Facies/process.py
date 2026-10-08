"""Routing, evidence collection, deliberation, fusion, and refinement."""

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from .api import get_json_result, get_result
    from .constants import (
        FACIES_LABELS,
        MARINE_LABELS,
        NONMARINE_LABELS,
        PREDICTION_COLUMN,
        TOOL_CLASSIFICATION,
        TOOL_EXPERT_FEATURES,
        TOOL_EXPERT_LABELS,
        TOOL_NAMES,
        TOOL_NEIGHBORS,
        TOOL_TREND,
    )
    from .tool_call import get_tool_selection
    from .tools import (
        ClassificationSuggestionsTool,
        ExpertFeatureDescriptionTool,
        ExpertLabelDescriptionTool,
        NeighborFindTool,
        TrendAnalysisTool,
    )
except ImportError:  # Support ``python Facies/main.py``.
    from api import get_json_result, get_result
    from constants import (
        FACIES_LABELS,
        MARINE_LABELS,
        NONMARINE_LABELS,
        PREDICTION_COLUMN,
        TOOL_CLASSIFICATION,
        TOOL_EXPERT_FEATURES,
        TOOL_EXPERT_LABELS,
        TOOL_NAMES,
        TOOL_NEIGHBORS,
        TOOL_TREND,
    )
    from tool_call import get_tool_selection
    from tools import (
        ClassificationSuggestionsTool,
        ExpertFeatureDescriptionTool,
        ExpertLabelDescriptionTool,
        NeighborFindTool,
        TrendAnalysisTool,
    )


ALLOWED_LABELS = frozenset(FACIES_LABELS)


def parse_answer(answer_str: str) -> Tuple[List[str], Dict[str, Any]]:
    try:
        obj = json.loads(answer_str)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("answer"), list):
        raise ValueError("Model output must contain an 'answer' list.")
    return [str(value) for value in obj["answer"]], obj


def validate_labels(labels: Sequence[str], expected_length: int) -> None:
    if len(labels) != expected_length:
        raise ValueError(
            f"Expected {expected_length} labels, received {len(labels)}."
        )
    invalid = sorted(set(labels) - ALLOWED_LABELS)
    if invalid:
        raise ValueError(f"Unsupported facies labels: {invalid}")


def route_window(meta_data: Dict[str, Any]) -> bool:
    """Return whether the complete interval should enter the slow stage."""

    config = meta_data.get("config", {})
    window_df: pd.DataFrame = meta_data["window_df"]
    prediction_column = config.get("prediction_column", PREDICTION_COLUMN)
    confidence_column = config.get("confidence_column", "Prediction_Confidence")
    threshold = float(config.get("routing_threshold", 0.7))

    route: Dict[str, Any] = {
        "enabled": bool(config.get("routing_enabled", True)),
        "threshold": threshold,
        "confidence_column": confidence_column,
        "anchor_offsets": [],
    }

    if not route["enabled"]:
        route.update(activated=True, reason="routing_disabled_full_deliberation")
        meta_data["routing"] = route
        return True

    if confidence_column in window_df.columns:
        scores = pd.to_numeric(window_df[confidence_column], errors="coerce")
        score_source = confidence_column
    else:
        probability_prefix = str(config.get("probability_prefix", "Prob_"))
        probability_columns = [
            column
            for column in window_df.columns
            if column.startswith(probability_prefix)
        ]
        if probability_columns:
            scores = window_df[probability_columns].apply(
                pd.to_numeric, errors="coerce"
            ).max(axis=1)
            score_source = f"max({', '.join(probability_columns)})"
        else:
            route.update(
                activated=True,
                reason="confidence_unavailable_safe_slow_path",
                score_source=None,
            )
            meta_data["routing"] = route
            return True

    score_values = [None if pd.isna(value) else float(value) for value in scores]
    anchors = [
        offset
        for offset, value in enumerate(score_values)
        if value is None or value < threshold
    ]
    activated = bool(anchors)
    route.update(
        activated=activated,
        reason="low_score_anchor" if activated else "all_points_confident",
        score_source=score_source,
        scores=score_values,
        anchor_offsets=anchors,
        anchor_fraction=(len(anchors) / len(window_df) if len(window_df) else 0.0),
    )

    if not activated:
        try:
            labels = [str(value) for value in window_df[prediction_column].tolist()]
            validate_labels(labels, len(window_df))
        except (KeyError, ValueError) as exc:
            route.update(
                activated=True,
                reason=f"invalid_fast_predictions_safe_slow_path: {exc}",
            )
            activated = True

    meta_data["routing"] = route
    return activated


def enforce_nm_m_consistency(
    meta_data: Dict[str, Any], labels: Sequence[str]
) -> Tuple[List[str], Dict[str, Any]]:
    window_df: pd.DataFrame = meta_data["window_df"]
    if "NM_M" not in window_df.columns:
        return list(labels), {"num_corrections": 0, "details": [], "applied": False}

    prediction_column = meta_data.get("config", {}).get(
        "prediction_column", PREDICTION_COLUMN
    )
    predictions = (
        [str(value) for value in window_df[prediction_column].tolist()]
        if prediction_column in window_df.columns
        else [None] * len(window_df)
    )
    fixed = list(labels)
    corrections = []

    for offset, (environment, label, initial) in enumerate(
        zip(window_df["NM_M"].tolist(), labels, predictions)
    ):
        if environment == 1:
            allowed = NONMARINE_LABELS
            fallback = "Nonmarine sandstone"
            environment_name = "non-marine"
        elif environment == 2:
            allowed = MARINE_LABELS
            fallback = "Marine siltstone and shale"
            environment_name = "marine"
        else:
            continue
        if label in allowed:
            continue
        replacement = initial if initial in allowed else fallback
        fixed[offset] = replacement
        corrections.append(
            {
                "offset": offset,
                "environment": environment_name,
                "old_label": label,
                "new_label": replacement,
                "evidence": f"NM_M={environment!r}",
            }
        )

    return fixed, {
        "applied": True,
        "num_corrections": len(corrections),
        "details": corrections,
    }


def aggregate_panel(
    label_lists: Dict[str, List[str]], expected_length: int
) -> Tuple[List[str], List[float], float]:
    valid = {}
    for name, labels in label_lists.items():
        try:
            validate_labels(labels, expected_length)
        except ValueError:
            continue
        valid[name] = labels
    if not valid:
        raise ValueError("No panel member returned a valid complete prediction.")

    final_labels = []
    agreement_per_depth = []
    for offset in range(expected_length):
        votes = [labels[offset] for labels in valid.values()]
        counts = {label: votes.count(label) for label in set(votes)}
        best_count = max(counts.values())
        candidates = {label for label, count in counts.items() if count == best_count}
        # An explicit deterministic tie policy avoids depending on dict insertion order.
        preferred = valid.get("model_aware")
        selected = (
            preferred[offset]
            if preferred is not None and preferred[offset] in candidates
            else sorted(candidates)[0]
        )
        final_labels.append(selected)
        agreement_per_depth.append(best_count / len(votes))

    global_agreement = sum(agreement_per_depth) / len(agreement_per_depth)
    return final_labels, agreement_per_depth, global_agreement


def process_logic_part1(meta_data: Dict[str, Any]) -> Dict[str, Any]:
    window_df: pd.DataFrame = meta_data["window_df"]
    table_str = window_df[meta_data["feature_columns"]].to_string(index=False)
    available_tools = list(TOOL_NAMES)
    if not meta_data.get("config", {}).get("neighbor_reference"):
        available_tools.remove(TOOL_NEIGHBORS)
    prompt, think, answer, selected, usage = get_tool_selection(
        table_str, available_tools
    )
    meta_data.update(
        tool_call_prompt=prompt,
        tool_call_think=think,
        tool_call_answer=answer,
        tool_call_list=selected,
    )
    meta_data.setdefault("llm_calls", []).append({"stage": "planner", **usage})
    return meta_data


def build_base_decision_prompt(meta_data: Dict[str, Any]) -> str:
    window_df: pd.DataFrame = meta_data["window_df"]
    config = meta_data.get("config", {})
    prediction_column = config.get("prediction_column", PREDICTION_COLUMN)
    display_columns = list(meta_data["feature_columns"])
    if prediction_column in window_df.columns:
        display_columns.append(prediction_column)

    parts = [
        "Classify every row of the target interval into one of the nine lithofacies classes.",
    ]
    for key in (
        "expert_feature_description",
        "expert_label_description",
        "classification_suggestions",
        "trend_analysis_answer",
        "neighbor_evidence",
    ):
        if meta_data.get(key):
            parts.append(f"## {key}\n{meta_data[key]}")

    history = meta_data.get("history", [])
    if history:
        parts.append(
            "## Finalized preceding predictions (soft context only)\n"
            + json.dumps(history, ensure_ascii=False)
        )

    parts.append(
        "## Target interval\n" + window_df[display_columns].to_string(index=False)
    )
    parts.append(
        "NM_M=1 permits only the three nonmarine classes. NM_M=2 permits the six "
        "marine classes, including Marine siltstone and shale and Mudstone."
    )
    parts.append(
        'Return {"answer": ["X1", "X2", ...]} with exactly one valid label per row.'
    )
    return "\n\n".join(parts)


def build_decision_prompt(style: str, meta_data: Dict[str, Any]) -> str:
    preferences = {
        "expert": (
            "Prioritize feature definitions and geological heuristics. Treat the "
            "initial prediction as a weak prior."
        ),
        "model_aware": (
            "Treat the initial prediction as a strong prior and revise it only when "
            "multiple evidence sources conflict with it."
        ),
        "trend_focus": (
            "Prioritize supported vertical trends and boundaries. Do not smooth away "
            "transitions that are supported by coordinated log changes."
        ),
    }
    return (
        build_base_decision_prompt(meta_data)
        + f"\n\n## View: {style}\n{preferences[style]}"
    )


def _call_panel_member(
    style: str, meta_data: Dict[str, Any]
) -> Tuple[str, str, List[str], Dict[str, Any]]:
    prompt = build_decision_prompt(style, meta_data)
    think, answer, usage = get_result(prompt)
    meta_data.setdefault("llm_calls", []).append(
        {"stage": f"panel_{style}", **usage}
    )
    labels, parsed = parse_answer(answer)
    validate_labels(labels, len(meta_data["window_df"]))
    return prompt, think, labels, {"raw": parsed, "usage": usage}


def _build_reconciliation_prompt(
    meta_data: Dict[str, Any], panel_outputs: Dict[str, Any], majority: List[str]
) -> str:
    candidates = {
        name: output["labels"] for name, output in panel_outputs.items() if output.get("valid")
    }
    evidence = {
        key: meta_data[key]
        for key in (
            "trend_analysis_answer",
            "neighbor_evidence",
            "history",
        )
        if meta_data.get(key)
    }
    return "\n\n".join(
        [
            "Reconcile the three candidate interpretations using the supplied evidence.",
            "The majority result is a candidate, not a mandatory choice. Select an "
            "alternative only when the interval evidence supports it.",
            "## Candidate labels\n" + json.dumps(candidates, ensure_ascii=False),
            "## Majority candidate\n" + json.dumps(majority, ensure_ascii=False),
            "## Evidence\n" + json.dumps(evidence, ensure_ascii=False),
            "## Target data\n"
            + meta_data["window_df"][meta_data["feature_columns"]].to_string(index=False),
            'Return {"answer": [...], "decisions": [{"offset": 0, '
            '"selected": "...", "support": ["specific evidence"]}]}. Include a '
            "decision record for every offset where the selected label differs from "
            "the majority candidate.",
        ]
    )


def reconcile_panel(
    meta_data: Dict[str, Any], panel_outputs: Dict[str, Any], majority: List[str]
) -> Tuple[List[str], str, str]:
    prompt = _build_reconciliation_prompt(meta_data, panel_outputs, majority)
    system = (
        "You are an evidence-aware lithofacies reconciler. Return only JSON. "
        "The answer must contain exactly one valid facies label per target row."
    )
    think, answer, usage = get_json_result(prompt, system)
    meta_data.setdefault("llm_calls", []).append(
        {"stage": "evidence_reconciliation", **usage}
    )
    try:
        labels, parsed = parse_answer(answer)
        validate_labels(labels, len(majority))
        decisions = parsed.get("decisions", [])
        supported_offsets = set()
        if isinstance(decisions, list):
            for item in decisions:
                if not isinstance(item, dict):
                    continue
                try:
                    offset = int(item["offset"])
                except (KeyError, TypeError, ValueError):
                    continue
                support = item.get("support", [])
                if (
                    0 <= offset < len(labels)
                    and item.get("selected") == labels[offset]
                    and isinstance(support, list)
                    and any(str(value).strip() for value in support)
                ):
                    supported_offsets.add(offset)
        unsupported = [
            offset
            for offset, (old, new) in enumerate(zip(majority, labels))
            if old != new and offset not in supported_offsets
        ]
        if unsupported:
            raise ValueError(
                f"Reconciliation changed offsets without cited support: {unsupported}"
            )
        meta_data["reconciliation"] = {
            "prompt": prompt,
            "think": think,
            "answer": parsed,
            "fallback": False,
        }
        return labels, prompt, think
    except ValueError as exc:
        meta_data["reconciliation"] = {
            "prompt": prompt,
            "think": think,
            "answer": answer,
            "fallback": True,
            "fallback_reason": str(exc),
        }
        return majority, prompt, think


def refine_with_evidence(
    meta_data: Dict[str, Any], fused_labels: List[str]
) -> Tuple[List[str], str, str]:
    prompt = "\n\n".join(
        [
            "Check the fused interpretation for boundaries unsupported by coordinated "
            "log changes and for incompatibility with the observed measurements.",
            "Do not revise a label unless the revision record cites specific observed "
            "evidence. Do not change a supported transition merely to make the sequence smooth.",
            "## Fused labels\n" + json.dumps(fused_labels, ensure_ascii=False),
            "## Target data\n"
            + meta_data["window_df"][meta_data["feature_columns"]].to_string(index=False),
            "## Trend evidence\n" + str(meta_data.get("trend_analysis_answer", "Unavailable")),
            'Return {"answer": [...], "revisions": [{"offset": 0, "old_label": '
            '"...", "new_label": "...", "evidence": ["observed support"]}]}.',
        ]
    )
    system = (
        "You perform geology-informed consistency refinement. Return only JSON with "
        "exactly one valid facies label per target row."
    )
    think, answer, usage = get_json_result(prompt, system)
    meta_data.setdefault("llm_calls", []).append(
        {"stage": "geology_refinement", **usage}
    )
    refined = list(fused_labels)
    accepted = []
    rejected = []
    try:
        labels, parsed = parse_answer(answer)
        validate_labels(labels, len(fused_labels))
        revisions = parsed.get("revisions", [])
        if not isinstance(revisions, list):
            revisions = []
        for item in revisions:
            if not isinstance(item, dict):
                continue
            try:
                offset = int(item["offset"])
            except (KeyError, TypeError, ValueError):
                rejected.append({"record": item, "reason": "invalid offset"})
                continue
            evidence = item.get("evidence", [])
            proposed = item.get("new_label")
            if (
                0 <= offset < len(refined)
                and item.get("old_label") == fused_labels[offset]
                and proposed == labels[offset]
                and proposed in ALLOWED_LABELS
                and isinstance(evidence, list)
                and any(str(value).strip() for value in evidence)
            ):
                refined[offset] = proposed
                accepted.append(item)
            else:
                rejected.append({"record": item, "reason": "unsupported revision"})
        meta_data["geology_refinement"] = {
            "prompt": prompt,
            "think": think,
            "raw_answer": parsed,
            "accepted_revisions": accepted,
            "rejected_revisions": rejected,
        }
    except ValueError as exc:
        meta_data["geology_refinement"] = {
            "prompt": prompt,
            "think": think,
            "raw_answer": answer,
            "accepted_revisions": [],
            "rejected_revisions": [{"reason": str(exc)}],
        }
    return refined, prompt, think


def _execute_selected_tools(meta_data: Dict[str, Any]) -> Dict[str, Any]:
    selected = meta_data["tool_call_list"]
    if TOOL_EXPERT_FEATURES in selected:
        meta_data["expert_feature_description"] = ExpertFeatureDescriptionTool().run()
    if TOOL_EXPERT_LABELS in selected:
        meta_data["expert_label_description"] = ExpertLabelDescriptionTool().run()
    if TOOL_CLASSIFICATION in selected:
        meta_data["classification_suggestions"] = ClassificationSuggestionsTool().run()
    if TOOL_TREND in selected:
        meta_data = TrendAnalysisTool().run(meta_data)
    if TOOL_NEIGHBORS in selected:
        config = meta_data["config"]
        meta_data = NeighborFindTool(
            config["neighbor_reference"],
            label_column=config.get("neighbor_label_column", "Facies"),
            well_column=config.get("neighbor_well_column"),
            k=int(config.get("neighbor_k", 3)),
        ).run(meta_data)
    return meta_data


def _fast_path(meta_data: Dict[str, Any]):
    prediction_column = meta_data["config"].get(
        "prediction_column", PREDICTION_COLUMN
    )
    labels = [str(value) for value in meta_data["window_df"][prediction_column].tolist()]
    validate_labels(labels, len(meta_data["window_df"]))
    fixed, environment_info = enforce_nm_m_consistency(meta_data, labels)
    meta_data["env_consistency"] = environment_info
    meta_data["final_source"] = "fast_classifier"
    meta_data["panel_aggregation"] = {
        "final_labels_before_env_fix": labels,
        "final_labels": fixed,
    }
    prompt = "Fast path: all confidence scores met the routing threshold."
    trace = json.dumps(
        {"source": "fast_classifier", "env_consistency": environment_info},
        ensure_ascii=False,
    )
    return prompt, trace, json.dumps({"answer": fixed}, ensure_ascii=False), meta_data


def process_logic(meta_data: Dict[str, Any]):
    meta_data.setdefault("llm_calls", [])
    if not route_window(meta_data):
        return _fast_path(meta_data)

    meta_data = process_logic_part1(meta_data)
    meta_data = _execute_selected_tools(meta_data)

    panel_outputs: Dict[str, Dict[str, Any]] = {}
    label_lists: Dict[str, List[str]] = {}
    for style in ("expert", "model_aware", "trend_focus"):
        try:
            prompt, think, labels, extra = _call_panel_member(style, meta_data)
            panel_outputs[style] = {
                "prompt": prompt,
                "think": think,
                "answer": extra["raw"],
                "labels": labels,
                "valid": True,
            }
            label_lists[style] = labels
        except ValueError as exc:
            panel_outputs[style] = {"valid": False, "error": str(exc)}

    meta_data["panel"] = panel_outputs
    majority, agreement, global_agreement = aggregate_panel(
        label_lists, len(meta_data["window_df"])
    )
    meta_data["panel_aggregation"] = {
        "majority_candidate": majority,
        "agreement_per_depth": agreement,
        "global_agreement": global_agreement,
    }

    fused, final_prompt, final_think = reconcile_panel(
        meta_data, panel_outputs, majority
    )
    if meta_data.get("config", {}).get("refinement_enabled", True):
        fused, final_prompt, final_think = refine_with_evidence(meta_data, fused)

    fixed, environment_info = enforce_nm_m_consistency(meta_data, fused)
    meta_data["env_consistency"] = environment_info
    meta_data["panel_aggregation"].update(
        final_labels_before_env_fix=fused,
        final_labels=fixed,
    )
    meta_data["final_source"] = "slow_stage"
    final_think = final_think + "\n\n" + json.dumps(
        {"deterministic_env_consistency": environment_info}, ensure_ascii=False
    )
    return (
        final_prompt,
        final_think,
        json.dumps({"answer": fixed}, ensure_ascii=False),
        meta_data,
    )
