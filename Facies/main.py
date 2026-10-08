"""Command-line entry point for the Facies GeoDecider pipeline."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from .constants import (
        CONFIDENCE_COLUMN,
        DEFAULT_FEATURE_COLUMNS,
        PREDICTION_COLUMN,
    )
    from .process import process_logic
except ImportError:  # Support ``python Facies/main.py``.
    from constants import CONFIDENCE_COLUMN, DEFAULT_FEATURE_COLUMNS, PREDICTION_COLUMN
    from process import process_logic


def _load_checkpoint(output_jsonl: str, history_size: int) -> Tuple[set, List[Dict[str, Any]]]:
    processed = set()
    history: List[Dict[str, Any]] = []
    if not os.path.exists(output_jsonl):
        return processed, history

    with open(output_jsonl, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                metadata = record["meta_data"]
                window_id = int(metadata["current_window_id"])
                labels = metadata["panel_aggregation"]["final_labels"]
                rows = metadata["window_df"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Cannot resume: invalid JSONL record at line {line_number}: {exc}"
                ) from exc
            if len(labels) != len(rows):
                raise ValueError(
                    f"Cannot resume: line {line_number} has {len(rows)} rows but "
                    f"{len(labels)} labels."
                )
            processed.add(window_id)
            for row, label in zip(rows, labels):
                history.append(
                    {
                        "depth": row.get("Depth"),
                        "label": label,
                        "window_id": window_id,
                    }
                )
    if processed and processed != set(range(1, max(processed) + 1)):
        raise ValueError(
            "Cannot resume from non-contiguous window IDs; this could leak future "
            "predictions into historical context."
        )
    return processed, history[-history_size:]


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient="records"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _serialize_metadata(meta_data: Dict[str, Any]) -> Dict[str, Any]:
    return _json_safe(meta_data)


def _usage_summary(calls: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "llm_call_count": len(calls),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for call in calls:
        usage = call.get("usage", {})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            summary[key] += int(usage.get(key, 0))
    return summary


def main(file_path: str, output_jsonl: str, **options: Any) -> None:
    input_path = Path(file_path).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    feature_columns = list(options.get("feature_columns", DEFAULT_FEATURE_COLUMNS))
    prediction_column = options.get("prediction_column", PREDICTION_COLUMN)
    required = feature_columns + [prediction_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    window_size = int(options.get("window_size", 16))
    step_size = int(options.get("step_size") or window_size)
    history_size = int(options.get("history_size", 32))
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive.")

    config = {
        "prediction_column": prediction_column,
        "confidence_column": options.get("confidence_column", CONFIDENCE_COLUMN),
        "probability_prefix": options.get("probability_prefix", "Prob_"),
        "routing_threshold": float(options.get("routing_threshold", 0.7)),
        "routing_enabled": bool(options.get("routing_enabled", True)),
        "refinement_enabled": bool(options.get("refinement_enabled", True)),
        "neighbor_reference": options.get("neighbor_reference"),
        "neighbor_label_column": options.get("neighbor_label_column", "Facies"),
        "neighbor_well_column": options.get("neighbor_well_column"),
        "neighbor_k": int(options.get("neighbor_k", 3)),
    }
    if config["neighbor_reference"]:
        config["neighbor_reference"] = str(
            Path(config["neighbor_reference"]).expanduser().resolve()
        )

    processed, history = _load_checkpoint(str(output_path), history_size)
    total_windows = (len(df) + step_size - 1) // step_size
    if len(processed) >= total_windows:
        print("All windows have already been processed.")
        return

    with open(output_path, "a", encoding="utf-8") as handle:
        for start in range(0, len(df), step_size):
            window_id = start // step_size + 1
            if window_id in processed:
                continue

            window_df = df.iloc[start : start + window_size].copy()
            print(
                f"Processing window {window_id}/{total_windows} "
                f"(rows {start}-{start + len(window_df) - 1})"
            )
            meta_data: Dict[str, Any] = {
                "feature_columns": feature_columns,
                "current_window_id": window_id,
                "window_df": window_df,
                "window_up": df.iloc[max(0, start - window_size) : start].copy(),
                "window_down": df.iloc[
                    start + window_size : start + (2 * window_size)
                ].copy(),
                "is_full_window": len(window_df) == window_size,
                "history": list(history),
                "config": config,
            }

            prompt, think, answer, meta_data = process_logic(meta_data)
            parsed_answer = json.loads(answer)
            labels = parsed_answer["answer"]
            meta_data["llm_usage_summary"] = _usage_summary(
                meta_data.get("llm_calls", [])
            )
            serializable_metadata = _serialize_metadata(meta_data)
            record = {
                "status": "success",
                "meta_data": serializable_metadata,
                "content": {"prompt": prompt, "think": think, "answer": answer},
            }
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

            for (_, row), label in zip(window_df.iterrows(), labels):
                history.append(
                    {
                        "depth": _json_safe(row.get("Depth")),
                        "label": label,
                        "window_id": window_id,
                    }
                )
            history = history[-history_size:]

    print(f"Done. Results saved in: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GeoDecider on a Facies CSV file.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--prediction-column", default=PREDICTION_COLUMN)
    parser.add_argument("--confidence-column", default=CONFIDENCE_COLUMN)
    parser.add_argument("--probability-prefix", default="Prob_")
    parser.add_argument("--routing-threshold", type=float, default=0.7)
    parser.add_argument("--disable-routing", action="store_true")
    parser.add_argument("--disable-refinement", action="store_true")
    parser.add_argument("--history-size", type=int, default=32)
    parser.add_argument(
        "--neighbor-reference",
        help="Training-only reference CSV used for nearest-neighbor evidence.",
    )
    parser.add_argument("--neighbor-label-column", default="Facies")
    parser.add_argument("--neighbor-well-column")
    parser.add_argument("--neighbor-k", type=int, default=3)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    main(
        arguments.input,
        arguments.output,
        window_size=arguments.window_size,
        step_size=arguments.step_size,
        prediction_column=arguments.prediction_column,
        confidence_column=arguments.confidence_column,
        probability_prefix=arguments.probability_prefix,
        routing_threshold=arguments.routing_threshold,
        routing_enabled=not arguments.disable_routing,
        refinement_enabled=not arguments.disable_refinement,
        history_size=arguments.history_size,
        neighbor_reference=arguments.neighbor_reference,
        neighbor_label_column=arguments.neighbor_label_column,
        neighbor_well_column=arguments.neighbor_well_column,
        neighbor_k=arguments.neighbor_k,
    )
