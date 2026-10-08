"""Aggregate complete GeoDecider runs without confusing them with panel fusion."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .process import validate_labels
except ImportError:  # Support ``python Facies/ensemble.py``.
    from process import validate_labels


def _read_run(path: str) -> Dict[int, Dict[str, Any]]:
    records = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                metadata = record["meta_data"]
                window_id = int(metadata["current_window_id"])
                labels = metadata["panel_aggregation"]["final_labels"]
                rows = metadata["window_df"]
                validate_labels(labels, len(rows))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid run {path!r}, line {line_number}: {exc}") from exc
            if window_id in records:
                raise ValueError(f"Duplicate window {window_id} in {path!r}.")
            records[window_id] = {"labels": labels, "rows": rows}
    return records


def aggregate_runs(input_paths: List[str], output_path: str, *, overwrite: bool = False) -> None:
    if len(input_paths) < 2:
        raise ValueError("At least two complete runs are required for cross-run voting.")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Pass --overwrite to replace it."
        )

    runs = [_read_run(path) for path in input_paths]
    window_ids = set(runs[0])
    for path, run in zip(input_paths[1:], runs[1:]):
        if set(run) != window_ids:
            raise ValueError(f"Run {path!r} does not contain the same window IDs.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for window_id in sorted(window_ids):
            rows = runs[0][window_id]["rows"]
            label_lists = []
            for path, run in zip(input_paths, runs):
                if run[window_id]["rows"] != rows:
                    raise ValueError(
                        f"Run {path!r}, window {window_id} has different input rows."
                    )
                label_lists.append(run[window_id]["labels"])

            voted = []
            agreement = []
            for offset in range(len(rows)):
                votes = [labels[offset] for labels in label_lists]
                counts = {label: votes.count(label) for label in set(votes)}
                best_count = max(counts.values())
                candidates = {label for label, count in counts.items() if count == best_count}
                selected = votes[0] if votes[0] in candidates else sorted(candidates)[0]
                voted.append(selected)
                agreement.append(best_count / len(votes))

            record = {
                "window_id": window_id,
                "source_runs": [str(Path(path).expanduser().resolve()) for path in input_paths],
                "rows": rows,
                "answer": voted,
                "agreement_per_depth": agreement,
                "global_agreement": sum(agreement) / len(agreement),
            }
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Majority-vote predictions from complete GeoDecider runs."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    aggregate_runs(arguments.inputs, arguments.output, overwrite=arguments.overwrite)
