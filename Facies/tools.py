"""Evidence tools used by the Facies agent."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .api import get_result_trend
    from .constants import DEFAULT_FEATURE_COLUMNS
    from .prompts import (
        CLASSIFICATION_SUGGESTIONS,
        FEATURE_DESCRIPTIONS,
        LABEL_DESCRIPTIONS,
        build_trend_prompt,
    )
except ImportError:  # Support ``python Facies/main.py``.
    from api import get_result_trend
    from constants import DEFAULT_FEATURE_COLUMNS
    from prompts import (
        CLASSIFICATION_SUGGESTIONS,
        FEATURE_DESCRIPTIONS,
        LABEL_DESCRIPTIONS,
        build_trend_prompt,
    )


class ExpertFeatureDescriptionTool:
    name = "expert_feature_description_tool"

    def run(self) -> str:
        rows = ["Here are the descriptions of the available well-log features:"]
        rows.extend(f"**{key}**: {value}" for key, value in FEATURE_DESCRIPTIONS.items())
        return "\n".join(rows)


class ExpertLabelDescriptionTool:
    name = "expert_label_description_tool"

    def run(self) -> str:
        rows = ["Here are the descriptions of the lithofacies labels:"]
        rows.extend(f"**{key}**: {value}" for key, value in LABEL_DESCRIPTIONS.items())
        return "\n".join(rows)


class ClassificationSuggestionsTool:
    name = "classification_suggestions_tool"

    def run(self) -> str:
        rows = ["Here are heuristic suggestions for facies classification:"]
        for label, suggestion in CLASSIFICATION_SUGGESTIONS.items():
            rows.append(f"### {label}\n{suggestion}")
        rows.append(
            "These are qualitative tendencies, not universal thresholds. Interpret "
            "multiple curves jointly and preserve supported facies boundaries."
        )
        return "\n\n".join(rows)


class TrendAnalysisTool:
    name = "trend_analysis_tool"

    def run(self, meta_data: Dict[str, Any]) -> Dict[str, Any]:
        full_window = pd.concat(
            [
                meta_data.get("window_up", pd.DataFrame()),
                meta_data["window_df"],
                meta_data.get("window_down", pd.DataFrame()),
            ]
        )
        cols = [
            column
            for column in meta_data["feature_columns"]
            if column in full_window.columns
        ]
        prompt = build_trend_prompt(full_window[cols])
        think, answer, usage = get_result_trend(prompt)
        meta_data["trend_analysis_prompt"] = prompt
        meta_data["trend_analysis_think"] = think
        meta_data["trend_analysis_answer"] = answer
        meta_data.setdefault("llm_calls", []).append(
            {"stage": "trend_analysis", **usage}
        )
        return meta_data


class NeighborFindTool:
    """Retrieve nearest rows from an explicitly supplied training-only CSV.

    Standardization statistics are fitted on the reference CSV. The tool never
    reads labels from the target interval and records the source row for audit.
    """

    name = "neighbor_finding_tool"

    def __init__(
        self,
        reference_csv: str,
        *,
        label_column: str = "Facies",
        well_column: Optional[str] = None,
        k: int = 3,
    ):
        self.reference_csv = str(Path(reference_csv).expanduser().resolve())
        self.label_column = label_column
        self.well_column = well_column
        self.k = k

    def run(self, meta_data: Dict[str, Any]) -> Dict[str, Any]:
        reference = pd.read_csv(self.reference_csv)
        target = meta_data["window_df"]
        feature_columns = [
            column
            for column in meta_data.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
            if column != "Depth"
            and column in reference.columns
            and column in target.columns
            and pd.api.types.is_numeric_dtype(reference[column])
        ]
        if not feature_columns:
            raise ValueError("Neighbor reference has no compatible numeric features.")
        if self.label_column not in reference.columns:
            raise ValueError(
                f"Neighbor reference is missing label column {self.label_column!r}."
            )
        if (
            self.well_column
            and self.well_column in reference.columns
            and self.well_column in target.columns
        ):
            overlap = set(reference[self.well_column].dropna().astype(str)) & set(
                target[self.well_column].dropna().astype(str)
            )
            if overlap:
                raise ValueError(
                    "Neighbor reference overlaps target well IDs: "
                    + ", ".join(sorted(overlap))
                )

        reference_values = reference[feature_columns].astype(float)
        target_values = target[feature_columns].astype(float)
        mean = reference_values.mean(axis=0)
        scale = reference_values.std(axis=0, ddof=0).replace(0, 1.0)
        ref_z = ((reference_values - mean) / scale).to_numpy()
        target_z = ((target_values - mean) / scale).to_numpy()

        records: List[Dict[str, Any]] = []
        limit = min(self.k, len(reference))
        for target_offset, vector in enumerate(target_z):
            distances = np.sqrt(np.square(ref_z - vector).sum(axis=1))
            nearest = np.argsort(distances)[:limit]
            neighbors = []
            for reference_position in nearest:
                row = reference.iloc[int(reference_position)]
                item: Dict[str, Any] = {
                    "reference_row": int(reference_position),
                    "label": str(row[self.label_column]),
                    "distance": float(distances[reference_position]),
                }
                if "Depth" in reference.columns:
                    item["depth"] = float(row["Depth"])
                if self.well_column and self.well_column in reference.columns:
                    item["well_id"] = str(row[self.well_column])
                neighbors.append(item)
            records.append(
                {
                    "target_offset": target_offset,
                    "target_depth": (
                        float(target.iloc[target_offset]["Depth"])
                        if "Depth" in target.columns
                        else None
                    ),
                    "neighbors": neighbors,
                }
            )

        meta_data["neighbor_retrieval"] = {
            "reference_csv": self.reference_csv,
            "label_column": self.label_column,
            "well_column": self.well_column,
            "feature_columns": feature_columns,
            "k": self.k,
            "records": records,
        }
        meta_data["neighbor_evidence"] = (
            "Training-reference nearest neighbors:\n" + str(records)
        )
        return meta_data
