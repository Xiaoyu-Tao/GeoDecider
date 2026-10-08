import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from Facies.main import main
from Facies.ensemble import aggregate_runs
from Facies.process import (
    aggregate_panel,
    enforce_nm_m_consistency,
    route_window,
    process_logic,
    validate_labels,
)
from Facies.tools import NeighborFindTool


class RoutingTests(unittest.TestCase):
    def _metadata(self, confidence):
        frame = pd.DataFrame(
            {
                "NM_M": [1, 2],
                "Predicted_Facies": ["Nonmarine sandstone", "Wackestone"],
                "Prediction_Confidence": confidence,
            }
        )
        return {
            "window_df": frame,
            "config": {"routing_enabled": True, "routing_threshold": 0.7},
        }

    def test_low_score_activates_complete_interval(self):
        metadata = self._metadata([0.9, 0.4])
        self.assertTrue(route_window(metadata))
        self.assertEqual(metadata["routing"]["anchor_offsets"], [1])

    def test_confident_interval_uses_fast_path(self):
        metadata = self._metadata([0.9, 0.8])
        self.assertFalse(route_window(metadata))
        self.assertEqual(metadata["routing"]["reason"], "all_points_confident")

    def test_missing_scores_use_safe_slow_path(self):
        metadata = self._metadata([0.9, 0.8])
        metadata["window_df"] = metadata["window_df"].drop(
            columns=["Prediction_Confidence"]
        )
        self.assertTrue(route_window(metadata))
        self.assertIn("confidence_unavailable", metadata["routing"]["reason"])


class AggregationTests(unittest.TestCase):
    def test_majority_and_agreement(self):
        labels, agreement, overall = aggregate_panel(
            {
                "expert": ["Nonmarine sandstone", "Dolomite"],
                "model_aware": ["Nonmarine sandstone", "Wackestone"],
                "trend_focus": ["Mudstone", "Wackestone"],
            },
            2,
        )
        self.assertEqual(labels, ["Nonmarine sandstone", "Wackestone"])
        self.assertEqual(agreement, [2 / 3, 2 / 3])
        self.assertEqual(overall, 2 / 3)

    def test_invalid_length_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_labels(["Dolomite"], 2)


class EnvironmentConstraintTests(unittest.TestCase):
    def test_marine_nonmarine_conflicts_are_corrected(self):
        metadata = {
            "window_df": pd.DataFrame(
                {
                    "NM_M": [1, 2],
                    "Predicted_Facies": ["Nonmarine sandstone", "Wackestone"],
                }
            ),
            "config": {},
        }
        fixed, details = enforce_nm_m_consistency(
            metadata, ["Dolomite", "Nonmarine sandstone"]
        )
        self.assertEqual(fixed, ["Nonmarine sandstone", "Wackestone"])
        self.assertEqual(details["num_corrections"], 2)


class FastPathIntegrationTests(unittest.TestCase):
    def test_fast_path_writes_jsonl_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.jsonl"
            pd.DataFrame(
                {
                    "Depth": [1000.0, 1000.5],
                    "NM_M": [1, 2],
                    "Predicted_Facies": ["Nonmarine sandstone", "Wackestone"],
                    "Prediction_Confidence": [0.95, 0.91],
                }
            ).to_csv(input_path, index=False)

            options = {
                "feature_columns": ["Depth", "NM_M"],
                "window_size": 2,
                "routing_threshold": 0.7,
            }
            main(str(input_path), str(output_path), **options)
            main(str(input_path), str(output_path), **options)

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["meta_data"]["final_source"], "fast_classifier")
            self.assertEqual(record["meta_data"]["llm_usage_summary"]["llm_call_count"], 0)


class NeighborRetrievalTests(unittest.TestCase):
    def test_training_reference_retrieval_is_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.csv"
            pd.DataFrame(
                {
                    "WellName": ["train-a", "train-b"],
                    "Depth": [900.0, 950.0],
                    "GR": [10.0, 90.0],
                    "PHIND": [0.25, 0.05],
                    "Facies": ["Nonmarine sandstone", "Mudstone"],
                }
            ).to_csv(reference_path, index=False)
            metadata = {
                "window_df": pd.DataFrame(
                    {
                        "WellName": ["test-a"],
                        "Depth": [1000.0],
                        "GR": [12.0],
                        "PHIND": [0.24],
                    }
                ),
                "feature_columns": ["Depth", "GR", "PHIND"],
            }
            result = NeighborFindTool(
                str(reference_path), well_column="WellName", k=1
            ).run(metadata)
            neighbor = result["neighbor_retrieval"]["records"][0]["neighbors"][0]
            self.assertEqual(neighbor["well_id"], "train-a")
            self.assertEqual(neighbor["label"], "Nonmarine sandstone")

    def test_target_well_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.csv"
            pd.DataFrame(
                {
                    "WellName": ["same-well"],
                    "GR": [10.0],
                    "Facies": ["Nonmarine sandstone"],
                }
            ).to_csv(reference_path, index=False)
            metadata = {
                "window_df": pd.DataFrame(
                    {"WellName": ["same-well"], "GR": [11.0]}
                ),
                "feature_columns": ["GR"],
            }
            with self.assertRaises(ValueError):
                NeighborFindTool(
                    str(reference_path), well_column="WellName", k=1
                ).run(metadata)


class SlowPathIntegrationTests(unittest.TestCase):
    @patch("Facies.process.get_json_result")
    @patch("Facies.process.get_result")
    @patch("Facies.process.get_tool_selection")
    def test_complete_slow_path_records_all_stages(
        self, tool_selection, classification, structured
    ):
        tool_selection.return_value = (
            "planner prompt",
            "planner reasoning",
            '{"tools": [{"name": "expert_feature_description_tool"}]}',
            ["expert_feature_description_tool"],
            {"model": "test", "usage": {"total_tokens": 5}},
        )
        panel_answer = '{"answer": ["Nonmarine sandstone", "Wackestone"]}'
        classification.return_value = (
            "panel reasoning",
            panel_answer,
            {"model": "test", "usage": {"total_tokens": 7}},
        )
        structured.side_effect = [
            (
                "fusion reasoning",
                '{"answer": ["Nonmarine sandstone", "Wackestone"], '
                '"decisions": []}',
                {"model": "test", "usage": {"total_tokens": 11}},
            ),
            (
                "refinement reasoning",
                '{"answer": ["Nonmarine sandstone", "Wackestone"], '
                '"revisions": []}',
                {"model": "test", "usage": {"total_tokens": 13}},
            ),
        ]
        metadata = {
            "window_df": pd.DataFrame(
                {
                    "Depth": [1000.0, 1000.5],
                    "NM_M": [1, 2],
                    "Predicted_Facies": ["Nonmarine sandstone", "Wackestone"],
                    "Prediction_Confidence": [0.4, 0.5],
                }
            ),
            "window_up": pd.DataFrame(),
            "window_down": pd.DataFrame(),
            "feature_columns": ["Depth", "NM_M"],
            "history": [],
            "config": {
                "routing_enabled": True,
                "routing_threshold": 0.7,
                "refinement_enabled": True,
            },
        }
        _, _, answer, result = process_logic(metadata)
        self.assertEqual(
            json.loads(answer)["answer"],
            ["Nonmarine sandstone", "Wackestone"],
        )
        self.assertEqual(result["final_source"], "slow_stage")
        self.assertEqual(len(result["llm_calls"]), 6)


class CrossRunEnsembleTests(unittest.TestCase):
    def test_cross_run_vote_is_separate_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{"Depth": 1000.0}, {"Depth": 1000.5}]
            predictions = [
                ["Nonmarine sandstone", "Wackestone"],
                ["Nonmarine sandstone", "Dolomite"],
                ["Mudstone", "Dolomite"],
            ]
            input_paths = []
            for index, labels in enumerate(predictions):
                path = Path(directory) / f"run-{index}.jsonl"
                record = {
                    "meta_data": {
                        "current_window_id": 1,
                        "window_df": rows,
                        "panel_aggregation": {"final_labels": labels},
                    }
                }
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                input_paths.append(str(path))

            output_path = Path(directory) / "ensemble.jsonl"
            aggregate_runs(input_paths, str(output_path))
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                result["answer"], ["Nonmarine sandstone", "Dolomite"]
            )
            self.assertEqual(result["agreement_per_depth"], [2 / 3, 2 / 3])


if __name__ == "__main__":
    unittest.main()
