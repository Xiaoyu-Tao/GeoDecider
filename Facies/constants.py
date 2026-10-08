"""Shared labels, feature names, and tool identifiers for the Facies pipeline."""

FACIES_LABELS = (
    "Nonmarine sandstone",
    "Nonmarine coarse siltstone",
    "Nonmarine fine siltstone",
    "Marine siltstone and shale",
    "Mudstone",
    "Wackestone",
    "Dolomite",
    "Packstone-grainstone",
    "Phylloid-algal bafflestone",
)

NONMARINE_LABELS = frozenset(FACIES_LABELS[:3])
MARINE_LABELS = frozenset(FACIES_LABELS[3:])

DEFAULT_FEATURE_COLUMNS = (
    "Depth",
    "GR",
    "ILD_log10",
    "DeltaPHI",
    "PHIND",
    "PE",
    "NM_M",
    "RELPOS",
)

PREDICTION_COLUMN = "Predicted_Facies"
CONFIDENCE_COLUMN = "Prediction_Confidence"

TOOL_EXPERT_FEATURES = "expert_feature_description_tool"
TOOL_EXPERT_LABELS = "expert_label_description_tool"
TOOL_CLASSIFICATION = "classification_suggestions_tool"
TOOL_TREND = "trend_analysis_tool"
TOOL_NEIGHBORS = "neighbor_finding_tool"

TOOL_NAMES = (
    TOOL_EXPERT_FEATURES,
    TOOL_EXPERT_LABELS,
    TOOL_CLASSIFICATION,
    TOOL_TREND,
    TOOL_NEIGHBORS,
)
