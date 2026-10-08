# GeoDecider: An Evidence-Guided Agent for Geological Interpretation

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Task](https://img.shields.io/badge/Task-Lithology%20Classification-2E8B57?style=flat-square)
![Agent](https://img.shields.io/badge/Agent-Evidence--Guided-8A2BE2?style=flat-square)

---

GeoDecider is an evidence-guided agent for geological interpretation from well logs. It combines efficient point-wise prediction models with tool-augmented reasoning, allowing confident samples to be classified directly while ambiguous intervals receive deeper geological analysis.

> "GeoDecider: An Evidence-Guided Agent for Geological Interpretation"
> Under review

---

## Overview

Lithology interpretation from well logs is an important task in subsurface characterization, supporting reservoir evaluation and geological modeling. However, automated lithology classification remains challenging because different rock types may exhibit similar logging responses, and reliable interpretation requires combining local observations with geological context, domain knowledge, and physical constraints.

GeoDecider introduces an adaptive coarse-to-fine workflow:

- Uses lightweight classifiers to generate point-wise lithology predictions and confidence scores.
- Routes uncertain intervals to an evidence-guided reasoning workflow.
- Builds an Evidence Profile using geological knowledge, depth-wise trends, historical interpretations, and similar cases.
- Performs multi-view reasoning over data patterns, geological context, and domain knowledge.
- Applies geological constraints to refine predictions and improve sequence consistency.

![GeoDecider Motivation](pic/0-Motivation.png)

![GeoDecider Framework](pic/1-Framework.png)

---

## Key Features

- Adaptive Inference Routing: Directly classifies confident cases and allocates reasoning resources to ambiguous intervals.
- Evidence Profile Construction: Collects complementary geological evidence from specialized tools.
- Tool-Augmented Geological Reasoning: Uses feature descriptions, label knowledge, heuristic classification suggestions, trend analysis, and similar-case retrieval.
- Multi-view Decision Panel: Integrates expert, model-aware, and trend-focused reasoning perspectives.
- Constraint-based Refinement: Enforces depositional environment consistency through geological constraints such as `NM_M`.
- Public Benchmark Evaluation: Experiments on four public well-log benchmarks show consistent improvements over representative baselines.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Xiaoyu-Tao/GeoDecider.git
cd GeoDecider
```

### 2. Environment Setup

```bash
conda create -n geodecider python=3.10
conda activate geodecider
pip install -r requirements.txt
```

### 3. Configure the LLM API

Credentials are read from environment variables and are never stored in source files:

```bash
export GEODECIDER_API_KEY="YOUR_API_KEY"
# Optional overrides:
export GEODECIDER_BASE_URL="https://api.deepseek.com"
export GEODECIDER_MODEL="deepseek-reasoner"
```

### 4. Prepare Input Data

GeoDecider expects well-log data in CSV format. The current workflow uses the following columns:

- `Depth`
- `GR`
- `ILD_log10`
- `DeltaPHI`
- `PHIND`
- `PE`
- `NM_M`
- `RELPOS`
- `Predicted_Facies`

For difficulty-aware routing, also provide either:

- `Prediction_Confidence`, containing the maximum classifier score; or
- one or more class-score columns whose names start with `Prob_`.

If neither is available, the interval is explicitly recorded as
`confidence_unavailable_safe_slow_path` and enters the slow stage. This avoids
silently treating an interval as confident without a score.

### 5. Run GeoDecider

```bash
python -m Facies.main \
  --input path/to/input.csv \
  --output path/to/output.jsonl \
  --routing-threshold 0.70
```

Run the command from the repository root. Use `python -m Facies.main --help` for
all options. Results are saved as JSONL records containing routing scores and
anchors, selected tools, panel outputs, within-run reconciliation, refinement
records, token accounting, and the final answer. Existing successful window IDs
are detected when a run resumes; resume no longer relies on the raw number of
lines in the output file.

To enable nearest-neighbor evidence, pass a labeled **training-only** reference
CSV:

```bash
python -m Facies.main \
  --input path/to/input.csv \
  --output path/to/output.jsonl \
  --neighbor-reference path/to/training_reference.csv \
  --neighbor-label-column Facies \
  --neighbor-well-column WellName
```

The reference path, fitted feature columns, distances, source rows, well IDs,
and returned labels are stored in the JSONL record for auditing. Validation and
test labels must not be included in this reference file.

### Aggregate Independent Runs

Evidence-aware reconciliation combines the three scientific views **within one
run**. Majority voting across complete stochastic runs is a separate operation:

```bash
python -m Facies.ensemble \
  --inputs outputs/run1.jsonl outputs/run2.jsonl outputs/run3.jsonl \
  --output outputs/ensemble.jsonl
```

The command verifies that all runs contain identical windows and rows, then
records per-depth agreement. It refuses to replace an existing output unless
`--overwrite` is supplied.

---

## Workflow

1. Routing: A below-threshold point activates its complete containing interval;
   an interval with no anchor retains the fast prediction.
2. Planner: Selects available geological tools for an activated interval.
3. Evidence Collection: Builds evidence from feature knowledge, label definitions,
   heuristics, depth trends, preceding finalized predictions, and optional
   training-only neighbor retrieval.
4. Multi-view Reasoning: Generates expert, model-aware, and trend-focused
   candidates. Every candidate must have the same length as the input and use
   only declared labels.
5. Evidence-aware Reconciliation: Uses the evidence profile to resolve the
   within-run candidates. Majority voting is retained as an explicit fallback;
   any departure from it requires a recorded support item.
6. Geological Refinement: Proposed revisions require recorded observed evidence,
   after which the Facies-specific `NM_M` environment constraint is applied.

## Reproducibility Scope

This repository contains the executable Facies pipeline. It does not currently
include the licensed/source datasets, trained base-classifier checkpoints, or
archived predictions needed to reproduce every multi-dataset number in the
paper. Adding a mechanism to the current code does not establish that an older
reported experiment used that mechanism. Reproduction packages should identify
the exact code commit, dataset version and well split, configuration, prompts,
raw predictions, API model version, and execution logs associated with each
reported result.

## Tests

The deterministic routing, validation, aggregation, and environment-constraint
tests do not call an external API:

```bash
python -m unittest discover -s tests -v
```

---

## Lithofacies Categories

GeoDecider predicts nine lithofacies classes:

- Nonmarine sandstone
- Nonmarine coarse siltstone
- Nonmarine fine siltstone
- Marine siltstone and shale
- Mudstone
- Wackestone
- Dolomite
- Packstone-grainstone
- Phylloid-algal bafflestone

---

## Benchmark Results

Experiments on four public well-log benchmarks demonstrate that GeoDecider consistently outperforms representative baselines, validating the effectiveness of evidence-guided reasoning for reliable geological interpretation.

Full benchmark tables and ablation results will be released with the paper.

---

## Citation

If you find this project useful, please consider citing our paper:

```bibtex
@inproceedings{geodecider2026,
  title={GeoDecider: An Evidence-Guided Agent for Geological Interpretation},
  author={Anonymous},
  booktitle={Under review},
  year={2026}
}
```

---

## Contact

For questions or collaborations, please open an issue or contact the project authors.

---

## License

This project is released for research use. A formal license will be added soon.
