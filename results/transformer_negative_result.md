# Transformer Negative Result

Status: closed experiment. The fine-tuned transformer is **not adopted**. The
classical OvR logistic-regression model remains the released baseline on `main`
(tag `v1.0-classical`, commit `28e4805`).

## Setup

- Model: `distilbert-base-uncased`, six-label classification head.
- Data: `data/bugs_preprocessed.csv`, `text_for_model`, same stratified 80/20
  split as the classical model (`random_state=42`); 7,198 train / 800
  validation / 2,000 test rows.
- Training: class-weighted cross-entropy, `max_length=256`, batch 8, gradient
  accumulation 2, fp16, lr `2e-5`, 5 epochs, warmup 10%, weight decay 0.01,
  early stopping on validation macro F1 (patience 2), seed 42.
- Hardware: NVIDIA GeForce RTX 2050, 4.29 GB VRAM.
- Runtime: 10.6 minutes.

The test split was evaluated exactly once, after training and validation were
complete.

## Validation result

| Metric | Value |
|---|---|
| Best validation macro F1 | 0.3586 (epoch 5) |
| Validation weighted F1 | 0.6805 |
| Validation accuracy | 66.25% |
| Validation blocker recall | 0.2500 |
| Validation critical recall | 0.5510 |

Validation never exceeded the 0.3702 test target, so the run proceeded to the
single test evaluation only to complete the pre-registered check.

## Test result

| Metric | Classical baseline | Transformer | Delta |
|---|---|---|---|
| Macro F1 | **0.3402** | 0.3483 | +0.0081 |
| Weighted F1 | 0.6707 | 0.6627 | -0.0080 |
| Accuracy | 65.30% | 63.65% | -1.65 pts |

Per-class F1:

| Class | Baseline | Transformer | Delta |
|---|---|---|---|
| blocker | 0.2090 | 0.2720 | +0.0630 |
| critical | 0.5060 | 0.4980 | -0.0080 |
| major | 0.1809 | 0.1765 | -0.0045 |
| minor | 0.1581 | 0.1206 | -0.0375 |
| normal | 0.7997 | 0.7894 | -0.0103 |
| trivial | 0.1875 | 0.2336 | +0.0461 |

High-severity recall:

| Class | Baseline | Transformer | Delta |
|---|---|---|---|
| blocker | 0.1707 | 0.4146 | +0.2439 |
| critical | 0.5207 | 0.5124 | -0.0083 |

## Adoption check

Required: test macro F1 >= **0.3702** (+0.03 over 0.3402) with no blocker or
critical recall regression.

| Condition | Result |
|---|---|
| Macro F1 >= 0.3702 | FAIL (0.3483) |
| Blocker recall not reduced | PASS |
| Critical recall not reduced | FAIL |
| **Adopt transformer** | **NO** |

## What the run showed

- The macro-F1 gain was +0.0081, roughly a quarter of the required +0.03 and
  within the noise seen across classical CV splits (+/- 0.0118).
- The transformer traded accuracy and weighted F1 for blocker recall. Blocker
  recall improved substantially (0.1707 -> 0.4146), but critical recall slipped
  slightly and overall calibration was weaker.
- Misclassifications concentrate around `normal`, `minor`, and `major`:
  - normal -> minor: 145
  - normal -> major: 102
  - major -> normal: 79
  - minor -> normal: 69
  - normal -> trivial: 51
- Confidence separates correct from incorrect predictions only weakly:
  mean confidence 0.625 when correct versus 0.495 when incorrect.
- These patterns match the classical model's failure mode, which points at
  label ambiguity and overlapping definitions rather than a representation
  limitation that a larger encoder alone would fix.

## Artifacts

- `results/transformer_evaluation_report.txt`: full test report and adoption check.
- `results/transformer_evaluation.json`: machine-readable metrics.
- `results/transformer_training_summary.txt`: validation metrics and run metadata.
- `results/transformer_training_metadata.json`: copy of the checkpoint metadata
  with `test_set_evaluated: true`.
- `results/transformer_confusion_matrix.csv`, `figures/transformer_confusion_matrix.png`.
- `results/transformer_test_predictions.csv`: all 2,000 test predictions and probabilities.
- `results/transformer_misclassified_cases.csv`: 727 test errors.
- `models/transformer-severity/`: ignored local checkpoint, tokenizer, split
  indices, and metadata.

## Reproduction

```bash
pip install -r requirements-transformer.txt
python scripts/06_transformer_training.py
python scripts/07_transformer_evaluation.py
```

The evaluation script refuses to overwrite results unless `--force` is passed.
Do not rerun the test evaluation for this checkpoint; it has already been used
once.

## Decision

Keep the classical model as the released baseline. `main` is unchanged. This
branch preserves the implementation and evidence for anyone who wants to retry
with a different encoder, a different pooling strategy, or the five-class
diagnostic. A retry must use a fresh validation-driven configuration and must
not tune on this test split.
