# Transformer Fine-Tuning Plan

Status: planned, not implemented.

## Goal

Test whether a fine-tuned transformer materially improves six-class severity
classification over the current classical baseline. Keep the classical pipeline
on `main` until the transformer is validated.

## Baseline to beat

| Metric | Classical baseline |
|---|---|
| Test macro F1 | **0.3402** |
| Test weighted F1 | 0.6707 |
| Test accuracy | 65.3% |
| 2x5-fold CV macro F1 | 0.3478 |
| Five-split macro F1 | 0.3499 +/- 0.0118 |

Reference commit: `28e4805` (tag `v1.0-classical`).

Supporting diagnostics:

- Merging `minor` into `normal` raises test macro F1 to **0.3911**
  (`results/five_class_diagnostic.txt`). Label noise costs roughly 0.05.
- Component/product metadata, character n-grams, and `min_df=1` did not clear
  the 0.005 CV tolerance (`results/feature_experiments.csv`).

## Branch strategy

- Work on `feature/transformer-severity`.
- Keep the classical code and artifacts untouched on `main`.
- Do not move legacy files. Branches are snapshots; moving paths would break
  imports and the working baseline.
- Use an optional `requirements-transformer.txt` so the base install stays light.
- Only merge after the transformer clears the adoption rule below.

## Data and splits

- Dataset: `data/bugs_preprocessed.csv`, 9,998 rows.
- Input: `text_for_model` (short description + long description).
- Keep the existing 80/20 split (`random_state=42`, stratified) for a directly
  comparable test result.
- Carve a stratified 10% validation split from the training rows for early
  stopping. Do not use the test set for tuning.
- Initial `max_length=256`; try 384 only if memory allows. Long reports will be
  truncated.
- Keep six classes for the primary comparison. The five-class result remains a
  diagnostic, not the main task.

## Model

- Start with `distilbert-base-uncased` with a six-label classification head.
- Fallbacks if quality is insufficient: `roberta-base`, `microsoft/deberta-v3-small`.
- Handle imbalance with class-weighted cross-entropy. No SMOTE.
- Optional label smoothing of 0.05.
- Do not add component/product metadata initially; test text-only first.

## Training configuration

| Setting | Initial value |
|---|---|
| Batch size | 8 or 16 |
| Gradient accumulation | 1-2 |
| Learning rate | 2e-5 |
| Epochs | 3-5 |
| Warmup | 10% |
| Weight decay | 0.01 |
| Precision | fp16 if stable |
| Early stopping | validation macro F1, patience 1-2 |
| Seed | 42 |

Run 2-3 seeds only after a single run shows promise.

## Evaluation

Primary metric: **test macro F1**, evaluated once after model selection.

Also report:

- Per-class precision, recall, and F1.
- Blocker and critical recall, since missed high-severity bugs matter most.
- Confusion matrix and top misclassification patterns.
- Calibration check (confidence vs accuracy), because `predict.py` exposes
  probabilities.
- Optional five-class sensitivity result, clearly marked as not comparable.

## Compute

Local GPU: NVIDIA GeForce RTX 2050, 4 GB VRAM.

- DistilBERT at 256 tokens, batch 8-16, should fit with fp16.
- Estimated runtime: 30-60 minutes per run on this GPU.
- If OOM: reduce batch size, use gradient accumulation, freeze lower layers, or
  shorten `max_length`.
- CPU training is possible but not practical for repeated experiments.
- Environment currently has no `torch`, `transformers`, `datasets`, or
  `accelerate` installed.

## Dependencies

Install CUDA-compatible PyTorch first, then the Hugging Face stack:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate
```

Record the exact working versions in `requirements-transformer.txt` after a
successful install. Do not guess-pin versions before verifying Python 3.14 and
CUDA compatibility.

## Implementation checklist

1. Add `requirements-transformer.txt` with verified versions.
2. Add `src/models/transformer.py` with tokenization, label mapping, and model
   factory helpers.
3. Add `scripts/06_transformer_training.py` for train/validation, early
   stopping, and artifact saving.
4. Add `scripts/07_transformer_evaluation.py` for the single test evaluation,
   per-class report, and comparison against the classical baseline.
5. Add focused tests for tokenization and label mapping; include a tiny
   smoke-test path that does not require a full epoch.
6. Update `.gitignore` for transformer checkpoints and large artifacts.
7. If the transformer wins, update `predict.py`, README, and docs in a separate
   commit. Otherwise keep the branch as an experiment and leave `main` intact.

## Adoption rule

Merge only if all of the following hold:

- Test macro F1 is meaningfully above **0.3402** (target at least +0.03).
- Blocker and critical recall do not regress.
- The result is reproducible from documented commands and saved metadata.
- Inference remains usable on the available hardware.

If the transformer does not clear these conditions, keep the classical model as
the released baseline and document the negative result.

## Risks

- 4 GB VRAM may force smaller batches or shorter sequences.
- Label noise still caps achievable F1; the five-class result suggests the
  ceiling is well below 1.0.
- Long reports lose information at 256 tokens.
- Python 3.14 and CUDA wheel compatibility must be verified before installing.
- Fine-tuning on ~8k training rows can overfit; early stopping is mandatory.
- Repeated test evaluation would invalidate the comparison; evaluate once.
