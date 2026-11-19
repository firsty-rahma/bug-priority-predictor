# Bug Severity Classifier

Automated bug severity classification using NLP and machine learning, trained on ~10,000 Mozilla Firefox bug reports from Bugzilla.

[![Tests](https://github.com/firsty-rahma/bug-priority-predictor/workflows/Tests/badge.svg)](https://github.com/firsty-rahma/bug-priority-predictor/actions)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## Result

**F1-Macro 0.34 on the held-out test set — 2.4x the majority-class baseline (0.14).**

| Metric            | Baseline (always "normal") | This model |
|-------------------|----------------------------|------------|
| Accuracy          | 76.0%                      | 65.3%      |
| F1-Macro          | 0.14                       | **0.34**   |
| Classes detected  | 1 of 6                     | **6 of 6** |

Selection uses repeated 5-fold cross-validation on the training split only. The test set is evaluated once, after the model is frozen. Across five independent 80/20 splits the selected model scores **0.350 +/- 0.012** macro F1, versus **0.310 +/- 0.013** for the previous Random Forest + SMOTE baseline.

The trade-off is intentional: the baseline gets higher accuracy by ignoring 24% of the data. This model sacrifices accuracy to give every severity class non-zero recall, which is what assisted triage actually needs.

**Recommended use:** human-in-the-loop triage assistant, not full automation. See [Limitations](#limitations).

---

## Problem

Bugzilla-style trackers receive thousands of reports daily. Manual severity assignment is slow, inconsistent across reviewers, and risks delayed response on critical bugs in high-volume environments.

The task: classify each bug into one of six severity levels (`blocker`, `critical`, `major`, `minor`, `normal`, `trivial`) from the report text.

The hard part: a 36:1 class imbalance between `normal` (76% of data) and `blocker` (2.1%). Naive models simply predict `normal` every time and get 76% accuracy while being useless.

---

## Approach

```
Raw bug reports (9,998)
        ↓
Near-raw text      ── short + long description; URLs/HTML stripped;
                      digits, identifiers, and negations kept
        ↓
Feature engineering ── TF-IDF words + bigrams (20k features, sublinear TF);
                      contractions preserved ("can't", "don't")
        ↓
Class imbalance     ── class_weight="balanced" in the classifier;
                      no SMOTE (it interpolates ordinal codes)
        ↓
Model               ── one-vs-rest logistic regression (C=1.0)
        ↓
Selection           ── repeated 5-fold CV on the training split;
                      test set touched once
        ↓
Evaluation          ── per-class precision/recall, error analysis,
                      5-split robustness check
```

The Random Forest + SMOTE pipeline is kept as a comparison baseline. It is not the primary model: on sparse 20k-feature text it underperforms a linear classifier, and applying SMOTE to ordinal-encoded categories created meaningless fractional values.

On this dataset, one-vs-rest beats a single multinomial softmax on macro F1 (0.350 vs 0.335 across five splits). Macro F1 weights all six classes equally, and OvR gives each class its own decision boundary instead of concentrating capacity on `normal` (76% of the data).

---

## Per-class performance

```
              precision    recall  f1-score   support

     blocker       0.27      0.17      0.21        41
    critical       0.49      0.52      0.51       121
       major       0.14      0.25      0.18       148
       minor       0.14      0.19      0.16       108
      normal       0.83      0.77      0.80      1521
     trivial       0.26      0.15      0.19        61

   macro avg       0.36      0.34      0.34      2000
weighted avg       0.69      0.65      0.67      2000
```

The model is most reliable on `normal` (F1 0.80) and `critical` (0.51). The other four classes are weak (F1 0.16–0.21): `blocker` has only 205 examples, and `minor` is partly label noise. See [results/error_analysis_report.txt](results/error_analysis_report.txt) for the breakdown.

---

## What I validated

**Candidate comparison across five random splits.** The selected OvR linear model scored 0.350 +/- 0.012 macro F1. A multinomial logistic regression scored 0.335 +/- 0.015, and the previous Random Forest + SMOTE baseline scored 0.310 +/- 0.013. The ordering is stable across all five seeds.

**Preprocessing ablation.** The old pipeline removed all NLTK stopwords (including `not`, `no`, `nor`), stripped digits and identifiers, and lemmatised every token. Switching to near-raw text raised the linear model from 0.306 to 0.330 in an earlier five-split check. That check changed several preprocessing choices at once, so the gain cannot be attributed to negations alone.

**Tuning is easy to overfit.** In an earlier single 5-fold run, `C=0.3` narrowly beat `C=1.0` (CV 0.358 vs 0.356), but five independent splits reversed the result: 0.342 versus 0.350. Repeated CV likewise preferred unigrams only by 0.003, but five independent splits gave 0.347 versus 0.350 for unigrams + bigrams. The tuning script therefore compares against the full default configuration and keeps it unless a tuned combination wins by more than 0.005.

**Error patterns by text length.** Error rate climbs from 22% on very short reports (<20 words) to 61% on very long ones (200+ words). TF-IDF has no notion of discourse structure, so long reports are harder.

---

## Transformer experiment

I also tested whether a fine-tuned transformer materially improves the six-class result. `distilbert-base-uncased` was trained on the same 80/20 split, with class-weighted cross-entropy, 256-token inputs, fp16, and early stopping on a stratified 10% validation split. The test set was evaluated once.

| Metric | Classical model | DistilBERT |
|---|---|---|
| Test macro F1 | **0.3402** | 0.3483 |
| Weighted F1 | 0.6707 | 0.6627 |
| Accuracy | 65.3% | 63.65% |
| Blocker recall | 0.1707 | 0.4146 |
| Critical recall | 0.5207 | 0.5124 |

The transformer gained only +0.0081 macro F1, short of the +0.03 adoption threshold, and critical recall slipped slightly. It is therefore not adopted: the released model remains the classical OvR logistic regression. Full metrics, error patterns, and reproduction steps are in [results/transformer_negative_result.md](results/transformer_negative_result.md); the plan is in [docs/transformer-plan.md](docs/transformer-plan.md).

---

## Limitations

I want to be direct about what this model is not:

- **Not production-ready for autonomous decisions.** 82% of `blocker` bugs are still missed. Any system using this model must keep humans in the loop for high-severity classifications.
- **Confidence scores are poorly calibrated.** Average confidence on correct predictions (0.42) is only modestly higher than on incorrect ones (0.33). Don't use confidence thresholds without calibration (Platt scaling or isotonic regression).
- **`minor` labels are noisy.** In the raw data, `severity_code` 2 covers both `minor` and `normal` (541 vs 7,604 rows), and code 3 is absent. The model cannot reliably separate those two classes; this is a label problem, not only a modelling problem.
- **Trained on Mozilla Firefox bugs only.** Generalization to other products, domains, or non-English bug reports is untested.
- **F1-Macro 0.34 is modest.** Reported results on similar tasks are often 0.6–0.8 with fine-tuned transformers and larger datasets; those numbers are not directly comparable. A DistilBERT experiment on this dataset reached 0.3483, so the classical model remains the released baseline. This project uses public data and classical ML to stay reproducible.

A realistic deployment would run this model in shadow mode against human triagers for several months before any automation, route low-confidence and high-severity predictions to humans, and retrain monthly on corrected labels.

---

## Quick start

Requires Python 3.11 or newer; CI covers 3.11 through 3.14.

```bash
git clone https://github.com/firsty-rahma/bug-priority-predictor.git
cd bug-priority-predictor

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, black, isort
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

pytest tests/ -v                  # verify install
```

Run the full pipeline:

```bash
python scripts/01_data_exploration.py
python scripts/02_text_preprocessing.py
python scripts/03_modelling.py
python scripts/04_modelling_hypertuning.py
python scripts/05_error_analysis.py
```

Total runtime: ~10–15 minutes on a 4-core CPU. Or run `make train`.

Interactive prediction: `python scripts/predict.py`.

---

## Project structure

```
bug-priority-predictor/
├── src/                  # Library code
│   ├── data/             # Loading and preprocessing
│   ├── evaluation/       # Metrics and plots
│   ├── models/           # Linear model, RF baseline, trainer
│   └── utils/            # Config and custom transformers
├── scripts/              # Numbered, runnable pipeline stages
├── tests/                # pytest suite
├── docs/                 # Dependency and license documentation
├── notebooks/            # Exploratory work
├── .github/workflows/    # CI: tests run on every push
└── requirements.txt
```

Design notes: library code is separated from execution scripts so the pipeline stages stay short and the logic stays testable. All hyperparameters live in `src/utils/config.py` so experiments don't require code changes. The main training seed is 42; the robustness check uses seeds 42, 0, 1, 2, and 3. Expect roughly ±1.5% macro-F1 variation between splits.

---

## Dataset

- **Source:** Gomes, Torres, & Côrtes (2021), *A Dataset for a Long-lived Prediction in FLOSS*, [Mendeley Data](https://data.mendeley.com/datasets/v446tfssgj/1)
- **Size:** 9,998 bug reports after cleaning
- **Distribution:** normal 76.0%, major 7.4%, critical 6.1%, minor 5.4%, trivial 3.1%, blocker 2.1%

---

## Further reading

- [results/error_analysis_report.txt](results/error_analysis_report.txt) — full error breakdown by class, text length, and confusion patterns
- [results/model_selection_summary.txt](results/model_selection_summary.txt) — candidate comparison and selection rationale
- [results/hyperparameter_tuning_summary.txt](results/hyperparameter_tuning_summary.txt) — tuning grid, chosen parameters, and final test report
- [results/transformer_negative_result.md](results/transformer_negative_result.md) — DistilBERT experiment, test metrics, and adoption decision
- [docs/transformer-plan.md](docs/transformer-plan.md) — transformer scope, training setup, and evaluation rule

---

## Author

**Firstyani Imannisa Rahma** — QA Engineer, Yogyakarta, Indonesia
[LinkedIn](https://www.linkedin.com/in/firstyani-rahma-412990236) · [Email](mailto:firsty.rahma9521@gmail.com)

## Acknowledgments

Dataset: Gomes, Torres, & Côrtes (2021). Tools: scikit-learn, NLTK, imbalanced-learn, pandas. Used Claude (Anthropic) as a technical mentor during development; all code was written and understood personally.

## License

MIT — see [LICENSE](LICENSE).