# Bug Severity Classifier

Automated bug severity classification using NLP and machine learning, trained on ~10,000 Mozilla Firefox bug reports from Bugzilla.

[![Tests](https://github.com/firsty-rahma/bug-priority-predictor/workflows/Tests/badge.svg)](https://github.com/firsty-rahma/bug-priority-predictor/actions)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![Coverage](https://img.shields.io/badge/coverage-85%25-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## Result

**F1-Macro 0.29 on the test set — a 123% improvement over the majority-class baseline (0.13).**

| Metric            | Baseline (always "normal") | This model |
|-------------------|----------------------------|------------|
| Accuracy          | 76.0%                      | 66.0%      |
| F1-Macro          | 0.13                       | **0.29**   |
| Classes detected  | 1 of 6                     | **6 of 6** |

The trade-off is intentional: the baseline gets higher accuracy by ignoring 24% of the data. This model sacrifices accuracy to give every severity class a non-zero recall, which is what assisted triage actually needs.

**Recommended use:** human-in-the-loop triage assistant, not full automation. See [Limitations](#limitations).

---

## Problem

Bugzilla-style trackers receive thousands of reports daily. Manual severity assignment is slow, inconsistent across reviewers, and risks delayed response on critical bugs in high-volume environments.

The task: classify each bug into one of six severity levels (`blocker`, `critical`, `major`, `minor`, `normal`, `trivial`) from the report text, component, and product.

The hard part: a 37:1 class imbalance between `normal` (76% of data) and `blocker` (2.1%). Naive models simply predict `normal` every time and get 76% accuracy while being useless.

---

## Approach

```
Raw bug reports (9,998)
        ↓
Text preprocessing  ── selective stop-word removal (keeps technical negations: "not", "can't")
        ↓
Feature engineering ── TF-IDF (unigrams) + text length + component + product
        ↓
Class imbalance     ── SMOTE oversampling on minority classes
        ↓
Model               ── Random Forest (n=200, max_depth=20), tuned via GridSearchCV
        ↓
Evaluation          ── per-class precision/recall, error analysis, ablation study
```

Random Forest was chosen over Logistic Regression for better minority-class handling. XGBoost was excluded due to memory constraints during development.

---

## Per-class performance

```
              precision    recall  f1-score   support

     blocker       0.10      0.05      0.06        41
    critical       0.48      0.48      0.48       121
       major       0.16      0.22      0.19       148
       minor       0.10      0.06      0.07       108
      normal       0.81      0.79      0.80      1521
     trivial       0.11      0.21      0.14        61

   macro avg       0.29      0.30      0.29      2000
weighted avg       0.67      0.66      0.66      2000
```

The model is genuinely useful on `normal` (the bulk of traffic) and `critical` (the class that matters most for triage), and weak on `blocker` and `minor` — both extreme minority classes where SMOTE alone isn't enough. See [docs/error-analysis.md](docs/error-analysis.md) for the breakdown.

---

## What I validated

**Ablation study on the top feature.** "crash" was the strongest single feature at 7.2% importance. To check whether it carried real signal or was just correlated with severity, I retrained without it: F1-Macro dropped from 0.291 to 0.278, critical recall from 48% to 43%, blocker recall from 5% to 3%. The signal is real and not redundant with other features.

**Stopword strategy comparison.** Tested three approaches: remove all NLTK stop words, remove none, and selective removal (keep technical negations like "not", "can't", "doesn't"). Selective removal won at 0.34 CV F1-Macro vs 0.32 and 0.31 for the other two. Domain-specific preprocessing matters more than generic NLP defaults.

**Error patterns by text length.** Error rate climbs from 23% on very short descriptions (<20 chars) to 64% on very long ones (200+ chars). The TF-IDF + Random Forest combination doesn't capture context well in long text — a transformer-based approach (BERT) is the obvious next step.

---

## Limitations

I want to be direct about what this model is not:

- **Not production-ready for autonomous decisions.** 95% of `blocker` bugs are missed. Any system using this model must keep humans in the loop for high-severity classifications.
- **Confidence scores are poorly calibrated.** Average confidence on correct predictions (34.7%) is barely higher than on incorrect ones (31.5%). Don't use confidence thresholds without calibration (Platt scaling or isotonic regression).
- **Trained on Mozilla Firefox bugs only.** Generalization to other products, domains, or non-English bug reports is untested.
- **F1-Macro 0.29 is modest.** Published research on similar tasks reports 0.6–0.8 using fine-tuned BERT and larger proprietary datasets. This project uses public data and classical ML to keep it reproducible.

A realistic deployment would run this model in shadow mode against human triagers for several months before any automation, route low-confidence and high-severity predictions to humans, and retrain monthly on corrected labels.

---

## Quick start

```bash
git clone https://github.com/firsty-rahma/bug-priority-predictor.git
cd bug-priority-predictor

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

pytest tests/ -v                  # verify install (85% coverage)
```

Run the full pipeline:

```bash
python scripts/01_data_exploration.py
python scripts/02_text_preprocessing.py
python scripts/03_modeling.py
python scripts/04_hyperparameter_tuning.py
python scripts/05_error_analysis.py
```

Total runtime: ~23 minutes on a 4-core CPU. Or run `make train`.

Interactive prediction: `python scripts/predict.py`.

---

## Project structure

```
bug-priority-predictor/
├── src/                  # Library code
│   ├── data/             # Loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # Training, evaluation, inference
│   └── config.py         # Centralized hyperparameters
├── scripts/              # Numbered, runnable pipeline stages
├── tests/                # pytest suite, 85% coverage
├── docs/                 # Detailed analysis writeups
├── notebooks/            # Exploratory work
├── .github/workflows/    # CI: tests run on every push
└── requirements.txt
```

Design notes: library code is separated from execution scripts so the pipeline stages stay short and the logic stays testable. All hyperparameters live in `src/config.py` so experiments don't require code changes. Random seeds are fixed at 42 throughout; expect ±2% variation between runs from SMOTE and parallel processing.

---

## Dataset

- **Source:** Gomes, Torres, & Côrtes (2021), *A Dataset for a Long-lived Prediction in FLOSS*, [Mendeley Data](https://data.mendeley.com/datasets/v446tfssgj/1)
- **Size:** 9,998 bug reports after cleaning
- **Distribution:** normal 76.0%, major 7.4%, critical 6.1%, minor 5.4%, trivial 3.1%, blocker 2.1%

---

## Further reading

- [docs/error-analysis.md](docs/error-analysis.md) — full error breakdown by class, text length, and confusion patterns
- [docs/experiments.md](docs/experiments.md) — stopword strategy, ablation study, hyperparameter tuning logs
- [docs/deployment-notes.md](docs/deployment-notes.md) — phased deployment strategy and monitoring requirements

---

## Author

**Firstyani Imannisa Rahma** — QA Engineer, Yogyakarta, Indonesia
[LinkedIn](https://www.linkedin.com/in/firstyani-rahma-412990236) · [Email](mailto:firsty.rahma9521@gmail.com)

## Acknowledgments

Dataset: Gomes, Torres, & Côrtes (2021). Tools: scikit-learn, NLTK, imbalanced-learn, pandas. Used Claude (Anthropic) as a technical mentor during development; all code was written and understood personally.

## License

MIT — see [LICENSE](LICENSE).