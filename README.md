# README

# Bug Priority Predictor using NLP and Machine Learning

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

![NLTK](https://img.shields.io/badge/NLTK-3.8+-green.svg)

![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

![Coverage](https://img.shields.io/badge/coverage-85%25-green.svg)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

> ML-powered system for automatically classifying bug severity using NLP and Random Forest
> 

**Key Achievement:** 109% improvement in F1-Macro score through systematic feature engineering and SMOTE oversampling

[Quick Start](about:blank#-quick-start) • [Demo](about:blank#-demo) • [Results](about:blank#-key-results) • [Documentation](about:blank#-table-of-contents)

---

## 📋 Table of Contents

- [Problem Statement](about:blank#-problem-statement)
- [Quick Start](about:blank#-quick-start)
- [Demo](about:blank#-demo)
- [Key Results](about:blank#-key-results)
- [Dataset](about:blank#-dataset)
- [Approach](about:blank#-approach)
- [Performance Evolution](about:blank#-performance-evolution)
- [Stopword Strategy](about:blank#-stopword-strategy-experimental-analysis)
- [Feature Importance](about:blank#-feature-importance)
- [Error Analysis](about:blank#-error-analysis)
- [Ablation Study](about:blank#-ablation-study-validating-crash-importance)
- [Production Recommendations](about:blank#-production-recommendations)
- [Installation](about:blank#-installation)
- [Usage](about:blank#-usage)
- [Project Structure](about:blank#-project-structure)
- [Key Learnings](about:blank#-key-learnings)
- [Future Improvements](about:blank#-future-improvements)
- [FAQ](about:blank#-faq-how-does-this-differ-from-commercial-tools)
- [Contributing](about:blank#-contributing)
- [Author](about:blank#-author)
- [Acknowledgments](about:blank#-acknowledgments)

---

## 🎯 Problem Statement

Bug tracking systems like Bugzilla receive thousands of bug reports daily. Manual severity classification is:

- **⏱️ Time-consuming:** Human triage takes hours per day
- **❌ Inconsistent:** Different reviewers apply different standards
- **🚨 Delays critical fixes:** Severe bugs may be missed in high-volume queues

**Goal:** Build an ML system to automatically classify bug severity into 6 categories: `blocker` | `critical` | `major` | `minor` | `normal` | `trivial`

**Business Impact:**
- Reduce manual triage time by 70%
- Ensure critical bugs are escalated within 1 hour
- Provide consistent severity assignments across teams

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/firsty-rahma/bug-priority-predictor.git
cd bug-priority-predictor

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"
```

### Verify Installation

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Or check specific functionality
python -c "from src.data.preprocessor import TextPreprocessor; print('✅ Installation successful!')"
```

Expected output:

```
✅ Installation successful!
```

### Running the Pipeline

```bash
# Complete pipeline
python scripts/01_data_exploration.py
python scripts/02_text_preprocessing.py
python scripts/03_modeling.py
python scripts/04_hyperparameter_tuning.py
python scripts/05_error_analysis.py

# Or use make commands (if Makefile is present)
make train
```

For detailed installation instructions, see [Installation](about:blank#-installation) section below.

---

## 🎬 Demo

### Predicting Bug Severity

```bash
$ python scripts/predict.py

Interactive Prediction Mode
======================================================================
Short description: Firefox crashes on startup
Long description: Browser immediately closes when launched, losing all tabs
Component (default: General): General
Product (default: FIREFOX): FIREFOX

Making prediction...

======================================================================
PREDICTION RESULTS
======================================================================
Predicted Severity: CRITICAL
Confidence: 78.5%

All Class Probabilities:
  critical    : 78.50% ████████████████████████████████████████
  blocker     : 12.30% ██████
  major       :  5.20% ███
  normal      :  3.00% ██
  minor       :  0.80%
  trivial     :  0.20%

⚠️  High severity - recommend human review
```

### Performance Visualization

![Confusion Matrix](https://www.notion.sofigures/confusion_matrix_tuned_final.png)

*Confusion matrix showing strong performance on normal class (81% accuracy)*

![Feature Importance](https://www.notion.sofigures/feature_importance_detailed.png)

*Top features: “crash” accounts for 7% of model decisions*

![Error Analysis](https://www.notion.sofigures/error_analysis_overview.png)

*Error rate increases with text length - suggests need for context preservation*

---

## 🏆 Key Results

### Overall Performance

| Metric | Baseline | Final Model | Improvement |
| --- | --- | --- | --- |
| **F1-Macro (CV)** | 0.16 | **0.335** | **+109%** |
| **F1-Macro (Test)** | 0.16 | **0.32** | **+100%** |
| **Critical Bug Recall** | 6% | **49%** | **+717%** |
| **Normal Bug Accuracy** | 76% | **81%** | **+6%** |

### Per-Class Performance (Test Set)

```
                precision    recall  f1-score   support
     blocker       0.22      0.05      0.08        41
    critical       0.48      0.49      0.49       121
       major       0.19      0.24      0.21       148
       minor       0.17      0.09      0.12       108
      normal       0.82      0.81      0.81      1521
     trivial       0.14      0.28      0.19        61

    accuracy                           0.68      2000
   macro avg       0.34      0.33      0.32      2000
weighted avg       0.68      0.68      0.68      2000
```

**Key Achievements:**
- ✅ **All classes detected** (baseline had 0% recall for some classes)
- ✅ **Critical bug recall: 49%** (from 6% baseline)
- ✅ **Normal class: 81% accuracy** (main use case)
- ⚠️ **Trade-off accepted:** Lower overall accuracy for better minority class detection

---

## 📊 Dataset

- **Source:** Mozilla Bug Report Data (Mendeley Data Repository)
- **Paper:** Gomes, Luiz; Torres, Ricardo; Côrtes, Mario (2021), “A Dataset for a Long-lived Prediction in FLOSS”
- **Link:** [Mendeley Data](https://data.mendeley.com/datasets/v446tfssgj/1)
- **Size:** 9,998 bug reports after cleaning
- **Features:** Text descriptions, component, product, metadata
- **Target:** Severity category (6 classes)

### Severe Class Imbalance

| Severity | Count | Percentage | Challenge |
| --- | --- | --- | --- |
| **normal** | 7,604 | 76.0% | Overwhelming majority |
| **major** | 740 | 7.4% | Moderate minority |
| **critical** | 607 | 6.1% | Important minority |
| **minor** | 541 | 5.4% | Difficult to detect |
| **trivial** | 301 | 3.1% | Very rare |
| **blocker** | 205 | 2.1% | **Extremely rare but critical!** |

**Challenge:** Only **205 blocker samples** (2.1%) in the entire dataset, yet these are the most important bugs to detect!

---

## 🔬 Approach

### 1. Data Preprocessing

**Text Cleaning Pipeline:**

```python
"Firefox CRASHES on startup!!!"
→ lowercase → "firefox crashes on startup"
→ remove special chars → "firefox crashes on startup"
→ remove stopwords → "crash startup"
→ POS-aware lemmatize → "crash startup"
```

**Custom Stopword Strategy:**
- Removed 237 domain-specific stopwords
- Preserved severity indicators: `crash`, `hang`, `error`, `freeze`
- See [Stopword Strategy](about:blank#-stopword-strategy-experimental-analysis) for detailed analysis

### 2. Handling Severe Class Imbalance

**Problem:** 76% of bugs are “normal”, only 2.1% are “blocker”

**Solution: SMOTE (Synthetic Minority Over-sampling Technique)**
- Applied **inside** cross-validation folds to prevent data leakage
- Used `imblearn.pipeline.Pipeline` not `sklearn.pipeline.Pipeline`
- Stratified K-Fold to maintain class distribution

**Impact:** F1-Macro improved from **0.16 → 0.30** (+87.5%)

### 3. Feature Extraction

**TF-IDF Vectorization:**
- Top 1,000 features
- Unigrams only `(1, 1)` based on hyperparameter tuning
- Min document frequency: 2
- Max document frequency: 0.8

**Combined Feature Space:**
- Text features (TF-IDF): **94.7%** of feature importance
- Categorical features: **5.3%** of feature importance
- `component_name`: 2.3%
- `product_name`: 2.0%
- `text_length`: 1.0%

### 4. Model Selection & Hyperparameter Tuning

**Best Hyperparameters (Grid Search, 5-fold CV):**

```python
{
    'classifier__n_estimators': 200,
    'classifier__max_depth': 20,
    'classifier__min_samples_split': 10,
    'feature_combiner__ngram_range': (1, 1),
}
```

**Models Compared:**
- ✅ **Random Forest:** F1-Macro = 0.335 (winner!)
- ❌ Logistic Regression: F1-Macro = 0.105
- ❌ XGBoost: Memory constraints with SMOTE

---

## 📈 Performance Evolution

### Stage-by-Stage Improvement

| Stage | Configuration | F1-Macro (CV) | Δ from Previous | Cumulative Δ |
| --- | --- | --- | --- | --- |
| **Stage 1** | Text only, class weights | 0.16 | baseline | - |
| **Stage 2** | + SMOTE + Categorical features | 0.30 | **+87.5%** | +87.5% |
| **Stage 3** | + Hyperparameter tuning | 0.3318 | +10.6% | +107% |
| **Stage 4** | + Extended custom stopwords | **0.335** | **+0.96%** | **+109%** |

*Note: CV = Cross-Validation (5-fold stratified)*

### Key Improvements by Stage

### **Stage 1 → 2: Addressing Class Imbalance (+87.5%)**

- **Added:** SMOTE oversampling
- **Added:** Categorical features (component, product, text_length)
- **Impact:** Enabled model to learn minority classes
- **Result:** F1-Macro jumped from 0.16 → 0.30

### **Stage 2 → 3: Optimization (+10.6%)**

- **Hyperparameter tuning:** Grid search with 5-fold stratified CV
- **Best parameters found:**
    - ngram_range: (1, 1) - unigrams only (bigrams didn’t help!)
    - n_estimators: 200
    - max_depth: 20
    - min_samples_split: 10
- **Result:** F1-Macro improved from 0.30 → 0.3318

### **Stage 3 → 4: Domain-Specific Feature Engineering (+0.96%)**

- **Extended custom stopwords:** 237 domain-specific words removed
- **Trade-off:** CV improved (+0.96%) but test slightly decreased (-3%)
- **Decision:** Accepted for cleaner features and better typical-case performance
- **Result:** F1-Macro improved from 0.3318 → 0.335

---

## 🔍 Stopword Strategy: Experimental Analysis

One of the key challenges in NLP feature engineering is determining which words to remove as “stopwords.” This project demonstrates systematic experimentation to find the optimal strategy.

### Motivation

Bug reports contain domain-specific language:
- **Platform terms:** “firefox”, “mozilla”, “gecko” (appear in all bugs)
- **Procedural language:** “reproduce”, “testcase”, “step” (reporting process)
- **Generic actions:** “use”, “open”, “click” (all severity levels)

**Question:** Which words should we remove to maximize signal while minimizing noise?

### Experimental Design

Three strategies tested with 5-fold CV + holdout test:

### **Strategy 1: Baseline (Standard English Stopwords)**

Used NLTK’s built-in 179 English stopwords only.

**Result:** F1-Macro = 0.16 (poor performance due to noise)

### **Strategy 2: Minimal Custom Stopwords (Conservative)**

Removed only 15 procedural words:

```python
minimal_stopwords = {
    'see', 'produce', 'new', 'step',
    'reproduce', 'testcase', 'expect', 'report'
}
```

**Result:** F1-Macro (CV) = 0.3318, F1-Macro (Test) = 0.33

### **Strategy 3: Extended Custom Stopwords (Aggressive) ⭐**

Removed 237 domain-specific words across 6 categories:

1. **Platform/Tool:** `firefox`, `mozilla`, `gecko`, `bugzilla`, `windows`
2. **Procedural:** `reproduce`, `testcase`, `expect`, `report`, `step`
3. **Generic Actions:** `use`, `get`, `try`, `open`, `close`, `click`
4. **Generic Nouns:** `file`, `line`, `page`, `result`
5. **Temporal:** `new`, `current`, `latest`, `recent`
6. **Single Letters:** `a`, `b`, `c`, …, `z`
- **📋 View Full Extended Stopword List (Click to Expand)**
    
    ```python
    import string
    
    extended_stopwords = {
        # Platform/tool specific
        'always', 'firefox', 'mozilla', 'gecko', 'bugzilla',
        'os', 'com', 'nt', 'http', 'window', 'windows', 'enus',
    
        # Procedural language
        'reproduce', 'reproduced', 'reproducing', 'reproduces',
        'reproducible', 'reproducibly',
        'see', 'saw', 'sees', 'seeing', 'seen',
        'step', 'steps', 'stepped', 'stepping',
        'report', 'reported', 'reports', 'reporting', 'reporter',
        'testcases', 'testcase',
        'expect', 'expected', 'expects',
    
        # Generic actions
        'use', 'uses', 'used', 'using',
        'get', 'gets', 'got', 'gotten', 'getting',
        'try', 'tries', 'tried', 'trying',
        'open', 'opens', 'opened', 'opening',
        'close', 'closes', 'closed', 'closing',
        'click', 'clicks', 'clicked', 'clicking',
        'produce', 'produces', 'produced', 'producing',
        'build', 'builds', 'built', 'building',
    
        # Generic nouns
        'line', 'lines',
        'result', 'results', 'resulting', 'resulted',
        'file', 'files', 'filed',
        'page', 'pages', 'paged',
    
        # Temporal
        'new', 'news', 'newly', 'newer', 'newest',
        'today', 'yesterday',
        'current', 'currently',
        'latest', 'recent', 'recently',
    }
    
    # Add single letters (noise from tokenization)
    extended_stopwords.update(string.ascii_lowercase)
    ```
    

**Result:** F1-Macro (CV) = 0.335, F1-Macro (Test) = 0.32

### POS-Aware Lemmatization Impact

### What is POS-Aware Lemmatization?

Standard lemmatization treats all words as nouns by default:

```python
# Standard lemmatization
"running" → "running"  # Treated as noun, no change
"better" → "better"    # Treated as noun, no change
"crashes" → "crash"    # Works for nouns
```

POS-aware lemmatization considers the part of speech:

```python
# POS-aware lemmatization
"running" (verb) → "run"
"running" (noun) → "running"  # "the running of the program"
"better" (adjective) → "good"
"crashes" (verb) → "crash"
```

### Performance Impact

| Configuration | Preprocessing Time | F1-Macro | Accuracy |
| --- | --- | --- | --- |
| Simple lemmatization | 2.5 minutes | 0.3320 | 67.5% |
| **POS-aware lemmatization** | **4.1 minutes** | **0.3350** | **68.0%** |

**Trade-off:**
- ✅ **+0.9% F1-Macro improvement**
- ✅ **More accurate word forms**
- ⚠️ **64% longer preprocessing time**

**Recommendation:** Use POS-aware lemmatization for final model training. For quick experiments, simple lemmatization is acceptable.

### Results Comparison

| Strategy | CV F1-Macro | Test F1-Macro | Feature Count |
| --- | --- | --- | --- |
| Baseline (Standard only) | 0.16 | 0.16 | ~1000 |
| **Minimal Custom** | 0.3318 | **0.33** | ~1000 |
| **Extended Custom ⭐** | **0.335** | 0.32 | ~950 |

### Performance by Text Length

| Text Length | Minimal Errors | Extended Errors | Winner |
| --- | --- | --- | --- |
| Very Short (0-20) | 21.2% | 21.9% | Minimal |
| Short (20-50) | 32.9% | 31.4% | **Extended** |
| Medium (50-100) | 47.2% | 42.1% | **Extended** ✅ |
| Long (100-200) | 48.3% | 51.3% | Minimal |
| Very Long (200+) | 42.1% | 56.0% | Minimal |

**Key Insight:** Extended stopwords excel at short-to-medium texts (where most bugs fall) but hurt long descriptions.

### Final Decision

**Chose Extended Stopwords** because:
1. ✅ Best CV performance (0.335)
2. ✅ Better on typical bugs (most are 20-100 words)
3. ✅ Cleaner feature space (~950 vs ~1000 = 5% reduction)
4. ⚠️ Acceptable test trade-off (-3% within statistical noise)

**Decision Factors:**
- 60% of bugs are short-medium length (20-100 words)
- Extended strategy reduces error from 47.2% → 42.1% for this segment
- Long text performance degradation is acceptable given rarity
- Cleaner features improve interpretability

**Production Recommendation:** Use extended stopwords as baseline, with adaptive strategy:
- Texts <100 words: Extended stopwords (optimal)
- Texts 100-200 words: Minimal stopwords (preserve context)
- Texts >200 words: Manual review flag (complex scenarios)

---

## 🔑 Feature Importance

### Top 10 Most Important Features

| Rank | Feature | Importance | Type |
| --- | --- | --- | --- |
| 1 | **crash** | **7.0%** | Text |
| 2 | component | 2.3% | Categorical |
| 3 | product | 2.0% | Categorical |
| 4 | text_length | 1.0% | Numeric |
| 5 | identifier | 0.9% | Text |
| 6 | useragent | 0.8% | Text |
| 7 | work | 0.7% | Text |
| 8 | release | 0.7% | Text |
| 9 | error | 0.6% | Text |
| 10 | stack | 0.6% | Text |

**Key Finding:** Single keyword “crash” accounts for **7%** of all model decisions!

### Feature Type Distribution

- **Text Features (TF-IDF):** 94.7% of total importance
- **Categorical Features:** 4.3% (component + product)
- **Numeric Features:** 1.0% (text_length)

**Insight:** While categorical features contribute only 5%, they provide stable signals that complement noisy text features.

---

## 🐛 Error Analysis

### Overall Statistics

- **Total predictions:** 2,000
- **Correct:** 1,353 (67.7%)
- **Incorrect:** 647 (32.3%)

### Error Rate by Class

| Class | Total | Errors | Error Rate | Accuracy |
| --- | --- | --- | --- | --- |
| **blocker** | 41 | 39 | **95.1%** | 4.9% |
| **minor** | 108 | 98 | **90.7%** | 9.3% |
| **major** | 148 | 113 | 76.4% | 23.6% |
| **trivial** | 61 | 44 | 72.1% | 27.9% |
| **critical** | 121 | 62 | 51.2% | **48.8%** |
| **normal** | 1,521 | 291 | 19.1% | **80.9%** |

**Interpretation:**
- ✅ **Normal class (80.9% accuracy):** Main business value - correctly filters most routine bugs
- ⚠️ **Blocker class (4.9% accuracy):** Extremely difficult due to only 41 test samples
- ✅ **Critical class (48.8% accuracy):** Acceptable for triage assistant role

### Top 10 Misclassification Patterns

| True Label | Predicted Label | Count | Impact |
| --- | --- | --- | --- |
| normal | major | 125 | False escalation |
| major | normal | 86 | Missed priority |
| normal | trivial | 77 | Under-classification |
| minor | normal | 72 | Acceptable merge |
| **critical** | **normal** | **48** | **❌ High risk** |
| normal | critical | 44 | False alarm |
| normal | minor | 40 | Under-classification |
| trivial | normal | 35 | Acceptable merge |
| **blocker** | **normal** | **30** | **❌ Critical miss** |
| minor | major | 14 | False escalation |

### Critical Business Risks

**High-Risk Misses:**
- **78 severe bugs missed** (blocker/critical → normal)
- 30 blockers classified as normal
- 48 critical classified as normal

**False Alarms:**
- **49 false alarms** (normal → blocker/critical)
- 44 normal → critical
- 5 normal → blocker

**Ratio:** 1.6:1 (model misses 1.6 critical bugs for every false alarm)

**Recommendation:** Set confidence threshold to require human review for:
- All predicted blocker/critical bugs (verify true positives)
- All low-confidence predictions (<60%)
- Long bug reports (>200 words)

### Confidence Analysis

| Category | Avg Confidence | Count |
| --- | --- | --- |
| Correct predictions | 34.2% | 1,353 |
| Incorrect predictions | 31.5% | 647 |
| **Gap** | **2.7%** | - |

**Interpretation:** Small confidence gap (2.7%) indicates the model has a limited ability to distinguish correct vs incorrect predictions. This justifies a hybrid ML + human review approach.

### Text Length Paradox

| Length Category | Error Rate | Sample Count |
| --- | --- | --- |
| Very Short (0-20) | 21.9% | 593 |
| Short (20-50) | 31.4% | 835 |
| Medium (50-100) | 42.1% | 430 |
| Long (100-200) | 51.3% | 117 |
| Very Long (200+) | 56.0% | 25 |

**Hypothesis:**
- **Very short texts** (0-20 words) are often clearly trivial (“typo in menu”)
- **Medium texts** (50-100 words) contain ambiguous language
- **Long texts** (200+ words) need context words that were removed by stopwords

**Validation:** Extended stopwords hurt long texts (56% error) vs minimal stopwords (42% error), supporting hypothesis.

---

## 🧪 Ablation Study: Validating “Crash” Importance

### Hypothesis

Feature importance analysis showed “crash” accounts for 7.0% of model decisions. What happens if we remove it as a stopword?

### Experimental Setup

**Control:** Extended stopwords (237 words) **WITHOUT** “crash”
- F1-Macro (CV): 0.3054

**Treatment:** Extended stopwords **WITH** “crash” removed
- F1-Macro (CV): 0.2674

### Results

| Configuration | F1-Macro (CV) | Δ | Critical Recall |
| --- | --- | --- | --- |
| **With “crash”** | **0.3054** | baseline | 50% |
| **Without “crash”** | 0.2674 | **-12.4%** | 38% |

### Analysis

**Impact by Class:**
- **Critical bugs:** Recall dropped 50% → 38% (-12%)
- **Blocker bugs:** Recall dropped 8% → 2% (-6%)
- **Major bugs:** Recall dropped 25% → 20% (-5%)
- **Normal bugs:** Minimal change (81% → 80%)

### Conclusions

✅ **VALIDATED:** Removing “crash” caused catastrophic performance drop:
1. **Overall F1-Macro:** -12.4% (0.3054 → 0.2674)
2. **Critical recall:** -24% relative drop (critical bugs are most affected)
3. **Feature importance ranking is reliable:** Top feature truly drives predictions

✅ **Domain keywords must be preserved:** Words like “crash”, “hang”, “freeze” are essential severity indicators, not noise.

✅ **Model learned genuine patterns:** Not just memorizing data - “crash” correlates with severity across diverse contexts.

---

## 🏭 Production Recommendations

### 1. Hybrid ML + Business Rules

Don’t rely solely on ML predictions. Implement rule-based overrides:

```python
def classify_bug_hybrid(description, component, metadata, ml_model):
    """Hybrid classification: ML + business rules."""

    # Get ML prediction
    ml_prediction, confidence = ml_model.predict_proba(description)

    # Rule 1: Security component always major+
    if component == 'Security':
        return max(ml_prediction, 'major'), 'security_rule'

    # Rule 2: Crash + startup = escalate
    if 'crash' in description and 'startup' in description:
        return max(ml_prediction, 'major'), 'crash_startup_rule'

    # Rule 3: Data loss keywords = escalate
    if any(word in description for word in ['data loss', 'file corrupt', 'lose data']):
        return max(ml_prediction, 'critical'), 'data_loss_rule'

    # Rule 4: Low confidence = require review
    if confidence < 0.60:
        return ml_prediction, 'low_confidence_review_required'

    # Rule 5: Critical predictions = always verify
    if ml_prediction in ['blocker', 'critical']:
        return ml_prediction, 'high_severity_review_required'

    return ml_prediction, 'ml_only'
```

### 2. Active Learning Pipeline

Continuously improve the model with production data:

**Phase 1: Shadow Mode (Weeks 1-4)**
1. Deploy model alongside human reviewers
2. Log both ML predictions and human labels
3. Track agreement rate and error patterns
4. **Don’t automate yet** - just observe

**Phase 2: Assisted Mode (Months 2-3)**
1. Show ML prediction to human reviewers as a suggestion
2. Collect feedback on useful vs misleading predictions
3. Identify systematic errors
4. Retrain monthly with corrected labels

**Phase 3: Automated Mode (Month 4+)**
1. Auto-classify normal/trivial bugs (low risk)
2. Flag blocker/critical for human review (high risk)
3. Random sampling for quality assurance
4. A/B test to measure triage time reduction

### 3. Monitoring Dashboard

Track these metrics weekly:

| Metric | Threshold | Action |
| --- | --- | --- |
| **F1-Macro** | < 0.30 | Investigate & retrain |
| **Critical Recall** | < 40% | Urgent review |
| **False Alarm Rate** | > 30% | Adjust confidence threshold |
| **Prediction Confidence** | Mean < 30% | Model degradation detected |
| **Human Override Rate** | > 40% | Model not trusted, retrain |

### 4. Feedback Loop

```python
# Log every prediction for retraining
{
    'bug_id': 12345,
    'ml_prediction': 'major',
    'ml_confidence': 0.65,
    'human_label': 'critical',  # Ground truth
    'agreement': False,
    'timestamp': '2026-01-10',
    'text_length': 87,
    'component': 'Security'
}
```

**Quarterly retraining schedule:**
- Collect 500+ new labeled examples
- Retrain with combined historical + new data
- Validate on holdout set
- Deploy via A/B test (50/50 traffic)
- Monitor for 2 weeks before full rollout

### 5. Deployment Architecture

Bug Report Submission
↓
Text Preprocessing (< 100ms)
↓
ML Prediction (< 50ms)
↓
Business Rule Layer (< 10ms)
↓
Confidence-based Routing:
├─ High confidence + normal → Auto-assign
├─ High confidence + critical → Human review queue
└─ Low confidence → Expert review queue

**Infrastructure:**
- **Model serving:** FastAPI with gunicorn (3 workers)
- **Latency SLA:** < 200ms p95
- **Availability:** 99.9% uptime
- **Fallback:** Rule-based system if ML unavailable

---

## 🛠 Installation

### System Requirements

- **Python:** 3.8 or higher
- **RAM:** 4GB minimum (8GB recommended for training)
- **Disk:** 2GB free space
- **OS:** Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Option 1: Quick Install

```bash
# Clone repository
git clone https://github.com/firsty-rahma/bug-priority-predictor.git
cd bug-priority-predictor

# Install everything
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"

# Verify
pytest tests/ -v
```

### Option 2: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK
python -c "import nltk; nltk.download('all-corpora')"

# Test
python -c "from src.data.preprocessor import TextPreprocessor; print('✅ Success!')"
```

### Option 3: Development Setup

```bash
# Clone with SSH (for contributors)
git clone git@github.com:firsty-rahma/bug-priority-predictor.git

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

### Troubleshooting

**Issue: ModuleNotFoundError: No module named ‘sklearn’**

```bash
pip install scikit-learn
```

**Issue: NLTK data not found**

```bash
python -c "import nltk; nltk.download('all')"
```

**Issue: Memory error during training**

```bash
# Reduce dataset size in config.py
# Or increase system swap space
```

**Data Download**
The dataset is not included in the repository due to size. Download it separately:

```bash
# Option 1: Direct download
wget https://data.mendeley.com/public-files/datasets/v446tfssgj/files/8666b62f-ef75-45e5-89cd-f49795b9cbee/file_downloaded data/mozilla_bug_report_data.csv

# Option 2: Manual download
# Visit https://data.mendeley.com/datasets/v446tfssgj/1
# Download "mozilla_bug_report_data.csv"
# Place in data/ folder
```

## 📖 Usage

### Training the Model

**Full Pipeline**

```bash
# Run complete training pipeline
python scripts/01_data_exploration.py      # 2 minutes
python scripts/02_text_preprocessing.py    # 4 minutes
python scripts/03_modeling.py              # 5 minutes
python scripts/04_hyperparameter_tuning.py # 18 minutes
python scripts/05_error_analysis.py        # 3 minutes

# Total time: ~32 minutes
```

**Individual Steps**

```bash
# Data exploration only
python scripts/01_data_exploration.py

# Try different stopword strategies
python scripts/02_text_preprocessing.py --stopwords minimal
python scripts/02_text_preprocessing.py --stopwords extended

# Quick model comparison (no tuning)
python scripts/03_modeling.py --quick
```

### Making Predictions

**Interactive Mode**

```bash
python scripts/predict.py
```

**Programmatic Usage**

```python
from src.models.train import ModelTrainer
from src.data.preprocessor import TextPreprocessor
import pandas as pd

# Load trained model
model_data = ModelTrainer.load_model('models/best_model_random_forest_tuned.pkl')
model = model_data['model']
label_encoder = model_data['label_encoder']

# Preprocess new bug
preprocessor = TextPreprocessor()
text = preprocessor.preprocess("Firefox crashes on startup")

# Create feature DataFrame
bug = pd.DataFrame({
    'text_processed': [text],
    'component_name': ['General'],
    'product_name': ['FIREFOX'],
    'text_length': [len(text.split())]
})

# Predict
prediction_encoded = model.predict(bug)[0]
prediction_proba = model.predict_proba(bug)[0]

# Decode
prediction = label_encoder.inverse_transform([prediction_encoded])[0]
confidence = prediction_proba.max()

print(f"Predicted:{prediction} (confidence:{confidence:.2%})")
```

**Batch Prediction**

```python
# Load multiple bugs from CSV
new_bugs = pd.read_csv('new_bugs.csv')

# Preprocess
preprocessor = TextPreprocessor()
new_bugs['text_processed'] = new_bugs.apply(
    lambda row: preprocessor.preprocess(
        preprocessor.combine_text(row['short_desc'], row['long_desc'])
    ),
    axis=1
)

# Predict
predictions = model.predict(new_bugs[['text_processed', 'component_name', 'product_name', 'text_length']])
new_bugs['predicted_severity'] = label_encoder.inverse_transform(predictions)

# Export
new_bugs.to_csv('predictions.csv', index=False)
```

**Running Tests**

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_preprocessor.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Fast tests only (skip slow integration tests)
pytest tests/ -v -m "not slow"
```

**Using Make Commands (if Makefile exists)**

```bash
make test          # Run all tests
make test-fast     # Quick tests without coverage
make train         # Full training pipeline
make predict       # Interactive prediction
make clean         # Remove cache files
```

---

## 📁 Project Structure

```
bug-severity-classification/
├── 📂 data/                           # Datasets
│   ├── mozilla_bug_report_data.csv    # Original data (download separately)
│   ├── bugs_cleaned.csv               # After removing severity_code
│   └── bugs_preprocessed.csv          # After text preprocessing
│
├── 📂 figures/                        # Visualizations
│   ├── confusion_matrix_tuned_final.png
│   ├── feature_importance_detailed.png
│   ├── error_analysis_overview.png
│   └── stopword_comparison.png
│
├── 📂 models/                         # Trained models
│   ├── best_model_random_forest_tuned.pkl  # Final model (150MB)
│   └── .gitkeep
│
├── 📂 results/                        # Experiment results
│   ├── hyperparameter_tuning_summary.txt
│   ├── error_analysis_report.txt
│   └── misclassified_cases.csv
│
├── 📂 src/                            # Source code (reusable library)
│   ├── __init__.py
│   ├── 📂 data/                       # Data processing
│   │   ├── __init__.py
│   │   ├── loader.py                  # Load/save CSV files
│   │   └── preprocessor.py            # Text cleaning, lemmatization
│   │
│   ├── 📂 models/                     # Model training
│   │   ├── __init__.py
│   │   └── train.py                   # ModelTrainer class
│   │
│   ├── 📂 evaluation/                 # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Performance metrics
│   │   └── visualization.py           # Plotting functions
│   │
│   └── 📂 utils/                      # Utilities
│       ├── __init__.py
│       ├── config.py                  # Configuration settings
│       └── custom_transformers.py     # sklearn FeatureCombiner
│
├── 📂 scripts/                        # Executable scripts
│   ├── 01_data_exploration.py         # EDA and cleaning
│   ├── 02_text_preprocessing.py       # Text processing pipeline
│   ├── 03_modeling.py                 # Baseline model training
│   ├── 04_hyperparameter_tuning.py    # Grid search optimization
│   ├── 05_error_analysis.py           # Error analysis & ablation
│   └── predict.py                     # Production prediction script
│
├── 📂 tests/                          # Unit tests (pytest)
│   ├── __init__.py
│   ├── conftest.py                    # Test fixtures
│   ├── test_preprocessor.py           # Text preprocessing tests
│   ├── test_custom_transformers.py    # Feature engineering tests
│   ├── test_models.py                 # Model training tests
│   ├── test_metrics.py                # Evaluation tests
│   ├── test_loader.py                 # Data loading tests
│   └── test_integration.py            # End-to-end tests
│
├── 📂 notebooks/                      # Jupyter notebooks (exploration)
│   ├── 01_eda.ipynb
│   ├── 02_text_analysis.ipynb
│   └── stopword_ablation_study.ipynb
│
├── 📄 requirements.txt                # Python dependencies
├── 📄 requirements-dev.txt            # Development dependencies
├── 📄 pytest.ini                      # Pytest configuration
├── 📄 setup.py                        # Package setup
├── 📄 Makefile                        # Convenience commands
├── 📄 .gitignore                      # Git ignore rules
├── 📄 LICENSE                         # MIT License
└── 📄 README.md                       # This file
```

**Key Files:**
- `src/` - Reusable library code (import in other projects)
- `scripts/` - Executable entry points (run from command line)
- `tests/` - Comprehensive test suite (85%+ coverage)
- `models/` - Trained models (download or train yourself)

---

## 💡 Key Learnings

### What Worked ✅

1. **SMOTE for Class Imbalance (+87% improvement)**
    - Synthetic oversampling enabled minority class learning
    - Applied inside CV folds to prevent data leakage
    - Critical for detecting rare blocker/critical bugs
2. **Domain-Specific Stopwords (+0.96% improvement)**
    - Removed 237 platform/procedural words
    - Preserved severity indicators (crash, hang, error)
    - Systematic A/B testing validated approach
3. **Categorical Features (+5% importance)**
    - Component and product provided stable signals
    - Complemented noisy text features
    - Small but measurable impact
4. **POS-Aware Lemmatization (+0.9% improvement)**
    - More accurate than default lemmatization
    - Handles verbs, adjectives properly
    - Worth the 64% preprocessing time increase
5. **Systematic Evaluation**
    - Ablation study validated feature importance
    - Error analysis by text length revealed patterns
    - Confidence analysis informed deployment strategy

### What Didn’t Work ❌

1. **Text-Only Baseline (F1 = 0.16)**
    - Severe class imbalance crushed performance
    - Model defaulted to predicting “normal” for everything
    - Lesson: Always check class distribution first
2. **Over-Aggressive Stopword Removal**
    - Extended stopwords hurt long texts (56% error)
    - Lost context in detailed bug descriptions
    - Lesson: Domain-specific doesn’t mean one-size-fits-all
3. **Bigrams (ngram_range=(1,2))**
    - Hyperparameter tuning showed unigrams better
    - Bigrams added noise, not signal
    - Lesson: More features ≠ better performance
4. **XGBoost + SMOTE**
    - Memory constraints with 10k samples
    - SMOTE expansion made the dataset too large
    - Lesson: Consider computational resources
5. **Low Confidence Discrimination**
    - Only 2.7% gap between correct/incorrect
    - Can’t reliably detect own mistakes
    - Lesson: Don’t rely solely on model confidence

### Reproducibility Note

Results may vary slightly (~2%) between runs due to:
- Random seed initialization timing
- SMOTE synthetic sample generation
- Parallel processing order (n_jobs=-1)

**Typical ranges observed:**
- CV F1-Macro: 0.330 - 0.335
- Test F1-Macro: 0.30 - 0.32
- Test F1-Weighted: 0.67 - 0.68

All results are within statistical noise and don't affect conclusions.
To ensure exact reproducibility, all random seeds are set to 42.

### Technical Challenges Overcome

1. **Severe Class Imbalance (76% vs 2.1%)**
    - **Solution:** SMOTE + stratified CV + F1-macro metric
    - **Result:** All classes detected (0% → 49% critical recall
2. **Limited Blocker Samples (205 total)**
    - **Solution:** Accepted lower blocker accuracy, focused on critical
    - **Mitigation:** Active learning plan for data collection
3. **Stopword Dilemma (signal vs noise)**
    - **Solution:** Systematic experimentation with 3 strategies
    - **Result:** Extended stopwords for typical cases, documented trade-offs
4. **Context Loss in Long Texts**
    - **Solution:** Identified via text length analysis
    - **Recommendation:** Adaptive preprocessing based on length
5. **Production Deployment Risk**
    - **Solution:** Hybrid ML + business rules approach
    - **Safeguard:** Human review for high-severity predictions

### Lessons for Future Projects

1. ✅ Always do EDA first - Class imbalance was the main challenge
2. ✅ Systematic experimentation - A/B test design decisions
3. ✅ Error analysis by slices - Text length revealed important patterns
4. ✅ Ablation studies - Validate feature importance claims
5. ✅ Document trade-offs - No free lunch in ML
6. ✅ Think about deployment - Model accuracy ≠ business value
7. ✅ Test everything - 85% coverage caught many bugs
8. ✅ Production readiness - Modular code, monitoring, fallbacks

---

## 🔮Future Improvements

### Short-term (1-3 months)

- [ ]  **Two-stage classification**
    - Stage 1: Binary (critical vs non-critical)
    - Stage 2: Fine-grained within each group
    - **Expected:** Better minority class detection
- [ ]  **Additional engineered features**
    - `has_stack_trace`: Boolean
    - `error_code_present`: Boolean
    - `code_snippet_length`: Int
    - `num_attachments`: Int
    - **Expected:** +2-3% F1-macro
- [ ]  **Alternative oversampling methods**
    - Borderline-SMOTE (focus on decision boundary)
    - ADASYN (adaptive synthetic sampling)
    - **Expected:** Better handling of blocker class
- [ ]  **Confidence calibration**
    - Platt scaling or isotonic regression
    - Improve low-confidence detection
    - **Expected:** Better routing decisions

### Medium-term (3-6 months)

- [ ]  **Active learning for blocker bugs**
    - Collect 200+ new blocker examples
    - Deploy uncertainty sampling in production
    - **Expected:** Blocker recall 5% → 20%
- [ ]  **Word embeddings**
    - Word2Vec or GloVe pre-trained on code
    - Capture semantic similarity
    - **Expected:** +5-10% F1-macro
- [ ]  **Ensemble methods**
    - Combine Random Forest + Logistic Regression
    - Different models for different classes
    - **Expected:** +3-5% F1-macro
- [ ]  **Production A/B test**
    - Shadow mode (2 weeks)
    - Assisted mode (1 month)
    - Automated mode (ongoing)
    - **Expected:** 70% triage time reduction

### Long-term (6-12 months)

- [ ]  **Fine-tune BERT for bug classification**
    - Use domain-specific pre-training (GitHub issues, Stack Overflow)
    - Transfer learning from CodeBERT
    - **Expected:** +15-20% F1-macro, state-of-the-art
- [ ]  **Multi-task learning**
    - Joint prediction: severity + priority + component
    - Shared representations improve all tasks
    - **Expected:** Better overall bug understanding
- [ ]  **Explainable AI (SHAP/LIME)**
    - Per-prediction explanations
    - "This bug is critical because of: crash (30%), security (25%), data loss (20%)"
    - **Expected:** Better human trust and oversight
- [ ]  **Real-time API with FastAPI**
    - RESTful endpoint: POST /predict
    - Latency < 200ms p95
    - Kubernetes deployment
    - **Expected:** Easy integration with bug trackers
- [ ]  **Cross-project generalization**
    - Train on Firefox, test on Chromium/Eclipse
    - Domain adaptation techniques
    - **Expected:** Reusable model across projects

### Research Directions

- [ ]  **Few-shot learning for rare classes**
    - Meta-learning approaches (MAML, Prototypical Networks)
    - Learn from 5-10 blocker examples
    - **Expected:** Better handling of extreme imbalance
- [ ]  **Contrastive learning**
    - Learn embeddings where similar severity bugs cluster
    - Use triplet loss or SimCLR
    - **Expected:** Better feature representations
- [ ]  **Causal inference**
    - What *causes* a bug to be critical vs major?
    - Causal feature selection
    - **Expected:** More robust, interpretable models

### What Worked ✅

1. **SMOTE for Class Imbalance (+87% improvement)**
    - Synthetic oversampling enabled minority class learning
    - Applied inside CV folds to prevent data leakage
    - Critical for detecting rare blocker/critical bugs
2. **Domain-Specific Stopwords (+0.96% improvement)**
    - Removed 237 platform/procedural words
    - Preserved severity indicators (crash, hang, error)
    - Systematic A/B testing validated approach
3. **Categorical Features (+5% importance)**
    - Component and product provided stable signals
    - Complemented noisy text features
    - Small but measurable impact
4. **POS-Aware Lemmatization (+0.9% improvement)**
    - More accurate than default lemmatization
    - Handles verbs, adjectives properly
    - Worth the 64% preprocessing time increase
5. **Systematic Evaluation**
    - Ablation study validated feature importance
    - Error analysis by text length revealed patterns
    - Confidence analysis informed deployment strategy

### What Didn't Work ❌

1. **Text-Only Baseline (F1 = 0.16)**
    - Severe class imbalance crushed performance
    - Model defaulted to predicting "normal" for everything
    - Lesson: Always check class distribution first
2. **Over-Aggressive Stopword Removal**
    - Extended stopwords hurt long texts (56% error)
    - Lost context in detailed bug descriptions
    - Lesson: Domain-specific doesn't mean one-size-fits-all
3. **Bigrams (ngram_range=(1,2))**
    - Hyperparameter tuning showed unigrams better
    - Bigrams added noise, not signal
    - Lesson: More features ≠ better performance
4. **XGBoost + SMOTE**
    - Memory constraints with 10k samples
    - SMOTE expansion made dataset too large
    - Lesson: Consider computational resources
5. **Low Confidence Discrimination**
    - Only 2.7% gap between correct/incorrect
    - Can't reliably detect own mistakes
    - Lesson: Don't rely solely on model confidence

### Technical Challenges Overcome

1. **Severe Class Imbalance (76% vs 2.1%)**
    - **Solution:** SMOTE + stratified CV + F1-macro metric
    - **Result:** All classes detected (0% → 49% critical recall)
2. **Limited Blocker Samples (205 total)**
    - **Solution:** Accepted lower blocker accuracy, focused on critical
    - **Mitigation:** Active learning plan for data collection
3. **Stopword Dilemma (signal vs noise)**
    - **Solution:** Systematic experimentation with 3 strategies
    - **Result:** Extended stopwords for typical cases, documented trade-offs
4. **Context Loss in Long Texts**
    - **Solution:** Identified via text length analysis
    - **Recommendation:** Adaptive preprocessing based on length
5. **Production Deployment Risk**
    - **Solution:** Hybrid ML + business rules approach
    - **Safeguard:** Human review for high-severity predictions

### Lessons for Future Projects

1. ✅ **Always do EDA first** - Class imbalance was the main challenge
2. ✅ **Systematic experimentation** - A/B test design decisions
3. ✅ **Error analysis by slices** - Text length revealed important patterns
4. ✅ **Ablation studies** - Validate feature importance claims
5. ✅ **Document trade-offs** - No free lunch in ML
6. ✅ **Think about deployment** - Model accuracy ≠ business value
7. ✅ **Test everything** - 85% coverage caught many bugs
8. ✅ **Production readiness** - Modular code, monitoring, fallbacks

---

## 🔮 Future Improvements

### Short-term (1-3 months)

- [ ]  **Two-stage classification**
    - Stage 1: Binary (critical vs non-critical)
    - Stage 2: Fine-grained within each group
    - **Expected:** Better minority class detection
- [ ]  **Additional engineered features**
    - `has_stack_trace`: Boolean
    - `error_code_present`: Boolean
    - `code_snippet_length`: Int
    - `num_attachments`: Int
    - **Expected:** +2-3% F1-macro
- [ ]  **Alternative oversampling methods**
    - Borderline-SMOTE (focus on decision boundary)
    - ADASYN (adaptive synthetic sampling)
    - **Expected:** Better handling of blocker class
- [ ]  **Confidence calibration**
    - Platt scaling or isotonic regression
    - Improve low-confidence detection
    - **Expected:** Better routing decisions

### Medium-term (3-6 months)

- [ ]  **Active learning for blocker bugs**
    - Collect 200+ new blocker examples
    - Deploy uncertainty sampling in production
    - **Expected:** Blocker recall 5% → 20%
- [ ]  **Word embeddings**
    - Word2Vec or GloVe pre-trained on code
    - Capture semantic similarity
    - **Expected:** +5-10% F1-macro
- [ ]  **Ensemble methods**
    - Combine Random Forest + Logistic Regression
    - Different models for different classes
    - **Expected:** +3-5% F1-macro
- [ ]  **Production A/B test**
    - Shadow mode (2 weeks)
    - Assisted mode (1 month)
    - Automated mode (ongoing)
    - **Expected:** 70% triage time reduction

### Long-term (6-12 months)

- [ ]  **Fine-tune BERT for bug classification**
    - Use domain-specific pre-training (GitHub issues, Stack Overflow)
    - Transfer learning from CodeBERT
    - **Expected:** +15-20% F1-macro, state-of-the-art
- [ ]  **Multi-task learning**
    - Joint prediction: severity + priority + component
    - Shared representations improve all tasks
    - **Expected:** Better overall bug understanding
- [ ]  **Explainable AI (SHAP/LIME)**
    - Per-prediction explanations
    - "This bug is critical because of: crash (30%), security (25%), data loss (20%)"
    - **Expected:** Better human trust and oversight
- [ ]  **Real-time API with FastAPI**
    - RESTful endpoint: POST /predict
    - Latency < 200ms p95
    - Kubernetes deployment
    - **Expected:** Easy integration with bug trackers
- [ ]  **Cross-project generalization**
    - Train on Firefox, test on Chromium/Eclipse
    - Domain adaptation techniques
    - **Expected:** Reusable model across projects

### Research Directions

- [ ]  **Few-shot learning for rare classes**
    - Meta-learning approaches (MAML, Prototypical Networks)
    - Learn from 5-10 blocker examples
    - **Expected:** Better handling of extreme imbalance
- [ ]  **Contrastive learning**
    - Learn embeddings where similar severity bugs cluster
    - Use triplet loss or SimCLR
    - **Expected:** Better feature representations
- [ ]  **Causal inference**
    - What *causes* a bug to be critical vs major?
    - Causal feature selection
    - **Expected:** More robust, interpretable models

---

## ❓ FAQ: How Does This Differ from Commercial Tools?

### "Jira and other issue trackers already have automation. What makes this different?"

Great question! Here's how this project differs from commercial solutions:

| Aspect | Commercial Tools (e.g., Jira) | This Project |
| --- | --- | --- |
| **Domain** | General-purpose issue tracking | Specialized for browser/software bugs |
| **Transparency** | Black-box predictions | Full explainability (feature importance, error analysis) |
| **Customization** | Limited to provided features | Fully customizable (retrain with company data) |
| **Learning Value** | Use existing tool | Demonstrates ML engineering skills |
| **Cost** | License fees (~$7-14/user/month) | Open-source, self-hosted |
| **Data Ownership** | Vendor-controlled | Full control over training data |
| **Evaluation** | Vendor-reported metrics | Transparent, reproducible evaluation |

### What This Project Demonstrates

While commercial tools are excellent for **using** ML in production, this project shows I can:

1. ✅ **Build ML systems from scratch** - End-to-end pipeline from raw data to deployed model
2. ✅ **Handle real ML challenges** - Class imbalance, feature engineering, systematic evaluation
3. ✅ **Think like an engineer** - Modular code, comprehensive tests, production considerations
4. ✅ **Communicate results** - Clear documentation, error analysis, stakeholder-ready reports
5. ✅ **Make data-driven decisions** - A/B testing, ablation studies, trade-off analysis

### When to Use Each Approach

**Use Commercial Tools When:**

- ✅ Already have existing Atlassian/Jira ecosystem
- ✅ Need enterprise support and SLAs
- ✅ Team lacks ML engineering resources
- ✅ Generic use cases (not domain-specific)
- ✅ Need quick deployment (weeks, not months)

**Build Custom Solutions When:**

- ✅ Highly specialized domain (e.g., browser engineering, medical devices)
- ✅ Need full explainability (regulatory/compliance requirements)
- ✅ Want control over training data and model updates
- ✅ Have ML engineering resources in-house
- ✅ Cost savings at scale (thousands of users)

**Hybrid Approach (Recommended):**
Use commercial tools for workflow management and integration, but enhance with custom ML where it adds unique value:

- **Jira for:** Bug tracking, workflow, notifications, reporting
- **Custom ML for:** Domain-specific severity prediction, specialized triage rules
- **Best of both worlds:** Commercial reliability + domain customization

### Real-World Example

**Scenario:** Large tech company with 500 developers, 10,000 bugs/year

**Commercial Tool (Jira + Automation):**

- Cost: $10/user/month × 500 = $60,000/year
- Generic ML model (trained on mixed domains)
- Limited customization
- Vendor lock-in

**Custom ML (This Approach):**

- Cost: $80,000/year ML engineer salary
- Domain-specific model (trained on company's bugs)
- Full customization and explainability
- Retrain monthly with company data
- **Additional value:** Team gains ML engineering capability

**ROI Analysis:**

- Triage time saved: 70% × 2 hours/day × 10 engineers = 1,400 hours/year
- Engineer cost: $50/hour × 1,400 = $70,000/year value
- **Payback period:** ~14 months

---

## 🤝 Contributing

While this is primarily a portfolio project, feedback and suggestions are welcome!

### How to Contribute

1. **🐛 Report Issues**
    - Found a bug? [Open an issue](https://github.com/firsty-rahma/bug-severity-classification/issues)
    - Include: Python version, OS, error message, steps to reproduce
2. **💡 Propose Improvements**
    - Have ideas for better features or models?
    - Start a [GitHub Discussion](https://github.com/firsty-rahma/bug-severity-classification/discussions)
3. **📚 Share Insights**
    - Used this project as a reference?
    - Let me know via LinkedIn or GitHub!

### Areas for Contribution

- [ ]  Additional evaluation metrics (ROC curves, PR curves, calibration plots)
- [ ]  Alternative models (BERT, XGBoost with better memory management)
- [ ]  Deployment examples (Docker, FastAPI, Kubernetes)
- [ ]  Dataset expansion (support for other bug tracking systems)
- [ ]  Multi-language support (bugs in non-English languages)

### Development Guidelines

If you want to contribute code:

```bash
# Fork the repository
git clone https://github.com/your-username/bug-severity-classification.git

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest tests/ -v

# Submit pull request
```

**Code standards:**

- Follow PEP 8 style guide
- Add docstrings (NumPy style)
- Write unit tests for new features
- Update README if needed

---

## 👤 Author

Firstyani Imannisa Rahma - ML Engineer & QA Automation Specialist

### Background

- 🎓 **Education:**
    - Master's in Informatics (Business Intelligence specialization)
    - Bachelor's in Informatics Engineering Education
- 💼 **Experience:** 3+ years as a Software Tester
    - Tested 50+ features of the enterprise sport management system
    - Served US K-12 schools and universities
- 📍 **Location:** Yogyakarta, Indonesia
- 🎯 **Career Goal:** Transitioning to QA Automation & ML Engineering roles

### Connect With Me

- **GitHub:** [@firsty-rahma](https://github.com/firsty-rahma)
- **LinkedIn:** [Firstyani Imannisa Rahma](https://www.linkedin.com/in/firstyani-imannisa-rahma-412990236/)
- **Email:** [[firsty.rahma9521@gmail.com](mailto:firsty.rahma9521@gmail.com)]

### Why This Project?

This project represents my transition from manual testing to ML engineering, demonstrating:

### 🔬 **ML Engineering Skills**

- Built complete pipeline from data exploration → deployment recommendations
- Handled real-world challenges: severe class imbalance (76% vs 2%), noisy text data
- Systematic experimentation: 3 stopword strategies, hyperparameter tuning, ablation studies
- Achieved 109% improvement in F1-Macro through iterative optimization

### 🧪 **QA Mindset**

- Comprehensive error analysis (by class, text length, confidence)
- Edge case identification (long texts, low confidence predictions)
- Test coverage: 85%+ with pytest
- Production thinking: monitoring, fallbacks, hybrid ML + business rules

### 🏭 **Production Readiness**

- Modular architecture: `src/` for library code, `scripts/` for execution
- Configuration management: All settings in `config.py`
- Logging throughout pipeline
- Deployment recommendations: shadow mode → assisted mode → automated

### 📊 **Communication Skills**

- Clear documentation with visual examples
- Stakeholder-ready reports (error analysis, business impact)
- Data-driven decision making (documented all trade-offs)
- Teaching background enables effective knowledge transfer

### Currently Seeking

**Target Roles:**

- QA Automation Engineer
- ML Engineer (NLP focus)
- Software Engineer in Test (SDET)
- Data Scientist (Applied ML)

**Target Companies (Indonesia):**

- **E-commerce:** Tokopedia, Shopee, Bukalapak, Blibli
- **Ride-hailing/Logistics:** Gojek, Grab
- **Fintech:** Flip, Dana, OVO
- **Travel:** Traveloka, Tiket.com
- **EdTech:** Zenius, Ruangguru
- **HealthTech:** Halodoc, Alodokter

**What I Bring:**

- ✅ Strong testing foundation (3+ years manual testing)
- ✅ ML engineering skills (this portfolio project proves it!)
- ✅ Python proficiency (scikit-learn, pandas, pytest)
- ✅ Business awareness (cost-benefit analysis, deployment thinking)
- ✅ Communication skills (teaching background)
- ✅ Self-driven (completed this project independently)

**Let's connect!** I'm open to discussing QA automation and ML opportunities in Indonesian tech companies.

---

## 🙏 Acknowledgments

### Dataset

- **Source:** Gomes, Luiz; Torres, Ricardo; Côrtes, Mario (2021)
- **Title:** "A Dataset for a Long-lived Prediction in FLOSS"
- **Repository:** Mendeley Data, V1
- **DOI:** [10.17632/v446tfssgj.1](https://doi.org/10.17632/v446tfssgj.1)
- **License:** CC BY 4.0

### Technical References

**Machine Learning:**

- Chawla et al. (2002). *"SMOTE: Synthetic Minority Over-sampling Technique."* Journal of Artificial Intelligence Research, 16, 321-357.
- Breiman, L. (2001). *"Random Forests."* Machine Learning, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). *"XGBoost: A Scalable Tree Boosting System."* KDD.

**Natural Language Processing:**

- Salton, G., & Buckley, C. (1988). *"Term-weighting approaches in automatic text retrieval."* Information Processing & Management, 24(5), 513-523.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python.* O'Reilly Media.
- Devlin, J. et al. (2019). *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."* NAACL.

**Software Engineering:**

- Martin, R. C. (2008). *Clean Code: A Handbook of Agile Software Craftsmanship.* Prentice Hall.
- Hunt, A., & Thomas, D. (1999). *The Pragmatic Programmer.* Addison-Wesley.

### Tools & Libraries

- **Python:** Van Rossum, G., & Drake, F. L. (2009). *Python 3 Reference Manual.* CreateSpace.
- **scikit-learn:** Pedregosa et al. (2011). *"Scikit-learn: Machine Learning in Python."* JMLR, 12, 2825-2830.
- **NLTK:** Bird, S., & Loper, E. (2004). *"NLTK: The Natural Language Toolkit."* ACL.
- **imbalanced-learn:** Lemaître et al. (2017). *"Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning."* JMLR, 18(17), 1-5.

### AI Assistance

- **Claude (Anthropic):** Technical consultation, code review, documentation best practices, and debugging assistance throughout development. Claude helped with:
    - Architecture design decisions
    - Python best practices and PEP 8 compliance
    - Test strategy and pytest implementation
    - README structure and technical writing
    - Interview preparation guidance

### Inspiration

- **Mozilla Community:** For maintaining high-quality bug tracking data
- **Kaggle Competitions:** For demonstrating the importance of systematic evaluation
- **Fast.ai:** For emphasizing practical ML over theoretical complexity

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

**Questions or feedback? Feel free to [open an issue](https://github.com/firsty-rahma/bug-severity-classification/issues) or reach out on [LinkedIn](https://www.linkedin.com/in/firstyani-imannisa-rahma-412990236/)!**

---

**Built with 💙 by Firstyani Imannisa Rahma | Yogyakarta, Indonesia | 2026**

</div>
