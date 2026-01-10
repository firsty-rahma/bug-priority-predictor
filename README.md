# Bug Severity Classification using NLP and Machine Learning

Automated classification of software bug severity from bug reports using Natural Language Processing and Random Forest with SMOTE oversampling. Achieved **106% improvement** in F1-Macro score through systematic feature engineering and hyperparameter optimization.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Key Results](#-key-results)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Performance Evolution](#-performance-evolution)
- [Stopword Strategy Experimentation](#-stopword-strategy-experimental-analysis)
- [Feature Importance](#-feature-importance)
- [Error Analysis](#-error-analysis)
- [Ablation Study](#-ablation-study)
- [Production Recommendations](#-production-recommendations)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

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

## 🏆 Key Results

### Overall Performance

| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
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

- **Source:** Mozilla Bug Report Data, taken from A Dataset for a Long-lived Prediction in FLOSS
- **Size:** 9,998 bug reports after cleaning
- **Features:** Text descriptions, component, product, metadata
- **Target:** Severity category (6 classes)

### Severe Class Imbalance

| Severity | Count | Percentage | Challenge |
|----------|-------|------------|-----------|
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
→ lemmatize with POS → "crash startup"
```

**Custom Stopword Strategy:**
- Removed 237 domain-specific stopwords
- Preserved severity indicators: `crash`, `hang`, `error`, `freeze`
- See [Stopword Strategy](#-stopword-strategy-experimental-analysis) for detailed analysis

### 2. Handling Severe Class Imbalance

**Problem:** 76% of bugs are "normal", only 2.1% are "blocker"

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
|-------|--------------|---------------|-----------------|--------------|
| **Stage 1** | Text only, class weights | 0.16 | baseline | - |
| **Stage 2** | + SMOTE + Categorical features | 0.30 | **+87.5%** | +87.5% |
| **Stage 3** | + Hyperparameter tuning | 0.3318 | +10.6% | +107% |
| **Stage 4** | + Extended custom stopwords | **0.335** | **+0.96%** | **+109%** |

### Key Improvements by Stage

#### **Stage 1 → 2: Addressing Class Imbalance (+87.5%)**
- **Added:** SMOTE oversampling
- **Added:** Categorical features (component, product, text_length)
- **Impact:** Enabled model to learn minority classes
- **Result:** F1-Macro jumped from 0.16 → 0.30

#### **Stage 2 → 3: Optimization (+10.6%)**
- **Hyperparameter tuning:** Grid search with 5-fold stratified CV
- **Best parameters found:**
  - ngram_range: (1, 1) - unigrams only (bigrams didn't help!)
  - n_estimators: 200
  - max_depth: 20
  - min_samples_split: 10
- **Result:** F1-Macro improved from 0.30 → 0.3318

#### **Stage 3 → 4: Domain-Specific Feature Engineering (+0.96%)**
- **Extended custom stopwords:** 237 domain-specific words removed
- **Trade-off:** CV improved (+0.96%) but test slightly decreased (-3%)
- **Decision:** Accepted for cleaner features and better typical-case performance
- **Result:** F1-Macro improved from 0.3318 → 0.335

---

## 🔍 Stopword Strategy: Experimental Analysis

One of the key challenges in NLP feature engineering is determining which words to remove as "stopwords." This project demonstrates systematic experimentation to find the optimal strategy.

### Motivation

Bug reports contain domain-specific language:
- **Platform terms:** "firefox", "mozilla", "gecko" (appear in all bugs)
- **Procedural language:** "reproduce", "testcase", "step" (reporting process)
- **Generic actions:** "use", "open", "click" (all severity levels)

**Question:** Which words should we remove to maximize signal while minimizing noise?

### Experimental Design

Three strategies tested with 5-fold CV + holdout test:

#### **Strategy 1: Minimal Custom Stopwords (Conservative)**

Removed only 15 procedural words:
```python
minimal_stopwords = {
    'see', 'produce', 'new', 'step',
    'reproduce', 'testcase', 'expect', 'report'
}
```

#### **Strategy 2: Extended Custom Stopwords (Aggressive) ⭐**

Removed 237 domain-specific words across 5 categories:

1. **Platform/Tool:** `firefox`, `mozilla`, `gecko`, `bugzilla`, `windows`
2. **Procedural:** `reproduce`, `testcase`, `expect`, `report`, `step`
3. **Generic Actions:** `use`, `get`, `try`, `open`, `close`, `click`
4. **Generic Nouns:** `file`, `line`, `page`, `result`
5. **Temporal:** `new`, `current`, `latest`, `recent`
6. **Single Letters:** `a`, `b`, `c`, ..., `z`

<details>
<summary><b>View Full Extended Stopword List (Click to Expand)</b></summary>

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

# Add single letters
extended_stopwords.update(string.ascii_lowercase)
```
</details>

### POS-Aware Lemmatization Impact

#### What is POS-Aware Lemmatization?

Standard lemmatization treats all words as nouns by default:
```python
# Standard lemmatization (WordNetLemmatizer with default POS)
"running" -> "running"  # Treated as noun, no change
"better" -> "better"    # Treated as noun, no change
"crashes" -> "crash"    # Works for nouns
```

POS-aware lemmatization considers the part of speech:
```python
# POS-aware lemmatization
"running" (verb) -> "run"
"running" (noun) -> "running"  # "the running of the program"
"better" (adjective) -> "good"
"crashes" (verb) -> "crash"
```

#### Performance Impact

| Configuration | Preprocessing Time | F1-Macro | Accuracy |
|---------------|-------------------|----------|----------|
| Simple lemmatization | 2.5 minutes | 0.3320 | 67.5% |
| **POS-aware lemmatization** | **4.1 minutes** | **0.3350** | **68.0%** |

**Trade-off:**
- ✅ **+0.9% F1-Macro improvement**
- ✅ **More accurate word forms**
- ⚠️ **64% longer preprocessing time**

**Recommendation:** Use POS-aware lemmatization for final model training. For quick experiments, simple lemmatization is acceptable.

### Results Comparison

| Strategy | CV F1-Macro | Test F1-Macro | Feature Count |
|----------|-------------|---------------|---------------|
| Baseline (Standard only) | 0.16 | 0.16 | ~1000 |
| **Minimal Custom** | 0.3318 | **0.33** | ~1000 |
| **Extended Custom ⭐** | **0.335** | 0.32 | ~950 |

### Performance by Text Length

| Text Length | Minimal Errors | Extended Errors | Winner |
|-------------|---------------|-----------------|--------|
| Very Short (0-20) | 21.2% | 21.9% | Minimal |
| Short (20-50) | 32.9% | 31.4% | **Extended** |
| Medium (50-100) | 47.2% | 42.1% | **Extended** ✅ |
| Long (100-200) | 48.3% | 51.3% | Minimal |
| Very Long (200+) | 42.1% | 56.0% | Minimal |

**Key Insight:** Extended stopwords excel at short-to-medium texts but hurt long descriptions.

### Final Decision

**Chose Extended Stopwords** because:
1. ✅ Best CV performance (0.335)
2. ✅ Better on typical bugs (most are short-medium)
3. ✅ Cleaner feature space (~950 vs ~1000)
4. ⚠️ Acceptable test trade-off (-3% within noise)

---

## 🔑 Feature Importance

### Top 10 Most Important Features

| Rank | Feature | Importance | Type |
|------|---------|------------|------|
| 1 | **crash** | **7.0%** | Text |
| 2 | component | 2.3% | Categorical |
| 3 | product | 2.0% | Categorical |
| 4 | text_length | 1.8% | Numeric |
| 5 | identifier | 1.6% | Text |
| 6 | useragent | 1.5% | Text |
| 7 | work | 1.4% | Text |
| 8 | release | 1.4% | Text |
| 9 | error | 1.3% | Text |
| 10 | stack | 1.3% | Text |

**Key Finding:** Single keyword "crash" accounts for **7%** of all model decisions!

---

## 🐛 Error Analysis

### Overall Statistics

- **Total predictions:** 2,000
- **Correct:** 1,353 (67.7%)
- **Incorrect:** 647 (32.3%)

### Error Rate by Class

| Class | Total | Errors | Error Rate | Accuracy |
|-------|-------|--------|------------|----------|
| **minor** | 108 | 98 | **95.4%** | 4.6% |
| **blocker** | 41 | 39 | **92.7%** | 7.3% |
| **major** | 148 | 113 | 83.8% | 16.2% |
| **trivial** | 61 | 44 | 72.1% | 27.9% |
| **critical** | 121 | 62 | 51.2% | **48.8%** |
| **normal** | 1,521 | 291 | 19.1% | **80.9%** |

### Critical Business Risks

**High-Risk Misses:**
- **78 severe bugs missed** (blocker/critical → normal)
  - 30 blockers → normal
  - 48 critical → normal

**False Alarms:**
- **49 false alarms** (normal → blocker/critical)

**Ratio:** 1.6:1 (model too conservative)

### Text Length Paradox

| Length | Error Rate | Observation |
|--------|------------|-------------|
| Very Short (0-20) | 21.9% | ✅ Best |
| Short (20-50) | 31.4% | Good |
| Medium (50-100) | 42.1% | Moderate |
| Long (100-200) | 51.3% | Poor |
| Very Long (200+) | 56.0% | ❌ Worst |

**Hypothesis:** Very short = clearly trivial; Medium = ambiguous; Very long = need context words we removed

---

## 🧪 Ablation Study: Validating "Crash" Importance

### Hypothesis

Feature importance showed "crash" = 7.0% importance. What if we remove it?

### Results

| Configuration | F1-Macro | Δ |
|--------------|----------|---|
| **With "crash"** | **0.3054** | baseline |
| **Without "crash"** | 0.2674 | **-12.4%** |

### Conclusion

✅ **CONFIRMED:** Removing "crash" caused **12.4% drop**, validating:
1. Feature importance rankings are reliable
2. Model genuinely learned severity patterns
3. Domain keywords must be preserved

---

## 🏭 Production Recommendations

### 1. Hybrid ML + Business Rules

```python
def classify_bug(description, component, ml_model):
    ml_prediction, confidence = ml_model.predict_proba(description)
    
    # Override rules
    if 'crash' in description and 'startup' in description:
        return max(ml_prediction, 'major')
    
    if component == 'Security':
        return max(ml_prediction, 'major')
    
    # Confidence routing
    if confidence < 0.60:
        return {'prediction': ml_prediction, 'review_required': True}
    
    if ml_prediction in ['blocker', 'critical']:
        return {'prediction': ml_prediction, 'review_required': True}
    
    return {'prediction': ml_prediction, 'review_required': False}
```

### 2. Active Learning Pipeline

1. **Deploy in Shadow Mode** - Log ML vs human labels
2. **Collect Disagreements** - Prioritize for retraining
3. **Retrain Monthly** - Add human corrections
4. **A/B Test** - Measure triage time reduction

### 3. Monitoring

| Metric | Threshold | Action |
|--------|-----------|--------|
| F1-Macro (weekly) | < 0.30 | Retrain |
| Critical recall | < 40% | Investigate |
| False alarm rate | > 30% | Adjust threshold |

---

## 🚀 Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/bug-severity-classification.git
cd bug-severity-classification

# Run setup
bash setup.sh  # Linux/Mac
setup.bat      # Windows
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 📖 Usage

### Training Pipeline

```bash
# Complete pipeline
python scripts/01_data_exploration.py
python scripts/02_text_preprocessing.py --strategy extended
python scripts/03_modeling.py
python scripts/04_hyperparameter_tuning.py
python scripts/05_error_analysis.py
```

### Making Predictions

```bash
# Interactive mode
python scripts/predict.py
```

```python
# Programmatic usage
from models.train import ModelTrainer
import pandas as pd

model_data = ModelTrainer.load_model('models/best_model_random_forest_tuned.pkl')
model = model_data['model']

bug = pd.DataFrame({
    'text_processed': ['crash startup'],
    'component_name': ['General'],
    'product_name': ['FIREFOX'],
    'text_length': [2]
})

prediction = model.predict(bug)
```

---

## 📁 Project Structure

```
bug-severity-classification/
├── data/
│   ├── bugs.csv
│   ├── bugs_cleaned.csv
│   └── bugs_preprocessed_extended.csv
├── figures/
│   ├── confusion_matrix_tuned_final.png
│   ├── feature_importance_detailed.png
│   └── error_analysis_overview.png
├── models/
│   └── best_model_random_forest_tuned.pkl
├── results/
│   ├── hyperparameter_tuning_summary.txt
│   ├── error_analysis_report.txt
│   └── misclassified_cases.csv
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/
│       ├── config.py
│       └── custom_transformers.py
├── scripts/
│   ├── 01_data_exploration.py
│   ├── 02_text_preprocessing.py
│   ├── 03_modeling.py
│   ├── 04_hyperparameter_tuning.py
│   ├── 05_error_analysis.py
│   └── predict.py
├── tests/
│   ├── test_preprocessor.py
│   ├── test_models.py
│   └── test_integration.py
├── requirements.txt
├── setup.py
├── Makefile
└── README.md
```

---

## 💡 Key Learnings

### What Worked ✅

1. **SMOTE for class imbalance** (+87% improvement)
2. **Domain-specific stopwords** (+0.96% improvement)
3. **Categorical features** (+5% feature importance)
4. **Systematic evaluation** (ablation studies, error analysis)

### What Didn't Work ❌

1. **Text-only baseline** (F1 = 0.16)
2. **Over-aggressive stopword removal** (hurt long texts)
3. **XGBoost + SMOTE** (memory constraints)
4. **Low confidence discrimination** (only 2.8% gap)

### Technical Challenges Overcome

1. **Severe class imbalance** → SMOTE + stratified CV
2. **Limited blocker samples** → Active learning plan
3. **Memory constraints** → Random Forest instead of XGBoost
4. **Stopword dilemma** → Systematic A/B testing

---

## 🔮 Future Improvements

### Short-term
- [ ] Two-stage classification (critical vs non-critical)
- [ ] Additional features (has_stack_trace, error_code_present)
- [ ] Borderline-SMOTE or ADASYN

### Medium-term
- [ ] Collect 200+ blocker examples via active learning
- [ ] Word embeddings (Word2Vec, GloVe)
- [ ] A/B test in production

### Long-term
- [ ] Fine-tune BERT for bug classification
- [ ] Multi-task learning (severity + priority + component)
- [ ] Real-time API with FastAPI
- [ ] Explainable AI (SHAP/LIME)

---
## ❓ FAQ: How Does This Differ from Commercial Tools?

### "Jira and other issue trackers already have automation. What makes this different?"

Great question! Here's how this project differs from commercial solutions:

| Aspect | Commercial Tools (e.g., Jira) | This Project |
|--------|-------------------------------|--------------|
| **Domain** | General-purpose issue tracking | Specialized for browser/software bugs |
| **Transparency** | Black-box predictions | Full explainability (feature importance, error analysis) |
| **Customization** | Limited to provided features | Fully customizable (retrain with company data) |
| **Learning Value** | Use existing tool | Demonstrates ML engineering skills |
| **Cost** | License fees (~$7-14/user/month) | Open-source, self-hosted |

### What This Project Demonstrates

While commercial tools are excellent for **using** ML in production, this project shows I can:

1. ✅ **Build ML systems from scratch** - End-to-end pipeline
2. ✅ **Handle real ML challenges** - Class imbalance, feature engineering, evaluation
3. ✅ **Think like an engineer** - Modular code, tests, production considerations
4. ✅ **Communicate results** - Error analysis, stakeholder documentation

### When to Use Each Approach

**Use Commercial Tools:**
- Existing enterprise ecosystem
- Need support/SLAs
- Generic use cases

**Build Custom Solutions:**
- Highly specialized domain
- Need explainability (compliance/audit)
- Want control over training data
- Have ML engineering resources

**Best Approach:** Hybrid - Use commercial tools for workflow, enhance with custom ML where needed.

## 👤 Author

**Niisa**
- **Background:** Software Tester → QA Automation & ML Engineer
- **Education:** Master's in Informatics | Bachelor's in Informatics Engineering Education
- **Location:** Yogyakarta, Indonesia
- **GitHub:** [@yourusername](https://github.com/yourusername)
- **LinkedIn:** [your-profile](https://linkedin.com/in/your-profile)

### About This Project

Developed as part of my career transition from manual testing to QA automation and machine learning. Demonstrates:

- ✅ ML Engineering: End-to-end pipeline
- ✅ Testing Mindset: Error analysis, edge cases
- ✅ Production Thinking: Business rules, monitoring
- ✅ Scientific Rigor: Systematic experiments
- ✅ Communication: Clear documentation

**Seeking:** QA Automation Engineer or ML Engineer roles in Indonesia.

---

## 🙏 Acknowledgments

- **Dataset:** Gomes, Luiz; Torres, Ricardo; Côrtes, Mario (2021), “A Dataset for a Long-lived Prediction in FLOSS ”, Mendeley Data, V1, doi: 10.17632/v446tfssgj.1
- **AI Assistance:** Claude (Anthropic) for technical consultation, code review, and documentation best practices
- **References:**
  - Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
  - Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32
  - Salton & Buckley (1988). Term-weighting approaches in automatic text retrieval

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

**Questions? Open an issue or reach out on LinkedIn!**

</div>