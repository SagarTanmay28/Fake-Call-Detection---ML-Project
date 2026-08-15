# Fake-Call-Detection---ML-Project
Emergency System - Fake Call Detection in Real 911 Scenario


Machine learning system that flags fake/prank 911 emergency calls from real call metadata and transcribed descriptions, built during a Machine Learning Internship at **CDAC** (Jan 2025 – May 2025).

## Problem

Emergency dispatch centers waste critical response time and resources on hoax or fake 911 calls. This project builds a classifier that scores incoming call records for the likelihood of being fake, so dispatchers can prioritize genuine emergencies.

## Dataset

- **100,000+** real-world 911 call records.
- Includes structured fields (call duration, time of day, location metadata, caller history, etc.) as well as free-text **call descriptions**.

## Approach

**1. Data Pipeline**
Raw call records are cleaned, missing values handled, and structured fields normalized. Free-text descriptions are cleaned separately (lowercasing, noise removal) before NLP processing.

**2. Feature Engineering**
- Structured features: call duration, time-of-day buckets, caller/location signals, historical call frequency from the same source.
- **NLP-derived features** from call descriptions: text length, keyword/phrase patterns associated with genuine distress vs. hoax language, sentiment/urgency indicators, and vectorized text representations (e.g., TF-IDF or embeddings) fed into the model alongside structured features.

**3. Modeling**
- **Ensemble models** (e.g., Random Forest / Gradient Boosting) — strong baseline for tabular + engineered NLP features, robust to noisy real-world data.
- **Artificial Neural Networks (ANNs)** — trained to capture non-linear interactions between structured and text-derived features that tree-based models may miss.
- Model outputs were compared, and results combined/ensembled where it improved generalization.

**4. Evaluation**
- **Precision** — of calls flagged as fake, how many actually were fake (minimizing false alarms that could delay attention to real calls incorrectly flagged).
- **Recall** — of all actual fake calls, how many were correctly caught.
- **Confusion Matrix** — used to inspect the false positive / false negative trade-off directly, since in this domain a **false negative (missing a real fake call)** and a **false positive (flagging a genuine emergency as fake)** carry very different real-world costs.
- **Result: 98%+ classification accuracy**, with precision/recall balance tuned to minimize false positives — reducing dispatcher time wasted on hoax calls without risking misclassifying genuine emergencies.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML Models | Ensemble methods (e.g., Random Forest / Gradient Boosting), ANN |
| NLP | Text preprocessing, feature extraction from call descriptions |
| Evaluation | scikit-learn (Precision, Recall, Confusion Matrix) |

## Project Structure

```
fake-call-detection/
├── data/                  # Raw + processed 911 call records
├── preprocessing/         # Cleaning, feature engineering, NLP pipeline
├── models/                # Ensemble model + ANN training scripts
├── evaluation/            # Precision/Recall/Confusion Matrix reporting
├── notebooks/             # EDA and experimentation
└── README.md
```

> Update this tree to reflect your actual repo layout.

## Setup

```bash
git clone <repo-url>
cd fake-call-detection
pip install -r requirements.txt
```

**Typical dependencies:**
```
pandas, numpy
scikit-learn
tensorflow / keras or pytorch   # for the ANN
nltk / spacy                     # NLP preprocessing
matplotlib, seaborn              # for confusion matrix / result plots
```

## Usage

```bash
# Preprocess raw call data
python preprocessing/build_features.py --input data/raw_calls.csv --output data/features.csv

# Train models
python models/train_ensemble.py --data data/features.csv
python models/train_ann.py --data data/features.csv

# Evaluate
python evaluation/evaluate.py --model models/best_model.pkl --test data/test_split.csv
```

> Replace with your actual script names/paths.

## Results

| Metric | Value |
|---|---|
| Accuracy | 98%+ |
| Precision | Tuned to minimize false alarms |
| Recall | Tuned to minimize missed fake calls |

## Acknowledgements

Developed as part of a Machine Learning Internship at **CDAC**, Jan 2025 – May 2025.
