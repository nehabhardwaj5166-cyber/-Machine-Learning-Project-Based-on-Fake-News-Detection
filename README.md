 #  Fake News Detection using Machine Learning

A Python-only Streamlit application for exploring a labelled news dataset, cleaning text, comparing four classical machine-learning classifiers, and screening user-entered articles.

## Models
1. Logistic Regression(LR)
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (Linear SVM)
4. Random Forest

## Project structure
```text
fake_news_detection_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── data/
│   └── fake_news_dataset.csv
├── models/                 # generated after training
└── outputs/                # generated after training
```

## Setup (Windows PowerShell)
```powershell
cd fake_news_detection_project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```.

## Run dashboard
```powershell
streamlit run app.py
```
Open the local URL printed by Streamlit (normally http://localhost:8501).

## Dataset
The supplied `data/fake_news_dataset.csv` is included. Required columns are `title`, `text`, and `label`; optional metadata columns include `date`, `source`, `author`, and `category`. Accepted labels are `fake` and `real` (case-insensitive).

## Methodology
- Exact duplicate removal and missing-value treatment.
- Text normalization and title/article concatenation.
- Stratified 80/20 train-test split.
- Accuracy, precision, recall, F1, ROC-AUC and confusion matrices.
- Three-fold stratified cross-validation for the selected candidate.
- The best model is selected by hold-out test F1 for the demonstration app.

## Important limitations
The included dataset should be audited for provenance, representativeness, and possible synthetic or artificial text. Predictive scores on this dataset do not establish real-world fact-checking ability. Do not use this prototype to label people, make consequential decisions, or publish claims without independent verification. No accuracy target is hard-coded; reported results come from the actual test split.
