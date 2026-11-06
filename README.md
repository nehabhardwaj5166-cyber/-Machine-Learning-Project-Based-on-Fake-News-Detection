 — Fake News Detection with Machine Learning

A Streamlit application for data cleaning, exploratory analysis, model comparison and experimental fake/real news classification.

## Dataset
The included dataset is `data/fake_news_dataset.csv` (20,000 rows; columns: `title`, `text`, `date`, `source`, `author`, `category`, `label`).

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit (normally http://localhost:8501).

## Features
- Data quality audit, cleaning and cleaned CSV download
- Interactive EDA with Plotly
- TF-IDF text representation
- Logistic Regression, Linear SVM, Multinomial Naive Bayes and Random Forest
- Stratified 80/20 hold-out evaluation
- Accuracy, precision, recall, F1, confusion matrix, classification report and ROC curve
- Live text prediction using the best model by F1-score

## Important accuracy note
The target requested was 90% accuracy. This is a target, not a result that can responsibly be guaranteed. An initial check of the supplied data found the article text to be highly synthetic/random-looking; text-only baseline models score around chance level (~50–51%) on a stratified hold-out split. This suggests the dataset may not contain a learnable relationship between article content and labels. The application displays the actual scores from its own evaluation rather than hard-coding 90%.

For a meaningful 90%+ result, use a verified, naturally collected and correctly labelled dataset with sufficient real article content. Do not leak labels or split duplicates across train and test. Also evaluate on a separate external dataset before claiming real-world performance.

## Academic integrity
Predictions are experimental and should not be treated as verified fact-checks. Accuracy alone is insufficient; report precision, recall, F1, confusion matrix, class balance and limitations.
