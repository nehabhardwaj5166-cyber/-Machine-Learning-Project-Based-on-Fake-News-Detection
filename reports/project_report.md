# Overview Project Report:  Fake News Detection Using Machine Learning

## 1. Problem Statement
Misinformation can spread rapidly through online platforms. This project explores a text-classification workflow that distinguishes news articles labelled `fake` or `real` using supervised machine learning.

## 2. Objectives
- Inspect and clean the supplied dataset.
- Explore class balance, source/category distributions, article length and date trends.
- Convert article text into TF-IDF features.
- Train and compare four classifiers.
- Evaluate with a held-out test set and provide an interactive prediction interface.

## 3. Dataset Description
The supplied CSV contains 20,000 records and seven columns: title, text, date, source, author, category and label. The target variable is `label` (`fake`/`real`). The classes are approximately balanced (10,056 fake and 9,944 real before cleaning).

## 4. Data Cleaning
The application standardizes column names, trims string values, handles blank/missing text, removes invalid labels, parses dates safely, removes duplicate title-text-label rows, removes empty articles, and normalizes article text by lowercasing and removing HTML tags, URLs, punctuation and repeated whitespace. The cleaning audit and cleaned CSV download are available in the app.

## 5. Exploratory Data Analysis
The app visualizes class distribution, category by label, top sources, article word-count distribution and monthly article volume when valid dates are available.

## 6. Methodology
The cleaned title and body are concatenated. TF-IDF with unigrams and bigrams is used. A stratified 80/20 train-test split (random_state=42) is applied. Models: Logistic Regression, Linear SVM, Multinomial Naive Bayes and Random Forest.

## 7. Evaluation
Accuracy, precision, recall and F1-score are computed on the held-out test set. The app also presents a confusion matrix, classification report and ROC curve where supported. See the Model Lab tab for the exact run-specific values.

### Accuracy target and limitation
A 90% score is an objective, not a guaranteed outcome. Initial text-only experiments on this supplied CSV yielded approximately 50–51% accuracy, close to chance. The article text appears synthetic/random-looking and does not show a useful relationship with the target labels. Therefore, reporting 90% for this dataset would be misleading. A higher score requires better source data, not an altered metric or leakage.

## 8. Conclusion
The project implements a complete and reproducible Streamlit ML workflow. Its main finding is that data quality determines whether a classifier can learn a meaningful signal. The current dataset is suitable for demonstrating the pipeline and evaluation interface, but appears unsuitable for demonstrating high-performing real-world fake-news detection.

## 9. Limitations and Future Scope
- Validate labels and provenance with a trusted dataset.
- Test on a separate source/time period to measure generalization.
- Add source/domain shift checks and human fact-checking references.
- Explore transformer models only after dataset quality is established.
- Calibrate probabilities and assess fairness and robustness.

## 10. Reproducibility
Install dependencies using `pip install -r requirements.txt`, then run `streamlit run app.py`. The split seed and model settings are defined in `app.py`.
