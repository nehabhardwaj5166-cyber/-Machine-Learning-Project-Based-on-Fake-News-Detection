
import re, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, roc_auc_score)
import joblib

st.set_page_config(page_title="Verity | Fake News Intelligence", page_icon="📰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg,#0b1020 0%,#101a31 55%,#111827 100%); color:#eef2ff; }
.block-container { padding-top: 1.8rem; max-width: 1450px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; color:#f8fafc; }
[data-testid="stMetric"] { background:rgba(30,41,59,.72); border:1px solid rgba(148,163,184,.18); padding:16px 18px; border-radius:16px; }
[data-testid="stSidebar"] { background:#0a1224; }
div[data-testid="stTabs"] button { font-weight:600; }
.hero { padding:26px 30px; border-radius:22px; background:linear-gradient(110deg,rgba(30,64,175,.42),rgba(13,148,136,.20)); border:1px solid rgba(125,211,252,.22); margin-bottom:22px; }
.hero-kicker { color:#67e8f9; text-transform:uppercase; letter-spacing:2px; font-size:12px; font-weight:700; }
.hero h1 { font-size:38px; margin:7px 0 4px; }
.hero p { color:#cbd5e1; margin:0; }
.panel { background:rgba(30,41,59,.62); border:1px solid rgba(148,163,184,.16); border-radius:16px; padding:18px; }
.small-note { color:#94a3b8; font-size:13px; }
.stButton>button { border-radius:12px; border:0; background:linear-gradient(90deg,#2563eb,#0d9488); color:white; font-weight:700; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data" / "fake_news_dataset.csv"

def clean_text(value):
    if pd.isna(value): return ""
    value = str(value).lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"http\S+|www\.\S+", " ", value)
    value = re.sub(r"[^a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()

@st.cache_data
def load_data(uploaded_bytes=None):
    if uploaded_bytes is not None:
        from io import BytesIO
        raw = pd.read_csv(BytesIO(uploaded_bytes))
    else:
        raw = pd.read_csv(DATA_PATH)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    return raw

def prepare_data(raw):
    required = {"title", "text", "label"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    data = raw.copy()
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("string").str.strip()
            data[col] = data[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    data["title"] = data["title"].fillna("")
    data["text"] = data["text"].fillna("")
    data["label"] = data["label"].astype("string").str.lower().str.strip()
    data = data[data["label"].notna() & data["label"].isin(["fake", "real"])].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce") if "date" in data else pd.NaT
    data["combined_text"] = (data["title"].fillna("") + " " + data["text"].fillna("")).map(clean_text)
    data = data[data["combined_text"].str.len() > 0].copy()
    data = data.drop_duplicates(subset=["title", "text", "label"])
    return data.reset_index(drop=True)

@st.cache_resource(show_spinner="Cleaning data and training four models…")
def train_models(data_frame):
    X = data_frame["combined_text"].astype(str)
    y = data_frame["label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=35,
                                                 min_samples_leaf=2, class_weight="balanced",
                                                 random_state=42, n_jobs=-1)
    }
    results, fitted = [], {}
    for name, classifier in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=60000, ngram_range=(1,2),
                                      min_df=2, max_df=0.98, sublinear_tf=True,
                                      stop_words="english")),
            ("classifier", classifier)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, pos_label="fake", zero_division=0),
            "Recall": recall_score(y_test, pred, pos_label="fake", zero_division=0),
            "F1-score": f1_score(y_test, pred, pos_label="fake", zero_division=0)
        })
        fitted[name] = pipe
    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    best_name = results_df.iloc[0]["Model"]
    return results_df, fitted, best_name, X_test, y_test

def chart_theme(fig, height=360):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(15,23,42,.15)", font_color="#dbeafe",
                      margin=dict(l=20,r=20,t=50,b=25),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)")
    return fig

st.markdown("""
<div class="hero">
 <div class="hero-kicker">AI-powered media verification workspace</div>
 <h1>Verity <span style="color:#67e8f9">/</span> Fake News Intelligence</h1>
 <p>Explore the dataset, audit data quality, compare machine-learning models and test a news headline.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📰 Verity")
    st.caption("FAKE NEWS DETECTION")
    upload = st.file_uploader("Use another CSV dataset", type=["csv"])
    st.markdown("---")
    st.markdown("**Navigation**")
    st.caption("Dashboard · Data quality · EDA · Model lab · Predictor")
    st.markdown("---")
    st.caption("Model results are measured on a held-out test split. No accuracy is fabricated.")

try:
    raw_df = load_data(upload.getvalue() if upload else None)
    df = prepare_data(raw_df)
except Exception as e:
    st.error(f"Could not load/clean the dataset: {e}")
    st.stop()

if len(df) < 20:
    st.error("At least 20 usable labelled rows are needed to train the models.")
    st.stop()

with st.spinner("Preparing evaluation…"):
    results, models, best_name, X_test, y_test = train_models(df)
best_model = models[best_name]
best_acc = float(results.iloc[0]["Accuracy"])

# Top navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["✨ Overview", "🧹 Data quality", "📊 Exploratory analysis", "🧠 Model lab", "🔎 Live predictor"]
)

with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Clean records", f"{len(df):,}", f"{len(raw_df)-len(df):,} removed")
    c2.metric("Fake news", f"{(df.label=='fake').sum():,}")
    c3.metric("Real news", f"{(df.label=='real').sum():,}")
    c4.metric("Best test accuracy", f"{best_acc:.1%}")
    st.markdown("### Dataset at a glance")
    left,right = st.columns([1.15,1])
    with left:
        counts = df["label"].value_counts().rename_axis("Label").reset_index(name="Articles")
        fig = px.pie(counts, names="Label", values="Articles", hole=.62,
                     color="Label", color_discrete_map={"fake":"#fb7185","real":"#2dd4bf"},
                     title="Label distribution")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(chart_theme(fig,380), use_container_width=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Project workflow")
        st.markdown("1. Load and inspect the raw CSV  \n2. Handle missing values and duplicates  \n3. Normalize text and parse dates  \n4. Convert text into TF-IDF features  \n5. Train and compare four classifiers  \n6. Evaluate on a stratified 20% hold-out set")
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("The supplied dataset contains randomly generated-looking article text. Review the data-quality and model-lab tabs before interpreting the score as real-world performance.")

with tab2:
    st.markdown("### Cleaning audit")
    before_missing = raw_df.isna().sum().sum()
    after_missing = df.isna().sum().sum()
    dup_before = raw_df.duplicated(subset=[c for c in ["title","text","label"] if c in raw_df.columns]).sum()
    audit = pd.DataFrame({
        "Check":["Rows before cleaning","Rows after cleaning","Duplicate rows in raw data","Missing cells before cleaning","Missing cells after cleaning","Invalid/missing labels removed","Empty text rows removed"],
        "Result":[len(raw_df),len(df),int(dup_before),int(before_missing),int(after_missing),
                  int((~raw_df.get("label",pd.Series(dtype=str)).astype(str).str.lower().isin(["fake","real"])).sum()),
                  int(((raw_df.get("title",pd.Series("",index=raw_df.index)).fillna("").astype(str)+" "+raw_df.get("text",pd.Series("",index=raw_df.index)).fillna("").astype(str)).str.strip()=="").sum())]
    })
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.markdown("#### Cleaning operations")
    st.markdown("- Standardized column names and trimmed text fields.\n- Replaced blank-like values with missing values.\n- Filled missing title/body with empty strings for text concatenation.\n- Removed rows without valid `fake`/`real` labels.\n- Parsed dates using safe datetime conversion; invalid dates become `NaT`.\n- Removed duplicate `(title, text, label)` records.\n- Removed records with no usable article text.\n- Lowercased text, removed HTML/URLs/punctuation and collapsed whitespace.")
    st.markdown("#### Cleaned preview")
    st.dataframe(df.drop(columns=["combined_text"]).head(20), use_container_width=True)
    st.download_button("⬇️ Download cleaned CSV", df.drop(columns=["combined_text"]).to_csv(index=False).encode("utf-8"),
                       "cleaned_fake_news.csv", "text/csv")

with tab3:
    st.markdown("### Exploratory data analysis")
    a,b = st.columns(2)
    with a:
        if "category" in df:
            cat = df.groupby(["category","label"]).size().reset_index(name="Count")
            fig = px.bar(cat, x="category", y="Count", color="label", barmode="group",
                         title="Articles by category", color_discrete_map={"fake":"#fb7185","real":"#2dd4bf"})
            st.plotly_chart(chart_theme(fig), use_container_width=True)
        else: st.info("No category column found.")
    with b:
        if "source" in df:
            src = df["source"].fillna("Unknown").value_counts().head(12).reset_index()
            src.columns=["Source","Articles"]
            fig=px.bar(src.sort_values("Articles"),x="Articles",y="Source",orientation="h",title="Top 12 sources")
            st.plotly_chart(chart_theme(fig), use_container_width=True)
        else: st.info("No source column found.")
    a,b=st.columns(2)
    with a:
        df["word_count"] = df["combined_text"].str.split().str.len()
        fig=px.histogram(df,x="word_count",color="label",nbins=35,barmode="overlay",
                         opacity=.78,title="Article length (word count)",
                         color_discrete_map={"fake":"#fb7185","real":"#2dd4bf"})
        st.plotly_chart(chart_theme(fig),use_container_width=True)
    with b:
        if "date" in df and df["date"].notna().any():
            dated=df.dropna(subset=["date"]).assign(month=lambda x:x["date"].dt.to_period("M").astype(str))
            trend=dated.groupby(["month","label"]).size().reset_index(name="Articles")
            fig=px.line(trend,x="month",y="Articles",color="label",markers=True,title="Article volume over time",
                        color_discrete_map={"fake":"#fb7185","real":"#2dd4bf"})
            st.plotly_chart(chart_theme(fig),use_container_width=True)
        else: st.info("No valid dates available for a time trend.")
    st.markdown("#### Numeric summary")
    st.dataframe(df["word_count"].describe().to_frame("Word count"),use_container_width=True)

with tab4:
    st.markdown("### Model comparison")
    st.caption("Same stratified train/test split and TF-IDF representation are used for each model.")
    st.dataframe(results.style.format({c:"{:.2%}" for c in ["Accuracy","Precision","Recall","F1-score"]}),
                 use_container_width=True, hide_index=True)
    metric_choice=st.selectbox("Compare metric",["Accuracy","Precision","Recall","F1-score"],index=0)
    fig=px.bar(results.sort_values(metric_choice,ascending=True),x=metric_choice,y="Model",orientation="h",
               text=results.sort_values(metric_choice,ascending=True)[metric_choice],
               title=f"Model comparison — {metric_choice}")
    fig.update_traces(texttemplate="%{text:.1%}",textposition="outside")
    fig.update_xaxes(range=[0,1.08],tickformat=".0%")
    st.plotly_chart(chart_theme(fig,400),use_container_width=True)
    pred=best_model.predict(X_test)
    cm=confusion_matrix(y_test,pred,labels=["fake","real"])
    fig=px.imshow(cm,text_auto=True,x=["Predicted fake","Predicted real"],y=["Actual fake","Actual real"],
                  color_continuous_scale="Tealgrn",title=f"Confusion matrix — {best_name}")
    st.plotly_chart(chart_theme(fig,380),use_container_width=True)
    st.markdown("#### Classification report")
    st.code(classification_report(y_test,pred,zero_division=0))
    if hasattr(best_model,"decision_function"):
        scores=best_model.decision_function(X_test)
        positive=(y_test=="real").astype(int)
        fpr,tpr,_=roc_curve(positive,scores,pos_label=1)
        auc=roc_auc_score(positive,scores)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"ROC (AUC={auc:.3f})",line=dict(width=3)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line=dict(dash="dash")))
        fig.update_layout(title="ROC curve (real class)",xaxis_title="False positive rate",yaxis_title="True positive rate")
        st.plotly_chart(chart_theme(fig),use_container_width=True)
    st.warning("A score near 50% on this dataset is not evidence of a working detector. The supplied article text appears largely random and may contain no learnable relationship with the labels. A 90% score cannot be guaranteed honestly.")

with tab5:
    st.markdown("### Check a news story")
    st.caption("Experimental educational prediction only — verify important claims with trusted sources.")
    title=st.text_input("News headline",placeholder="Enter the news headline…")
    body=st.text_area("News article text",height=190,placeholder="Paste the article content here…")
    if st.button("Analyze article",type="primary",use_container_width=True):
        sample=clean_text(title+" "+body)
        if len(sample.split())<3:
            st.warning("Please enter a headline and/or article with at least three words.")
        else:
            label=best_model.predict([sample])[0]
            if hasattr(best_model,"decision_function"):
                score=float(best_model.decision_function([sample])[0])
                confidence=float(1/(1+np.exp(-min(max(abs(score),0),20))))
            elif hasattr(best_model,"predict_proba"):
                probs=best_model.predict_proba([sample])[0]
                confidence=float(np.max(probs))
            else: confidence=float("nan")
            if label=="fake":
                st.error(f"Model prediction: {label.upper()}")
            else:
                st.success(f"Model prediction: {label.upper()}")
            if np.isfinite(confidence): st.metric("Model score (not a calibrated probability)",f"{confidence:.1%}")
            st.caption(f"Classifier: {best_name} · This is a model output, not a fact-check.")
    st.markdown("---")
    if st.button("Save best model locally"):
        joblib.dump(best_model,Path(__file__).parent/"best_fake_news_model.joblib")
        st.success("Saved as best_fake_news_model.joblib in the project folder.")

st.markdown("---")
st.markdown('<p class="small-note">VERITY · Academic ML project · Results depend on dataset quality, split and preprocessing. Not a substitute for professional fact-checking.</p>',unsafe_allow_html=True)
