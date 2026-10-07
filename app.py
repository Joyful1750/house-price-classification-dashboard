
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)

st.set_page_config(
    page_title="House Price Classification Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Classification Dashboard")
st.caption("Exploratory analysis, statistical testing, logistic regression, and test-set predictions")
st.write("My Streamlit dashboard is working!")
# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("house_train.csv")
    test = pd.read_csv("house_test.csv")
    return train, test

train_raw, test_raw = load_data()

# -----------------------------
# Data cleaning helpers
# -----------------------------
def convert_age(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except (ValueError, TypeError):
        pass

    words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20,
        "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90
    }
    s = str(x).strip().lower().replace("-", " ")
    if s in words:
        return words[s]
    parts = s.split()
    if len(parts) == 2 and parts[0] in words and parts[1] in words:
        return words[parts[0]] + words[parts[1]]
    return np.nan

def clean_data(df):
    d = df.copy()
    d["age"] = d["age"].apply(convert_age)

    # Negative population values are invalid for this practice dataset.
    d.loc[d["population"] < 0, "population"] = np.nan

    # Make sure numeric columns are numeric.
    numeric_cols = [
        "age", "rooms", "bedrooms", "population",
        "income", "persons_per_house", "median_selling_price"
    ]
    for col in numeric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d["classification"] = pd.to_numeric(d["classification"], errors="coerce")
    return d

train = clean_data(train_raw)
test = clean_data(test_raw)

features = [
    "age", "rooms", "bedrooms",
    "population", "income", "persons_per_house"
]
target = "classification"

X_train = train[features]
y_train = train[target].astype(int)

X_test = test[features]
y_test = test[target].astype(int)

# -----------------------------
# Model
# -----------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]

test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Dashboard Sections")
section = st.sidebar.radio(
    "Go to:",
    [
        "Overview",
        "Data Exploration",
        "Statistical Tests",
        "Logistic Regression",
        "Model Evaluation",
        "Individual Prediction",
        "Test Predictions"
    ]
)

# -----------------------------
# Overview
# -----------------------------
if section == "Overview":
    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Houses", f"{len(train):,}")
    c2.metric("Test Houses", f"{len(test):,}")
    c3.metric("Test Accuracy", f"{accuracy_score(y_test, test_pred):.1%}")
    c4.metric("Test ROC-AUC", f"{roc_auc_score(y_test, test_prob):.3f}")

    st.markdown("""
    ### Objective

    Predict whether a house belongs to the **High** or **Low** selling-price
    category using demographic and housing characteristics.

    ### Modeling workflow

    **Data → Cleaning → EDA → Statistical Testing → Logistic Regression → 
    Test Evaluation → Prediction**

    ### Predictors

    - Age
    - Rooms
    - Bedrooms
    - Population
    - Income
    - Persons per house

    The `classification` variable is the target. `median_selling_price` is
    intentionally not used as a predictor because it defines the outcome.
    """)

    st.info(
        "The model is trained on the training dataset and evaluated on the "
        "separate test dataset. The test labels are available because this is "
        "a practice dataset."
    )

# -----------------------------
# Data Exploration
# -----------------------------
elif section == "Data Exploration":
    st.header("🔎 Data Exploration")

    st.subheader("Dataset preview")
    display_df = train.copy()
    display_df["classification"] = display_df["classification"].map({0: "Low", 1: "High"})
    st.dataframe(display_df.head(10), use_container_width=True)

    st.subheader("Dataset information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(train):,}")
    c2.metric("Columns", f"{train.shape[1]:,}")
    c3.metric("Missing values", f"{int(train.isna().sum().sum()):,}")

    st.subheader("Class distribution")
    class_counts = train["classification"].map({0: "Low", 1: "High"}).value_counts()
    st.bar_chart(class_counts)

    st.subheader("Explore a variable")
    selected = st.selectbox("Select numeric variable", features)

    fig, ax = plt.subplots()
    sns.boxplot(
        data=train.assign(
            classification=train["classification"].map({0: "Low", 1: "High"})
        ),
        x="classification",
        y=selected,
        ax=ax
    )
    ax.set_title(f"{selected.replace('_', ' ').title()} by Price Classification")
    st.pyplot(fig)

    st.subheader("Income vs. Rooms")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=train.assign(
            classification=train["classification"].map({0: "Low", 1: "High"})
        ),
        x="income",
        y="rooms",
        hue="classification",
        ax=ax
    )
    ax.set_title("Income vs. Rooms")
    st.pyplot(fig)

# -----------------------------
# Statistical Tests
# -----------------------------
elif section == "Statistical Tests":
    st.header("📊 Statistical Tests")

    st.write(
        "Welch's independent two-sample t-test compares the mean of each "
        "numeric predictor between Low- and High-price houses."
    )

    results = []
    for col in features:
        high = train.loc[train[target] == 1, col].dropna()
        low = train.loc[train[target] == 0, col].dropna()
        result = ttest_ind(high, low, equal_var=False)
        results.append({
            "Variable": col,
            "p-value": result.pvalue,
            "Significant at 0.05": "Yes" if result.pvalue < 0.05 else "No"
        })

    results_df = pd.DataFrame(results)
    st.dataframe(
        results_df.style.format({"p-value": "{:.4g}"}),
        use_container_width=True
    )

    st.caption(
        "A nonsignificant t-test does not automatically mean a predictor is "
        "useless for multivariable prediction. These tests are used here as "
        "supporting EDA rather than an automatic variable-selection rule."
    )

# -----------------------------
# Logistic Regression
# -----------------------------
elif section == "Logistic Regression":
    st.header("🤖 Logistic Regression")

    st.write(
        "The model estimates the probability that a house belongs to the "
        "High-price class."
    )

    st.code(
        """logit(P(High)) = β0 + β1(age) + β2(rooms) + β3(bedrooms)
                 + β4(population) + β5(income) + β6(persons_per_house)"""
    )

    classifier = model.named_steps["classifier"]
    coefficients = classifier.coef_[0]

    coef_df = pd.DataFrame({
        "Variable": features,
        "Coefficient": coefficients,
        "Odds Ratio": np.exp(coefficients)
    })

    st.subheader("Coefficients and odds ratios")
    st.dataframe(
        coef_df.style.format({
            "Coefficient": "{:.4f}",
            "Odds Ratio": "{:.4f}"
        }),
        use_container_width=True
    )

    st.info(
        "Because the numeric predictors are standardized before fitting, "
        "the coefficient magnitude is interpreted per one standard deviation "
        "increase in the predictor. An odds ratio above 1 indicates higher "
        "odds of the High class; below 1 indicates lower odds, holding the "
        "other predictors constant."
    )

# -----------------------------
# Model Evaluation
# -----------------------------
elif section == "Model Evaluation":
    st.header("📈 Model Evaluation")

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    train_auc = roc_auc_score(y_train, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)

    st.subheader("Training vs. Test Performance")

    metric_df = pd.DataFrame({
        "Metric": ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"],
        "Training": [
            train_accuracy,
            train_auc,
            precision_score(y_train, train_pred),
            recall_score(y_train, train_pred),
            f1_score(y_train, train_pred)
        ],
        "Test": [
            test_accuracy,
            test_auc,
            precision_score(y_test, test_pred),
            recall_score(y_test, test_pred),
            f1_score(y_test, test_pred)
        ]
    })

    st.dataframe(
        metric_df.style.format({
            "Training": "{:.3f}",
            "Test": "{:.3f}"
        }),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Test Confusion Matrix")
        cm = confusion_matrix(y_test, test_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Low", "High"],
            yticklabels=["Low", "High"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with c2:
        st.subheader("Test ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Test Classification Report")
    report = classification_report(
        y_test, test_pred,
        target_names=["Low", "High"],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.style.format({
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1-score": "{:.3f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

# -----------------------------
# Individual Prediction
# -----------------------------
elif section == "Individual Prediction":
    st.header("🏡 Individual House Prediction")
    st.write("Enter house characteristics to obtain a model prediction.")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=40.0)
        rooms = st.number_input("Rooms", min_value=1.0, max_value=20.0, value=5.0)

    with c2:
        bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=15.0, value=3.0)
        population = st.number_input("Population", min_value=0.0, value=1000.0)

    with c3:
        income = st.number_input("Income", min_value=0.0, value=60000.0, step=1000.0)
        persons_per_house = st.number_input(
            "Persons per house", min_value=0.0, value=2.5, step=0.1
        )

    if st.button("Predict House Classification", type="primary"):
        new_house = pd.DataFrame([{
            "age": age,
            "rooms": rooms,
            "bedrooms": bedrooms,
            "population": population,
            "income": income,
            "persons_per_house": persons_per_house
        }])

        prediction = int(model.predict(new_house)[0])
        probability = model.predict_proba(new_house)[0]

        label = "High" if prediction == 1 else "Low"

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", label)
        c2.metric("Probability Low", f"{probability[0]:.1%}")
        c3.metric("Probability High", f"{probability[1]:.1%}")

# -----------------------------
# Test Predictions
# -----------------------------
elif section == "Test Predictions":
    st.header("📋 Test Customer Predictions")

    prediction_df = pd.DataFrame({
        "customer_id": test["customer_id"],
        "prediction": test_pred,
        "prediction_label": np.where(test_pred == 1, "High", "Low"),
        "probability_high": test_prob
    })

    st.dataframe(
        prediction_df.style.format({"probability_high": "{:.1%}"}),
        use_container_width=True
    )

    # Required 0/1 output
    submission = prediction_df[["customer_id", "prediction"]].copy()

    csv = submission.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download customer predictions CSV",
        data=csv,
        file_name="house_price_predictions.csv",
        mime="text/csv",
        type="primary"
    )

    st.caption(
        "The downloadable prediction file contains only customer_id and the "
        "0/1 classification requested for the test customers."
    )
