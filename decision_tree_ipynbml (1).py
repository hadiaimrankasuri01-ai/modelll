# -*- coding: utf-8 -*-
"""Starbucks Nutrition & Decision Tree Classifier - Zero External Plotting Dependencies"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Starbucks Nutrition Classifier", layout="wide")
st.title("Starbucks Nutrition & Decision Tree Classifier")

# --- 1. Load the Dataset & Exploratory Data Analysis (EDA) ---
st.header("1. Dataset Overview & EDA")

uploaded_file = st.file_uploader("Upload 'starbucks-menu-nutrition-drinks.csv' (or leave empty to load from repo)", type=["csv"])
default_path = "starbucks-menu-nutrition-drinks.csv"

@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    if df.columns[0].startswith("Unnamed"):
        df.rename(columns={df.columns[0]: "Drink_Name"}, inplace=True)
    return df

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = load_data(default_path)
except Exception as e:
    st.warning(f"Could not find '{default_path}' in the repository ({e}). Please upload the CSV file above to proceed.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.subheader("First 5 Rows")
    st.dataframe(df.head())
with col2:
    st.subheader("Missing Values per Column")
    st.dataframe(df.isnull().sum().rename("Missing Count"))


# --- 2. Data Cleaning & Drink Categories Distribution ---
st.header("2. Drink Categories Distribution")

numeric_cols = ["Calories", "Fat (g)", "Carb. (g)", "Fiber (g)", "Protein", "Sodium"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("-", "", regex=False).str.strip(), errors="coerce")

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

def categorize_calories(cal):
    if cal < 150:
        return "Low Calorie (<150)"
    elif cal <= 300:
        return "Medium Calorie (150-300)"
    else:
        return "High Calorie (>300)"

df["Calorie_Category"] = df["Calories"].apply(categorize_calories)

# Built-in Streamlit chart (works natively without matplotlib/seaborn)
category_counts = df["Calorie_Category"].value_counts()
st.subheader("Distribution of Starbucks Drinks by Calorie Level")
st.bar_chart(category_counts)
st.dataframe(
    category_counts.reset_index().rename(columns={"index": "Calorie Category", "Calorie_Category": "Count", "count": "Count"})
)


# --- 3. Split the Data and Train a Decision Tree Classifier ---
st.header("3. Train Decision Tree Classifier")

features = ["Fat (g)", "Carb. (g)", "Fiber (g)", "Protein", "Sodium"]
X = df[features]
y = df["Calorie_Category"]

max_depth = st.slider("Select Decision Tree Max Depth", min_value=1, max_value=15, value=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
clf.fit(X_train, y_train)

st.success("Decision Tree model trained successfully!")


# --- 4. Evaluate Model Performance ---
st.header("4. Model Performance Evaluation")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.metric(label="Test Accuracy Score", value=f"{accuracy:.2%}")

st.subheader("Classification Report")
report_dict = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report_dict).transpose())


# --- 5. Compare Actual vs. Predicted Labels ---
st.header("5. Actual vs. Predicted Labels (10 Test Samples)")

test_indices = X_test.head(10).index
drink_col = "Drink_Name" if "Drink_Name" in df.columns else df.columns[0]
test_drinks = df.loc[test_indices, drink_col].values
actual_samples = y_test.head(10).values
predicted_samples = y_pred[:10]

comparison_df = pd.DataFrame({
    "Drink Name": test_drinks,
    "Actual Category": actual_samples,
    "Predicted Category": predicted_samples,
    "Match": actual_samples == predicted_samples
})

st.dataframe(comparison_df)
