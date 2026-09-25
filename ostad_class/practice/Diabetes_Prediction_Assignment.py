"""
Pima Indians Diabetes Prediction
=================================
Objective: Train and compare ML classification models to predict whether a
patient has diabetes, using the Pima Indians Diabetes Dataset.

Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Run in VS Code either:
  - as a plain script:            python diabetes_prediction.py
  - as a Jupyter-style notebook:  open this file, and VS Code's Python
    extension will let you run each "# %%" cell individually (click
    "Run Cell" above each block), showing plots inline just like Colab.

Requirements (install once):
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
"""

# %% [markdown]
# ## 0. Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
import joblib

sns.set_style("whitegrid")

# %% [markdown]
# ## 1. Data Loading & Exploration
#
# Option A (recommended): download `diabetes.csv` from Kaggle and place it
# in the same folder as this script, then it will be loaded automatically.
#
# Option B (fallback): if `diabetes.csv` isn't found locally, the script
# pulls a public mirror of the same dataset with identical columns.

# %%
import os

LOCAL_PATH = "diabetes.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"

if os.path.exists(LOCAL_PATH):
    df = pd.read_csv(LOCAL_PATH)
    print(f"Loaded dataset from local file: {LOCAL_PATH}")
else:
    df = pd.read_csv(FALLBACK_URL)
    print("Local diabetes.csv not found — loaded dataset from fallback URL.")

# %%
# Show first few rows
print(df.head())

# %%
# Check dataset shape
print("Shape of dataset (rows, columns):", df.shape)

# %%
# Column info and data types
df.info()

# %%
# Check for missing values (nulls)
print("Missing values per column:")
print(df.isnull().sum())

# %%
# Note: In this dataset, missing data is often encoded as 0 in columns
# where 0 is medically impossible (Glucose, BloodPressure, SkinThickness,
# Insulin, BMI). Let's check how many zero values exist in these columns.
cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_invalid_zero:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zero values ({zero_count/len(df)*100:.1f}%)")

# %%
# Basic statistics
print(df.describe())

# %%
# Target class balance
print(df["Outcome"].value_counts())
sns.countplot(x="Outcome", data=df)
plt.title("Class Distribution (0 = No Diabetes, 1 = Diabetes)")
plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Data Preprocessing

# %%
# Replace invalid zero values with NaN, then impute with column median
df_clean = df.copy()
for col in cols_with_invalid_zero:
    df_clean[col] = df_clean[col].replace(0, np.nan)
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("Missing/invalid values handled. Remaining nulls:")
print(df_clean.isnull().sum())

# %%
# Separate features (X) and target (y)
X = df_clean.drop("Outcome", axis=1)
y = df_clean["Outcome"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# %%
# Train/Test Split - 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# %%
# Feature scaling (important for Logistic Regression and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## 3. Model Training
#
# We train three classification models: Logistic Regression, Random
# Forest, and Support Vector Machine (SVM).

# %%
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

trained_models = {}

for name, model in models.items():
    # Tree-based models don't strictly need scaling, but scaled data works fine too
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} trained.")

# %% [markdown]
# ## 4. Model Evaluation

# %%
results = []
roc_data = {}

for name, model in trained_models.items():
    X_test_use = X_test_scaled if name in ["Logistic Regression", "SVM"] else X_test

    y_pred = model.predict(X_test_use)
    y_proba = model.predict_proba(X_test_use)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, roc_auc)

    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print(results_df)

# %%
# Bar chart comparing metrics across models
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
results_df.set_index("Model")[metrics_to_plot].plot(kind="bar", figsize=(10, 6))
plt.title("Model Comparison Across Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ROC Curve comparison
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.savefig("roc_curve_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Best Model Selection & Saving

# %%
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print(f"Best performing model (by ROC-AUC): {best_model_name}")
print(results_df.iloc[0])

# %%
# Save the best model using joblib
joblib.dump(best_model, "diabetes_model.pkl")

# Also save the scaler, since Logistic Regression / SVM need scaled inputs at prediction time
joblib.dump(scaler, "scaler.pkl")

print("Model saved as diabetes_model.pkl")
print("Scaler saved as scaler.pkl")

# %% [markdown]
# ## Summary
#
# - Loaded and explored the Pima Indians Diabetes dataset (768 rows, 9 columns).
# - Handled biologically invalid zero values in Glucose, BloodPressure,
#   SkinThickness, Insulin, and BMI by imputing medians.
# - Split data into 80% training / 20% testing, with feature scaling
#   applied for scale-sensitive models.
# - Trained three models: Logistic Regression, Random Forest, and SVM.
# - Evaluated each model with Accuracy, Precision, Recall, F1 Score, and
#   ROC-AUC, and plotted ROC curves for comparison.
# - Selected the best model by ROC-AUC and saved it (plus the scaler) with
#   joblib for later use.