# %% [markdown]
# # Module 7 — Pandas Data Analysis Using Heart Disease Dataset
#
# Dataset: Heart Disease Dataset (Kaggle — johnsmith88/heart-disease-dataset)
#
# ## Setup (before running)
# 1. Install the Python extension and Jupyter extension in VS Code.
# 2. Download `heart.csv` from the Kaggle link and place it in the same folder as this script
#    (or update `file_path` below).
# 3. Run cells one by one with "Run Cell" (appears above each `# %%` block),
#    or run the whole file with "Run All".

# %%
import pandas as pd
import numpy as np

# Update this path to wherever heart.csv is located on your machine
file_path = "C:\\Users\\Faisal\\Downloads\\heart.csv"   # e.g. "C:/Users/YourName/Downloads/heart.csv"

df = pd.read_csv(file_path)
print("Dataset loaded successfully!")

# %% [markdown]
# ## Question 1: Dataset Loading and Initial Exploration

# %%
# 1. First five rows
print(df.head())

# %%
# 2. Last five rows
print(df.tail())

# %%
# 3. Shape of the dataset
print("Shape of dataset (rows, columns):", df.shape)

# %% [markdown]
# ## Question 2: Dataset Structure and Missing Value Analysis

# %%
# 1. Column names
print("Column names:")
print(df.columns.tolist())

# %%
# 2. Dataset info (data types and non-null counts)
df.info()

# %%
# 3. Missing values in each column
print("Missing values per column:")
print(df.isnull().sum())

# %% [markdown]
# ## Question 3: Categorical Data Analysis

# %%
# 1. Count of patients with heart disease (target = 1) and without (target = 0)
target_counts = df['target'].value_counts()
print("Heart disease counts (1 = disease, 0 = no disease):")
print(target_counts)

# %%
# 2. Count of male (sex = 1) and female (sex = 0) patients
sex_counts = df['sex'].value_counts()
print("Sex counts (1 = Male, 0 = Female):")
print(sex_counts)

# %% [markdown]
# ## Question 4: Data Selection Using iloc

# %%
# 1. First 10 rows
first_10_rows = df.iloc[0:10]
print(first_10_rows)

# %%
# 2. Extract age, sex, and cholesterol columns using iloc
# (column positions: age=0, sex=1, chol=4 in the standard heart.csv layout —
#  verify with df.columns.tolist() from Question 2 if your file order differs)
age_sex_chol = df.iloc[:, [0, 1, 4]]
print(age_sex_chol.head())

# %%
# 3. Rows from index 20 to 30 and columns from index 0 to 4
subset = df.iloc[20:31, 0:5]
print(subset)

# %% [markdown]
# ## Question 5: Data Filtering Using Conditions

# %%
# 1. Patients older than 50 years
older_than_50 = df[df['age'] > 50]
print("Number of patients older than 50:", len(older_than_50))
print(older_than_50.head())

# %%
# 2. Patients who have heart disease AND cholesterol > 240
disease_high_chol = df[(df['target'] == 1) & (df['chol'] > 240)]
print(disease_high_chol.head())

# %%
# 3. Count of patients satisfying both conditions
count_both = disease_high_chol.shape[0]
print("Number of patients with heart disease AND cholesterol > 240:", count_both)