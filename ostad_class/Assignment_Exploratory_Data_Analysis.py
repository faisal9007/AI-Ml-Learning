

import os
import sys
import subprocess
import urllib.request
try:
    import pandas as pd  # type: ignore[import]
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd  # type: ignore[import]
try:
    import matplotlib.pyplot as plt  # type: ignore[import]
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt  # type: ignore[import]
try:
    import seaborn as sns  # type: ignore[import]
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns  # type: ignore[import]

sns.set_style('whitegrid')
pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 20)

# ---------------------------------------------------------------
# Cell 0 — Setup: download the dataset
# ---------------------------------------------------------------
if not os.path.exists('titanic.csv'):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv",
        "titanic.csv"
    )

# ---------------------------------------------------------------
# Task 1 — Data Loading and Initial Inspection
# ---------------------------------------------------------------
df = pd.read_csv('titanic.csv')

print("="*70)
print("TASK 1: DATA LOADING AND INITIAL INSPECTION")
print("="*70)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types / info:")
df.info()

print("\nDescriptive statistics (numerical columns):")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

# ---------------------------------------------------------------
# Task 2 — Handling Missing Values
# ---------------------------------------------------------------
print("\n" + "="*70)
print("TASK 2: HANDLING MISSING VALUES")
print("="*70)

# Cabin
cabin_missing_pct = df['Cabin'].isnull().mean() * 100
print(f"\nCabin missing: {cabin_missing_pct:.2f}%")
print("Decision: drop Cabin — 77% missing is too sparse to impute reliably.")

# Embarked
embarked_mode = df['Embarked'].mode()[0]
print(f"\nEmbarked mode (most frequent port): {embarked_mode}")
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

# Age
age_median = df['Age'].median()
print(f"\nAge median: {age_median}")
df['Age'] = df['Age'].fillna(age_median)

df_clean = df.drop(columns=['Cabin'])
df_clean.to_csv('titanic_clean.csv', index=False)

print("\nRemaining missing values after cleaning:")
print(df_clean.isnull().sum())
print("\nSaved cleaned dataset -> titanic_clean.csv")

# ---------------------------------------------------------------
# Task 3 — Univariate Analysis
# ---------------------------------------------------------------
print("\n" + "="*70)
print("TASK 3: UNIVARIATE ANALYSIS")
print("="*70)

# Survival rate
survival_rate = df_clean['Survived'].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.2f}%")

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df_clean, hue='Survived',
              palette=['#c0392b', '#27ae60'], legend=False)
plt.title('Survival Count')
plt.xticks([0, 1], ['Did not survive', 'Survived'])
plt.tight_layout()
plt.savefig('1_survival_count.png', dpi=120)
plt.show()

# Pclass
pclass_counts = df_clean['Pclass'].value_counts().sort_index()
print("\nPclass counts:\n", pclass_counts)
print(f"Class with most passengers: {pclass_counts.idxmax()} ({pclass_counts.max()} passengers)")

plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', data=df_clean, hue='Pclass', palette='Blues_d', legend=False)
plt.title('Passenger Class Distribution')
plt.tight_layout()
plt.savefig('2_pclass_count.png', dpi=120)
plt.show()

# Age distribution
plt.figure(figsize=(6, 4))
sns.histplot(df_clean['Age'], bins=30, kde=True, color='#2980b9')
plt.title('Age Distribution (missing values imputed with median)')
plt.tight_layout()
plt.savefig('3_age_hist.png', dpi=120)
plt.show()

# ---------------------------------------------------------------
# Task 4 — Bivariate and Multivariate Analysis
# ---------------------------------------------------------------
print("\n" + "="*70)
print("TASK 4: BIVARIATE / MULTIVARIATE ANALYSIS")
print("="*70)

# Survival by Sex
sex_cross = pd.crosstab(df_clean['Sex'], df_clean['Survived'])
sex_cross.columns = ['Did not survive', 'Survived']
print("\nSurvival counts by Sex:\n", sex_cross)

sex_rate = df_clean.groupby('Sex')['Survived'].mean() * 100
print("\nSurvival rate by Sex (%):\n", sex_rate.round(2))

sex_cross.plot(kind='bar', stacked=True, figsize=(6, 4), color=['#c0392b', '#27ae60'])
plt.title('Survival Count by Sex (stacked)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('4_survival_by_sex.png', dpi=120)
plt.show()

# Survival by Pclass
pclass_rate = df_clean.groupby('Pclass')['Survived'].mean() * 100
print("\nSurvival rate by Pclass (%):\n", pclass_rate.round(2))

plt.figure(figsize=(6, 4))
sns.barplot(x=pclass_rate.index, y=pclass_rate.values,
            hue=pclass_rate.index, palette='Greens_d', legend=False)
plt.ylabel('Survival rate (%)')
plt.xlabel('Pclass')
plt.title('Survival Rate by Passenger Class')
plt.tight_layout()
plt.savefig('5_survival_by_pclass.png', dpi=120)
plt.show()

# Survival by Age
plt.figure(figsize=(7, 4.5))
sns.kdeplot(df_clean[df_clean['Survived'] == 0]['Age'], label='Did not survive',
            fill=True, alpha=0.4, color='#c0392b')
sns.kdeplot(df_clean[df_clean['Survived'] == 1]['Age'], label='Survived',
            fill=True, alpha=0.4, color='#27ae60')
plt.title('Age Distribution: Survivors vs Non-Survivors')
plt.xlabel('Age')
plt.legend()
plt.tight_layout()
plt.savefig('6_age_survival_kde.png', dpi=120)
plt.show()

child_rate = df_clean[df_clean['Age'] <= 12]['Survived'].mean() * 100
elderly_rate = df_clean[df_clean['Age'] >= 60]['Survived'].mean() * 100
print(f"\nSurvival rate, children (<=12): {child_rate:.2f}%")
print(f"Survival rate, elderly (>=60): {elderly_rate:.2f}%")

# Survival by Embarked
embarked_rate = df_clean.groupby('Embarked')['Survived'].mean() * 100
print("\nSurvival rate by Embarked (%):\n", embarked_rate.round(2))

plt.figure(figsize=(6, 4))
sns.barplot(x=embarked_rate.index, y=embarked_rate.values,
            hue=embarked_rate.index, palette='Purples_d', legend=False)
plt.ylabel('Survival rate (%)')
plt.xlabel('Embarked')
plt.title('Survival Rate by Port of Embarkation')
plt.tight_layout()
plt.savefig('7_survival_by_embarked.png', dpi=120)
plt.show()

# ---------------------------------------------------------------
# Task 5 — Conclusion and Insights
# ---------------------------------------------------------------
print("\n" + "="*70)
print("TASK 5: CONCLUSION AND INSIGHTS")
print("="*70)

conclusion = """
Sex was the strongest predictor of survival on the Titanic: women survived
at 74.2% versus only 18.9% for men, a difference of over 55 percentage
points, reflecting the "women and children first" evacuation policy.
Passenger class (Pclass) was the second strongest predictor, with survival
falling steadily from 63.0% in 1st class to 47.3% in 2nd class to 24.2% in
3rd class, most plausibly driven by cabin proximity to lifeboats and
boarding priority for wealthier passengers. Age mattered at the margins:
children (<=12) survived above average (58.0%), consistent with the same
rescue priority, while elderly passengers (>=60) fared worse than average
(26.9%). Port of Embarkation showed a survival gap too (Cherbourg highest
at 55.4%), but this is largely a proxy effect of class composition rather
than an independent causal factor. In short, being female and traveling
in a higher ticket class were the two dominant survival advantages on the
Titanic, with age acting as a secondary, reinforcing factor.
"""
print(conclusion)

print("\nAll plots saved as PNG files in the current working directory.")