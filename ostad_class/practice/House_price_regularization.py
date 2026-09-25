"""
House Price Predictor: Linear Regression vs Ridge vs Lasso
=============================================================
A hands-on project to see overfitting happen, and watch ridge/lasso fix it.

Run this file section by section (or paste into a Jupyter notebook cell by cell).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------------------
# STEP 1: Load the data
# -------------------------------------------------------------------
# California Housing: 8 features, ~20,000 rows, predicting median house value.
data = fetch_california_housing(as_frame=True)
df = data.frame
print("Shape:", df.shape)
print(df.head())
print("\nFeatures:", list(data.feature_names))

X = df[data.feature_names]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features — IMPORTANT for ridge/lasso, since the penalty treats all
# coefficients equally. If one feature is measured in thousands and another
# in single digits, the penalty punishes them unfairly unless scaled.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------
# STEP 2: Baseline — plain linear regression (least squares)
# -------------------------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

train_pred = lin_reg.predict(X_train_scaled)
test_pred = lin_reg.predict(X_test_scaled)

print("\n--- Plain Linear Regression ---")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))
print("Test R^2:", r2_score(y_test, test_pred))

# -------------------------------------------------------------------
# STEP 3: Force overfitting on purpose
# -------------------------------------------------------------------
# We add polynomial + interaction features. This creates lots of redundant,
# correlated columns — exactly the situation where plain least squares
# overfits (great training fit, worse test fit).
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print(f"\nExpanded from {X_train_scaled.shape[1]} to {X_train_poly.shape[1]} features")

overfit_model = LinearRegression()
overfit_model.fit(X_train_poly, y_train)

train_pred_of = overfit_model.predict(X_train_poly)
test_pred_of = overfit_model.predict(X_test_poly)

print("\n--- Overfit Linear Regression (polynomial features) ---")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, train_pred_of)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred_of)))
print("Test R^2:", r2_score(y_test, test_pred_of))
# Watch for: Train RMSE goes DOWN, but Test RMSE goes UP relative to baseline.
# That gap is overfitting, made visible.

# -------------------------------------------------------------------
# STEP 4: Fix it with Ridge (L2 penalty)
# -------------------------------------------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)

test_pred_ridge = ridge.predict(X_test_poly)
print("\n--- Ridge Regression (alpha=1.0) ---")
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred_ridge)))
print("Test R^2:", r2_score(y_test, test_pred_ridge))
print("Non-zero coefficients:", np.sum(ridge.coef_ != 0), "/", len(ridge.coef_))

# -------------------------------------------------------------------
# STEP 5: Fix it with Lasso (L1 penalty)
# -------------------------------------------------------------------
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train_poly, y_train)

test_pred_lasso = lasso.predict(X_test_poly)
print("\n--- Lasso Regression (alpha=0.01) ---")
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred_lasso)))
print("Test R^2:", r2_score(y_test, test_pred_lasso))
print("Non-zero coefficients:", np.sum(lasso.coef_ != 0), "/", len(lasso.coef_))
# Watch for: Lasso zeroes out many coefficients entirely — automatic
# feature selection. Ridge shrinks everything but rarely hits exactly zero.

# -------------------------------------------------------------------
# STEP 6: Side-by-side comparison table
# -------------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Linear (baseline)", "Linear (overfit)", "Ridge", "Lasso"],
    "Test RMSE": [
        np.sqrt(mean_squared_error(y_test, test_pred)),
        np.sqrt(mean_squared_error(y_test, test_pred_of)),
        np.sqrt(mean_squared_error(y_test, test_pred_ridge)),
        np.sqrt(mean_squared_error(y_test, test_pred_lasso)),
    ],
    "Test R^2": [
        r2_score(y_test, test_pred),
        r2_score(y_test, test_pred_of),
        r2_score(y_test, test_pred_ridge),
        r2_score(y_test, test_pred_lasso),
    ],
})
print("\n=== Comparison Table ===")
print(results.to_string(index=False))

# -------------------------------------------------------------------
# STEP 7: Regularization path — how alpha shrinks coefficients
# -------------------------------------------------------------------
alphas = np.logspace(-3, 2, 30)

ridge_coefs = []
lasso_coefs = []
for a in alphas:
    r = Ridge(alpha=a).fit(X_train_poly, y_train)
    l = Lasso(alpha=a, max_iter=10000).fit(X_train_poly, y_train)
    ridge_coefs.append(r.coef_)
    lasso_coefs.append(l.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(alphas, ridge_coefs)
axes[0].set_xscale("log")
axes[0].set_xlabel("alpha (log scale)")
axes[0].set_ylabel("coefficient value")
axes[0].set_title("Ridge: coefficients shrink smoothly")

axes[1].plot(alphas, lasso_coefs)
axes[1].set_xscale("log")
axes[1].set_xlabel("alpha (log scale)")
axes[1].set_ylabel("coefficient value")
axes[1].set_title("Lasso: coefficients hit zero (feature selection)")

plt.tight_layout()
plt.savefig("regularization_path.png", dpi=150)
print("\nSaved plot to regularization_path.png")
plt.close(fig)

# -------------------------------------------------------------------
# STEP 9: Stop guessing alpha — use cross-validation
# -------------------------------------------------------------------
# So far we picked alpha=1.0 for ridge and alpha=0.01 for lasso by guessing.
# In real ML engineering, you never guess a hyperparameter — you use
# cross-validation to test many alpha values and pick the one that
# generalizes best to unseen data.
from sklearn.linear_model import RidgeCV, LassoCV

alphas_to_try = np.logspace(-3, 2, 50)

ridge_cv = RidgeCV(alphas=alphas_to_try, cv=5)
ridge_cv.fit(X_train_poly, y_train)

lasso_cv = LassoCV(alphas=alphas_to_try, cv=5, max_iter=10000)
lasso_cv.fit(X_train_poly, y_train)

print("\n--- Cross-Validation Tuned Alphas ---")
print("Best Ridge alpha (5-fold CV):", ridge_cv.alpha_)
print("Best Lasso alpha (5-fold CV):", lasso_cv.alpha_)

test_pred_ridge_cv = ridge_cv.predict(X_test_poly)
test_pred_lasso_cv = lasso_cv.predict(X_test_poly)

print("\n--- Ridge (CV-tuned alpha) ---")
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred_ridge_cv)))
print("Test R^2:", r2_score(y_test, test_pred_ridge_cv))

print("\n--- Lasso (CV-tuned alpha) ---")
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred_lasso_cv)))
print("Test R^2:", r2_score(y_test, test_pred_lasso_cv))
print("Non-zero coefficients:", np.sum(lasso_cv.coef_ != 0), "/", len(lasso_cv.coef_))

# -------------------------------------------------------------------
# STEP 10: Updated comparison table (with CV-tuned models)
# -------------------------------------------------------------------
results_final = pd.DataFrame({
    "Model": [
        "Linear (baseline)", "Linear (overfit)",
        "Ridge (guessed)", "Lasso (guessed)",
        "Ridge (CV-tuned)", "Lasso (CV-tuned)",
    ],
    "Alpha used": ["-", "-", 1.0, 0.01, ridge_cv.alpha_, lasso_cv.alpha_],
    "Test RMSE": [
        np.sqrt(mean_squared_error(y_test, test_pred)),
        np.sqrt(mean_squared_error(y_test, test_pred_of)),
        np.sqrt(mean_squared_error(y_test, test_pred_ridge)),
        np.sqrt(mean_squared_error(y_test, test_pred_lasso)),
        np.sqrt(mean_squared_error(y_test, test_pred_ridge_cv)),
        np.sqrt(mean_squared_error(y_test, test_pred_lasso_cv)),
    ],
    "Test R^2": [
        r2_score(y_test, test_pred),
        r2_score(y_test, test_pred_of),
        r2_score(y_test, test_pred_ridge),
        r2_score(y_test, test_pred_lasso),
        r2_score(y_test, test_pred_ridge_cv),
        r2_score(y_test, test_pred_lasso_cv),
    ],
})
print("\n=== Final Comparison Table (with CV-tuned models) ===")
print(results_final.to_string(index=False))

# -------------------------------------------------------------------
# STEP 11: Your takeaways (fill this in after running the above)
# -------------------------------------------------------------------
"""
Questions to answer in your write-up:
1. How much did test RMSE change between baseline linear and overfit linear?
2. Did ridge or lasso get closer to (or beat) the baseline's test RMSE?
3. How many features did lasso zero out? Which ones (look at poly.get_feature_names_out())?
4. Looking at the regularization path plots — at what alpha do lasso coefficients
   start disappearing? Is that alpha too aggressive (hurts test RMSE) or just right?
5. Based on YOUR numbers: would you use ridge or lasso for this dataset, and why?
6. Did the CV-tuned alpha beat your guessed alpha? By how much?
7. Was the best CV alpha closer to your original guess or far off?

What's next after this project:
1. Elastic Net — combines L1 + L2 penalties in one model
2. Cross-validation in depth — why k-fold CV works, how to choose k
3. Logistic Regression — same ideas, but for classification
4. Feature engineering — meaningful features instead of blind polynomial expansion
"""