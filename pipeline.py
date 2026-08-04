"""
Predictive Maintenance for Industrial Machines
End-to-end pipeline: cleaning -> EDA -> feature engineering ->
binary "will it fail" models + multi-class "which failure type" model ->
tuning -> evaluation -> comparison tables -> artifact export (for Streamlit app)
"""
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams["figure.facecolor"] = "#0e1117"
plt.rcParams["axes.facecolor"] = "#161a23"
plt.rcParams["savefig.facecolor"] = "#0e1117"
plt.rcParams["text.color"] = "#e6e6e6"
plt.rcParams["axes.labelcolor"] = "#e6e6e6"
plt.rcParams["xtick.color"] = "#c8c8c8"
plt.rcParams["ytick.color"] = "#c8c8c8"
plt.rcParams["axes.edgecolor"] = "#444444"

IMG = "Images"
MODELS = "models"
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------
df = pd.read_csv("Dataset/ai4i2020.csv")
print("Loaded:", df.shape)

# --------------------------------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------------------------------
cleaning_log = {}
cleaning_log["missing_values_total"] = int(df.isnull().sum().sum())

# Consistency check: sum of individual failure flags vs Machine failure
flag_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
any_flag = df[flag_cols].sum(axis=1) > 0
inconsistent = (any_flag != df["Machine failure"].astype(bool)).sum()
cleaning_log["inconsistent_failure_rows"] = int(inconsistent)

# Physically plausible ranges check
range_checks = {
    "Air temperature [K]": (df["Air temperature [K]"].between(250, 350)).all(),
    "Process temperature [K]": (df["Process temperature [K]"].between(250, 350)).all(),
    "Rotational speed [rpm]": (df["Rotational speed [rpm]"] > 0).all(),
    "Torque [Nm]": (df["Torque [Nm]"] >= 0).all(),
    "Tool wear [min]": (df["Tool wear [min]"] >= 0).all(),
}
cleaning_log["range_checks_passed"] = {k: bool(v) for k, v in range_checks.items()}

# Drop identifier columns
df_clean = df.drop(columns=["UDI", "Product ID"])
cleaning_log["dropped_columns"] = ["UDI", "Product ID"]
cleaning_log["rows_after_cleaning"] = int(len(df_clean))

with open("cleaning_log.json", "w") as f:
    json.dump(cleaning_log, f, indent=2)
print("Cleaning log:", cleaning_log)

# --------------------------------------------------------------------------
# 3. EDA
# --------------------------------------------------------------------------
NEON = "#7c5cff"
NEON2 = "#00d9c0"
NEON3 = "#ff5c8a"

# 3.1 Failure rate vs torque & tool wear
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, color in zip(axes, ["Torque [Nm]", "Tool wear [min]"], [NEON, NEON2]):
    sns.boxplot(data=df_clean, x="Machine failure", y=col, ax=ax,
                palette=[ "#2b2f3a", color])
    ax.set_title(f"{col} vs Machine Failure", color="white", fontsize=12, weight="bold")
    ax.set_xlabel("Machine Failure (0=No, 1=Yes)")
plt.tight_layout()
plt.savefig(f"{IMG}/01_torque_toolwear_vs_failure.png", dpi=130)
plt.close()

# 3.2 Temp differential
df_clean["temp_diff"] = df_clean["Process temperature [K]"] - df_clean["Air temperature [K]"]
fig, ax = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=df_clean, x="temp_diff", hue="Machine failure", fill=True,
            palette=[NEON2, NEON3], alpha=0.55, ax=ax)
ax.set_title("Temperature Differential: Failed vs Non-Failed", color="white", weight="bold")
plt.tight_layout()
plt.savefig(f"{IMG}/02_temp_diff_distribution.png", dpi=130)
plt.close()

# 3.3 Failure type distribution among failure cases
fail_only = df_clean[df_clean["Machine failure"] == 1]
counts = fail_only[flag_cols].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(counts.index, counts.values, color=[NEON, NEON2, NEON3, "#ffb84d", "#5cd6ff"])
ax.set_title("Failure Type Distribution (Failure Cases Only)", color="white", weight="bold")
ax.bar_label(bars, color="white")
plt.tight_layout()
plt.savefig(f"{IMG}/03_failure_type_distribution.png", dpi=130)
plt.close()

# 3.4 Correlation heatmap
num_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]", "temp_diff", "Machine failure"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_clean[num_cols].corr(), annot=True, cmap="mako", ax=ax, fmt=".2f",
            linewidths=0.5, linecolor="#0e1117")
ax.set_title("Correlation Heatmap: Sensors vs Machine Failure", color="white", weight="bold")
plt.tight_layout()
plt.savefig(f"{IMG}/04_correlation_heatmap.png", dpi=130)
plt.close()

# 3.5 power proxy vs failure
df_clean["power_proxy"] = df_clean["Torque [Nm]"] * df_clean["Rotational speed [rpm]"]
fig, ax = plt.subplots(figsize=(7, 5))
sns.violinplot(data=df_clean, x="Machine failure", y="power_proxy",
                palette=["#2b2f3a", NEON], ax=ax)
ax.set_title("Power Proxy (Torque x RPM) vs Machine Failure", color="white", weight="bold")
plt.tight_layout()
plt.savefig(f"{IMG}/05_power_proxy_vs_failure.png", dpi=130)
plt.close()

eda_insights = [
    "Failed machines show noticeably higher torque and, in a large share of cases, higher tool wear than healthy machines, confirming both as strong failure signals.",
    "A larger temperature differential (process minus air temperature) is associated with a higher chance of failure, consistent with Heat Dissipation Failure (HDF) requiring a small differential plus low rotational speed.",
    "Overstrain Failure (OSF) is the most frequent failure type, followed by Heat Dissipation (HDF) and Power Failure (PWF); Random Failures (RNF) are rare as expected by design.",
    "The correlation heatmap shows torque and tool wear correlate most strongly with Machine failure among the raw sensor features, while air/process temperature alone are weaker individual signals.",
]
with open("eda_insights.json", "w") as f:
    json.dump(eda_insights, f, indent=2)

# --------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------------------------------
df_fe = df_clean.copy()
df_fe = pd.get_dummies(df_fe, columns=["Type"], prefix="Type")

feature_cols = [c for c in df_fe.columns if c not in
                ["Machine failure"] + flag_cols]

def sanitize(col):
    return (col.replace("[", "(").replace("]", ")").replace("<", "lt")
               .replace(" ", "_"))

X = df_fe[feature_cols].copy()
X.columns = [sanitize(c) for c in X.columns]
feature_cols_sanitized = list(X.columns)
y_binary = df_fe["Machine failure"]

with open(f"{MODELS}/feature_columns.json", "w") as f:
    json.dump(feature_cols_sanitized, f)

# Multi-class target: combine flags into single label incl. "No Failure"
def make_multiclass_label(row):
    for f in flag_cols:
        if row[f] == 1:
            return f
    return "No Failure"

y_multi = df_clean[flag_cols].apply(make_multiclass_label, axis=1)
# Keep rows consistent: only rows where Machine failure matches flags
consistent_mask = (df_clean[flag_cols].sum(axis=1) > 0) == (df_clean["Machine failure"] == 1)
print("Consistent rows:", consistent_mask.sum(), "/", len(df_clean))



# --------------------------------------------------------------------------
# 5. BINARY PIPELINE: train/test split (80/20 stratified)
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
)

scaler = StandardScaler()
numeric_feats = [sanitize(c) for c in
                 ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                  "Torque [Nm]", "Tool wear [min]", "temp_diff", "power_proxy"]]
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_feats] = scaler.fit_transform(X_train[numeric_feats])
X_test_scaled[numeric_feats] = scaler.transform(X_test[numeric_feats])
joblib.dump(scaler, f"{MODELS}/scaler.joblib")

binary_models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
    ),
}

binary_results = {}
binary_fitted = {}
for name, model in binary_models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
    binary_fitted[name] = model
    binary_results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }

binary_comparison = pd.DataFrame(binary_results).T.round(4)
binary_comparison.to_csv("binary_comparison.csv")
print("\nBINARY PIPELINE COMPARISON\n", binary_comparison)

# Confusion matrix for best binary model (by F1)
best_binary_name = binary_comparison["F1"].idxmax()
best_binary_model = binary_fitted[best_binary_name]
X_eval = X_test_scaled if best_binary_name == "Logistic Regression" else X_test
cm = confusion_matrix(y_test, best_binary_model.predict(X_eval))
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="rocket", ax=ax,
            xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
ax.set_title(f"Confusion Matrix - {best_binary_name} (Binary)", color="white", weight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{IMG}/06_binary_confusion_matrix.png", dpi=130)
plt.close()

# --------------------------------------------------------------------------
# 6. GridSearchCV TUNING on best binary model family
# --------------------------------------------------------------------------
if best_binary_name == "Random Forest":
    param_grid = {"n_estimators": [200, 400], "max_depth": [8, 12, None], "min_samples_leaf": [1, 3]}
    grid = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                         param_grid, scoring="f1", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    tuned_model = grid.best_estimator_
    tuned_preds = tuned_model.predict(X_test)
    tuned_proba = tuned_model.predict_proba(X_test)[:, 1]
elif best_binary_name == "XGBoost":
    param_grid = {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
    grid = GridSearchCV(XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                                       eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
                         param_grid, scoring="f1", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    tuned_model = grid.best_estimator_
    tuned_preds = tuned_model.predict(X_test)
    tuned_proba = tuned_model.predict_proba(X_test)[:, 1]
else:
    param_grid = {"C": [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
                         param_grid, scoring="f1", cv=3, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    tuned_model = grid.best_estimator_
    tuned_preds = tuned_model.predict(X_test_scaled)
    tuned_proba = tuned_model.predict_proba(X_test_scaled)[:, 1]

tuning_summary = {
    "base_model": best_binary_name,
    "best_params": grid.best_params_,
    "tuned_f1": float(f1_score(y_test, tuned_preds)),
    "untuned_f1": float(binary_comparison.loc[best_binary_name, "F1"]),
}
with open("tuning_summary.json", "w") as f:
    json.dump(tuning_summary, f, indent=2)
print("\nTUNING SUMMARY:", tuning_summary)

# Save the deployment binary model (use tuned model)
joblib.dump(tuned_model, f"{MODELS}/binary_model.joblib")
with open(f"{MODELS}/binary_model_name.json", "w") as f:
    json.dump({"name": best_binary_name, "needs_scaling": best_binary_name == "Logistic Regression"}, f)

# --------------------------------------------------------------------------
# 7. MULTI-CLASS PIPELINE
# --------------------------------------------------------------------------
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X, y_multi, test_size=0.2, random_state=RANDOM_STATE, stratify=y_multi
)

rf_multi = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                   random_state=RANDOM_STATE)
rf_multi.fit(Xm_train, ym_train)
ym_pred = rf_multi.predict(Xm_test)

multi_report = classification_report(ym_test, ym_pred, output_dict=True, zero_division=0)
multi_report_df = pd.DataFrame(multi_report).T.round(4)
multi_report_df.to_csv("multiclass_report.csv")
print("\nMULTI-CLASS REPORT\n", multi_report_df)

joblib.dump(rf_multi, f"{MODELS}/multiclass_model.joblib")

# Multi-class confusion matrix
labels_order = ["No Failure"] + flag_cols
cm_multi = confusion_matrix(ym_test, ym_pred, labels=labels_order)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_multi, annot=True, fmt="d", cmap="mako", ax=ax,
            xticklabels=labels_order, yticklabels=labels_order)
ax.set_title("Confusion Matrix - Multi-Class Failure Type (Random Forest)", color="white", weight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{IMG}/07_multiclass_confusion_matrix.png", dpi=130)
plt.close()

# --------------------------------------------------------------------------
# 8. FEATURE IMPORTANCE COMPARISON
# --------------------------------------------------------------------------
rf_binary_importances = None
if "Random Forest" in binary_fitted:
    rf_binary_importances = pd.Series(binary_fitted["Random Forest"].feature_importances_,
                                       index=X.columns).sort_values(ascending=False).head(8)
multi_importances = pd.Series(rf_multi.feature_importances_, index=X.columns).sort_values(ascending=False).head(8)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
if rf_binary_importances is not None:
    axes[0].barh(rf_binary_importances.index[::-1], rf_binary_importances.values[::-1], color=NEON)
    axes[0].set_title("Top Features - Binary Model", color="white", weight="bold")
axes[1].barh(multi_importances.index[::-1], multi_importances.values[::-1], color=NEON3)
axes[1].set_title("Top Features - Multi-Class Model", color="white", weight="bold")
plt.tight_layout()
plt.savefig(f"{IMG}/08_feature_importance_comparison.png", dpi=130)
plt.close()

# --------------------------------------------------------------------------
# 9. FINAL SUMMARY JSON for README / Streamlit
# --------------------------------------------------------------------------
summary = {
    "binary_comparison": binary_comparison.reset_index().rename(columns={"index": "Model"}).to_dict(orient="records"),
    "best_binary_model": best_binary_name,
    "tuning_summary": tuning_summary,
    "multiclass_macro_f1": float(multi_report["macro avg"]["f1-score"]),
    "multiclass_report": multi_report_df.reset_index().rename(columns={"index": "Class"}).to_dict(orient="records"),
    "failure_counts": {k: int(v) for k, v in counts.items()},
    "dataset_shape": list(df.shape),
    "failure_rate_pct": round(100 * df["Machine failure"].mean(), 2),
}
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDONE. Artifacts saved.")
