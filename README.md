# AIML-Project9-Predictive-Maintenance-Industrial-Machines

## 1. Problem Statement

A manufacturing plant wants to predict machine failure **before it happens** using live sensor readings — air temperature, process temperature, rotational speed, torque, and tool wear — so that maintenance can be scheduled proactively instead of reactively.

## 2. Business Objective

Build and compare classifiers that predict:
1. **Whether** a machine will fail (binary pipeline), and
2. **Which type** of failure is most likely — Tool Wear (TWF), Heat Dissipation (HDF), Power Failure (PWF), Overstrain (OSF), or Random (RNF) — (multi-class pipeline)

so maintenance teams can act early and stock the right replacement parts on the first visit.

## 3. Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository / Kaggle (stephanmatzka)
10,000 operating cycles, 14 columns, ~3.4% overall failure rate.
`Dataset/ai4i2020.csv`

## 4. Data Cleaning Summary

| Check | Result |
|---|---|
| Missing values | 0 |
| Rows where failure flags disagree with `Machine failure` | 27 / 10,000 (~0.27%, negligible label noise — kept, documented) |
| Sensor ranges physically plausible | ✅ all pass |
| Identifier columns dropped | `UDI`, `Product ID` |

## 5. Feature Engineering

- `temp_diff` = Process temperature − Air temperature
- `power_proxy` = Torque × Rotational speed
- `Type` one-hot encoded (L / M / H)
- Numeric sensor features scaled (for Logistic Regression only)

## 6. Binary Pipeline — "Will It Fail?"

80/20 stratified split. Models trained with class-weight balancing for the ~3.4% failure rate.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8585 | 0.1772 | 0.8676 | 0.2943 | 0.9340 |
| Random Forest | 0.9875 | 0.9388 | 0.6765 | 0.7863 | 0.9704 |
| **XGBoost** | 0.9855 | 0.7671 | 0.8235 | **0.7943** | **0.9727** |

**GridSearchCV tuning** (on XGBoost, the best base model by F1): best params `{learning_rate: 0.1, max_depth: 6, n_estimators: 400}` → tuned F1 **0.8000** (up from 0.7943 untuned).

**Recommendation:** the tuned XGBoost model is deployed for the binary "will it fail" pipeline — it gives the best balance of recall (catching real failures) and precision (avoiding false alarms), backed by the highest ROC-AUC of the three candidates.

## 7. Multi-Class Extension — "Which Failure Type?"

Random Forest trained on a combined 6-class target (`No Failure` + the 5 failure flags).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| HDF | 0.9583 | 1.0000 | 0.9787 | 23 |
| No Failure | 0.9887 | 0.9995 | 0.9941 | 1930 |
| OSF | 1.0000 | 0.4375 | 0.6087 | 16 |
| PWF | 1.0000 | 1.0000 | 1.0000 | 18 |
| RNF | 0.0000 | 0.0000 | 0.0000 | 4 |
| TWF | 0.0000 | 0.0000 | 0.0000 | 9 |
| **Macro avg** | 0.6578 | 0.5728 | **0.5969** | 2000 |
| Weighted avg | 0.9821 | 0.9885 | 0.9844 | 2000 |

**Limitation:** RNF and TWF each have fewer than 50 examples in the entire 10,000-row dataset, so the multi-class model has almost no signal to learn these patterns — their recall is close to zero. This is reported honestly rather than hidden. In production these rare classes would be grouped into a "manual inspection required" bucket or backfilled with more labeled examples over time.

## 8. How the Two Models Work Together

The **binary model** runs continuously on live sensor streams and fires a maintenance alert the moment failure risk crosses the decision threshold. Once an alert fires, the **multi-class model** classifies which failure type is most likely, so the maintenance team brings the correct replacement part (cutting tool, cooling component, drive belt) on the first visit instead of guessing.

## 9. Key EDA Insights

1. Failed machines show noticeably higher torque and, in a large share of cases, higher tool wear than healthy machines.
2. A larger temperature differential (process − air) is associated with a higher failure likelihood — consistent with HDF requiring a small differential plus low RPM.
3. Overstrain Failure (OSF) is the most frequent failure type, followed by HDF and PWF; Random Failures (RNF) stay rare, matching how they were designed into the dataset.
4. Torque and tool wear correlate most strongly with `Machine failure` among the raw sensor features.

*(See `Images/` for all 8 EDA and evaluation visualizations, including the correlation heatmap and both confusion matrices.)*

## 10. Streamlit App — MAINT/OS Command Center

An interactive control-room style dashboard (`app/app.py`):
- **Live Risk Monitor** — adjust sensor sliders (or load a quick scenario preset), get a real-time failure-risk gauge plus the multi-class failure-type breakdown
- **Model Comparison** — both pipelines' metrics tables, confusion matrices, feature importance
- **EDA Insights** — all exploratory visualizations with narrative insights
- **Dataset** — raw data explorer

Run locally:
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## 11. Repository Structure

```
AIML-Project9-Predictive-Maintenance-Industrial-Machines/
├── Dataset/
│   └── ai4i2020.csv
├── Notebook/
│   └── Predictive_Maintenance_Industrial_Machines.ipynb
├── Images/
│   ├── 01_torque_toolwear_vs_failure.png
│   ├── 02_temp_diff_distribution.png
│   ├── 03_failure_type_distribution.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_power_proxy_vs_failure.png
│   ├── 06_binary_confusion_matrix.png
│   ├── 07_multiclass_confusion_matrix.png
│   └── 08_feature_importance_comparison.png
├── app/
│   ├── app.py                  # Streamlit command-center dashboard
│   ├── requirements.txt
│   ├── models/                 # saved binary + multi-class models, scaler
│   ├── Dataset/, Images/       # copies bundled for the app
│   └── *.json / *.csv          # summary + comparison artifacts
├── pipeline.py                  # full reproducible training pipeline
└── README.md
```

## 12. Common Mistakes Avoided

- Multi-class target built directly from the 5 flag columns (not from rows where `Machine failure == 0`), so "no failure" rows are labeled consistently.
- Multi-class evaluation reports **per-class precision/recall/F1**, not just overall accuracy — important given the severe class imbalance.
- `UDI` / `Product ID` dropped before training to prevent identifier leakage.
- Binary and multi-class results kept in two clearly separate comparison tables.
