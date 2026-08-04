"""
Predictive Maintenance Command Center
A control-room style Streamlit dashboard for the AI4I 2020 predictive
maintenance binary + multi-class pipelines.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MAINT/OS — Predictive Maintenance Command Center",
    page_icon="⛭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    binary_model = joblib.load("models/binary_model.joblib")
    multiclass_model = joblib.load("models/multiclass_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    with open("models/feature_columns.json") as f:
        feature_cols = json.load(f)
    with open("models/binary_model_name.json") as f:
        binary_meta = json.load(f)
    with open("summary.json") as f:
        summary = json.load(f)
    with open("eda_insights.json") as f:
        insights = json.load(f)
    binary_cmp = pd.read_csv("binary_comparison.csv").rename(columns={"Unnamed: 0": "Model"})
    multi_cmp = pd.read_csv("multiclass_report.csv").rename(columns={"Unnamed: 0": "Class"})
    return binary_model, multiclass_model, scaler, feature_cols, binary_meta, summary, insights, binary_cmp, multi_cmp

(binary_model, multiclass_model, scaler, feature_cols, binary_meta,
 summary, insights, binary_cmp, multi_cmp) = load_artifacts()

NUMERIC_RAW = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
               "Torque [Nm]", "Tool wear [min]"]

def sanitize(c):
    return c.replace("[", "(").replace("]", ")").replace("<", "lt").replace(" ", "_")

# --------------------------------------------------------------------------
# CONTROL-ROOM THEME — CSS
# --------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg: #0a0d13;
  --panel: #111621;
  --panel-2: #161c2a;
  --line: #232b3d;
  --amber: #ff9340;
  --cyan: #2be2c8;
  --red: #ff4d5e;
  --text: #e8ecf3;
  --muted: #7c8698;
}
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp{ background:
  radial-gradient(circle at 15% 0%, rgba(43,226,200,0.05), transparent 40%),
  radial-gradient(circle at 85% 100%, rgba(255,147,64,0.06), transparent 45%),
  var(--bg);
}
h1, h2, h3, .display-font { font-family: 'Rajdhani', sans-serif !important; letter-spacing: 0.02em; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* Top bar */
.topbar{
  display:flex; align-items:center; justify-content:space-between;
  border:1px solid var(--line); border-radius:10px;
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  padding: 14px 22px; margin-bottom: 18px;
}
.topbar .title{ font-family:'Rajdhani',sans-serif; font-weight:700; font-size:28px; color:var(--text); letter-spacing:0.03em; }
.topbar .subtitle{ font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--muted); margin-top:-2px;}
.status-pill{
  display:flex; align-items:center; gap:8px; font-family:'JetBrains Mono',monospace;
  font-size:12px; color: var(--cyan); border:1px solid rgba(43,226,200,0.35);
  background: rgba(43,226,200,0.06); padding:6px 12px; border-radius:20px;
}
.dot{ width:8px; height:8px; border-radius:50%; background:var(--cyan); box-shadow:0 0 8px var(--cyan);
  animation: pulse 1.6s infinite ease-in-out; }
@keyframes pulse{ 0%{opacity:1;} 50%{opacity:0.35;} 100%{opacity:1;} }


</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# TOP BAR
# --------------------------------------------------------------------------
st.markdown(f"""
<div class="topbar">
  <div>
    <div class="title">⛭ MAINT/OS — Predictive Maintenance Command Center</div>
    <div class="subtitle">AI4I-2020 · {summary['dataset_shape'][0]:,} logged operating cycles · base failure rate {summary['failure_rate_pct']}%</div>
  </div>
  <div class="status-pill"><span class="dot"></span> LIVE MODEL ONLINE — {binary_meta['name'].upper()} (TUNED)</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SIDEBAR — SENSOR CONTROLS
# --------------------------------------------------------------------------
st.sidebar.markdown("### ⛭ SENSOR INPUT PANEL")
st.sidebar.markdown("<span class='mono' style='color:#7c8698;font-size:11px;'>Adjust live readings to simulate a machine cycle</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

machine_type = st.sidebar.selectbox("PRODUCT VARIANT (TYPE)", ["L", "M", "H"], index=0)
air_temp = st.sidebar.slider("AIR TEMPERATURE [K]", 295.0, 304.0, 300.0, 0.1)
process_temp = st.sidebar.slider("PROCESS TEMPERATURE [K]", 305.0, 314.0, 310.0, 0.1)
rpm = st.sidebar.slider("ROTATIONAL SPEED [RPM]", 1150, 2900, 1500, 10)
torque = st.sidebar.slider("TORQUE [Nm]", 3.0, 77.0, 40.0, 0.5)
tool_wear = st.sidebar.slider("TOOL WEAR [min]", 0, 260, 100, 1)

st.sidebar.markdown("---")
preset = st.sidebar.selectbox("QUICK SCENARIO PRESET", ["Custom (manual)", "Healthy baseline", "High-torque overstrain risk", "Heat dissipation risk", "Aging tool wear risk"])

if preset != "Custom (manual)":
    presets = {
        "Healthy baseline": dict(air=300.0, proc=310.0, rpm=1500, torque=35.0, wear=60),
        "High-torque overstrain risk": dict(air=300.5, proc=310.5, rpm=1350, torque=68.0, wear=210),
        "Heat dissipation risk": dict(air=302.5, proc=310.8, rpm=1330, torque=42.0, wear=90),
        "Aging tool wear risk": dict(air=300.2, proc=309.9, rpm=1550, torque=45.0, wear=245),
    }
    p = presets[preset]
    air_temp, process_temp, rpm, torque, tool_wear = p["air"], p["proc"], p["rpm"], p["torque"], p["wear"]
    st.sidebar.info(f"Preset loaded: {preset}")

# --------------------------------------------------------------------------
# BUILD FEATURE VECTOR & PREDICT
# --------------------------------------------------------------------------
temp_diff = process_temp - air_temp
power_proxy = torque * rpm

raw_row = {
    sanitize("Air temperature [K]"): air_temp,
    sanitize("Process temperature [K]"): process_temp,
    sanitize("Rotational speed [rpm]"): rpm,
    sanitize("Torque [Nm]"): torque,
    sanitize("Tool wear [min]"): tool_wear,
    "temp_diff": temp_diff,
    "power_proxy": power_proxy,
    "Type_H": 1 if machine_type == "H" else 0,
    "Type_L": 1 if machine_type == "L" else 0,
    "Type_M": 1 if machine_type == "M" else 0,
}
row_df = pd.DataFrame([raw_row])
for c in feature_cols:
    if c not in row_df.columns:
        row_df[c] = 0
row_df = row_df[feature_cols]

if binary_meta["needs_scaling"]:
    numeric_feats = [sanitize(c) for c in NUMERIC_RAW + ["temp_diff", "power_proxy"]]
    row_scaled = row_df.copy()
    row_scaled[numeric_feats] = scaler.transform(row_df[numeric_feats])
    fail_proba = binary_model.predict_proba(row_scaled)[0, 1]
else:
    fail_proba = binary_model.predict_proba(row_df)[0, 1]

multi_proba = multiclass_model.predict_proba(row_df)[0]
multi_classes = multiclass_model.classes_

# --------------------------------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------------------------------
tab_live, tab_models, tab_eda, tab_data = st.tabs(
    ["◉ LIVE RISK MONITOR", "▤ MODEL COMPARISON", "▦ EDA INSIGHTS", "▥ DATASET"]
)

# ===================== TAB 1: LIVE RISK MONITOR =====================
with tab_live:
    col_gauge, col_metrics, col_type = st.columns([1.1, 1, 1])

    with col_gauge:
        risk_pct = fail_proba * 100
        gauge_color = "#2be2c8" if risk_pct < 30 else ("#ff9340" if risk_pct < 65 else "#ff4d5e")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 42, "family": "Rajdhani", "color": "#e8ecf3"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#7c8698", "tickfont": {"color": "#7c8698", "size": 10}},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "#161c2a",
                "borderwidth": 1,
                "bordercolor": "#232b3d",
                "steps": [
                    {"range": [0, 30], "color": "rgba(43,226,200,0.12)"},
                    {"range": [30, 65], "color": "rgba(255,147,64,0.12)"},
                    {"range": [65, 100], "color": "rgba(255,77,94,0.12)"},
                ],
                "threshold": {"line": {"color": "#ff4d5e", "width": 3}, "thickness": 0.8, "value": 65},
            },
            title={"text": "FAILURE RISK — NEXT CYCLE", "font": {"size": 13, "family": "JetBrains Mono", "color": "#7c8698"}},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(t=50, b=10, l=25, r=25),
        )
        st.plotly_chart(fig, width='stretch')

        if risk_pct >= 65:
            st.markdown('<span class="badge-danger">⚠ ALERT — SCHEDULE MAINTENANCE NOW</span>', unsafe_allow_html=True)
        elif risk_pct >= 30:
            st.markdown('<span class="badge-danger" style="animation:none;color:#ff9340;border-color:rgba(255,147,64,0.5);background:rgba(255,147,64,0.1);">⚠ WATCH — ELEVATED RISK</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-ok">✓ NOMINAL — NO ACTION REQUIRED</span>', unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<div class="panel"><div class="panel-title">Derived Engineering Features</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Temp Differential</div><div class="metric-value mono">{temp_diff:.1f} K</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Power Proxy</div><div class="metric-value mono">{power_proxy:,.0f}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Product Variant</div><div class="metric-value mono">{machine_type}</div></div>', unsafe_allow_html=True)
        with m4:
            hdf_flag = "YES" if (temp_diff < 8.6 and rpm < 1380) else "NO"
            st.markdown(f'<div class="metric-box"><div class="metric-label">HDF Zone?</div><div class="metric-value mono">{hdf_flag}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""<div class="panel"><div class="panel-title">Active Pipeline</div>
        <span class="mono" style="color:#e8ecf3;font-size:13px;">Binary model &nbsp;→&nbsp; <b>{binary_meta['name']}</b> (GridSearchCV tuned)</span><br><br>
        <span class="mono" style="color:#e8ecf3;font-size:13px;">Multi-class model &nbsp;→&nbsp; <b>Random Forest</b></span>
        </div>""", unsafe_allow_html=True)

    with col_type:
        st.markdown('<div class="panel"><div class="panel-title">Predicted Failure Type Breakdown</div>', unsafe_allow_html=True)
        order = np.argsort(multi_proba)[::-1]
        colors = {"No Failure": "#2be2c8", "TWF": "#ff9340", "HDF": "#ff4d5e", "PWF": "#ffb84d", "OSF": "#8a5cff", "RNF": "#5cd6ff"}
        for idx in order:
            cls = multi_classes[idx]
            p = multi_proba[idx] * 100
            bar_color = colors.get(cls, "#2be2c8")
            st.markdown(f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:12px; color:#c8ceda;">
                <span>{cls}</span><span>{p:.1f}%</span>
              </div>
              <div style="background:#161c2a; border-radius:4px; height:8px; overflow:hidden;">
                <div style="width:{p}%; background:{bar_color}; height:100%; box-shadow:0 0 6px {bar_color};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div class="panel"><div class="panel-title">How to Read This Console</div>
    <span style="color:#c8ceda; font-size:14px;">
    The gauge on the left shows the tuned binary model's probability that <b>this exact combination of sensor readings</b> will result in a machine failure.
    Above 65% the console raises a maintenance alert. The panel on the right shows the multi-class model's breakdown of which specific failure mode
    (Tool Wear, Heat Dissipation, Power Failure, Overstrain, or Random) is most likely, so a maintenance team knows which spare part to bring on the first visit.
    Try the quick scenario presets in the sidebar to see how torque, tool wear, and the temperature differential each push risk up.
    </span></div>""", unsafe_allow_html=True)

# ===================== TAB 2: MODEL COMPARISON =====================
with tab_models:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><div class="panel-title">Binary Pipeline — "Will It Fail?"</div>', unsafe_allow_html=True)
        st.dataframe(binary_cmp.set_index("Model").style.background_gradient(cmap="YlGnBu", axis=0), width='stretch')
        st.markdown(f"""<span class="mono" style="font-size:12px; color:#7c8698;">
        Best model: <b style="color:#2be2c8;">{summary['best_binary_model']}</b> ·
        GridSearchCV best params: {summary['tuning_summary']['best_params']} ·
        Tuned F1: <b>{summary['tuning_summary']['tuned_f1']:.4f}</b> (vs {summary['tuning_summary']['untuned_f1']:.4f} untuned)
        </span>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><div class="panel-title">Multi-Class Pipeline — "Which Failure Type?"</div>', unsafe_allow_html=True)
        st.dataframe(multi_cmp.set_index("Class").style.background_gradient(cmap="OrRd_r", axis=0), width='stretch')
        st.markdown(f"""<span class="mono" style="font-size:12px; color:#7c8698;">
        Macro-avg F1: <b style="color:#ff9340;">{summary['multiclass_macro_f1']:.4f}</b> — pulled down by the rarest classes (TWF, RNF)
        which have fewer than 50 examples in the full 10,000-row dataset.
        </span>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Confusion Matrices</div>', unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        st.image("Images/06_binary_confusion_matrix.png", width='stretch')
    with ic2:
        st.image("Images/07_multiclass_confusion_matrix.png", width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Feature Importance — Binary vs Multi-Class</div>', unsafe_allow_html=True)
    st.image("Images/08_feature_importance_comparison.png", width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class="panel"><div class="panel-title">Deployment Recommendation</div>
    <span style="color:#c8ceda; font-size:14px; line-height:1.7;">
    The tuned binary model runs continuously on live sensor streams and fires a maintenance alert the moment failure risk crosses the decision threshold.
    Once an alert fires, the multi-class model classifies which failure type is most likely, so the maintenance team can bring the correct replacement
    part (cutting tool, cooling component, drive belt) on the first visit instead of guessing. The rarest failure types (TWF, RNF) currently have too
    few examples for the multi-class model to learn reliably — in production these would either be grouped into a "manual inspection required" bucket
    or backfilled with more labeled cycles over time.
    </span></div>""", unsafe_allow_html=True)

# ===================== TAB 3: EDA INSIGHTS =====================
with tab_eda:
    st.markdown('<div class="panel"><div class="panel-title">Key Insights</div>', unsafe_allow_html=True)
    for i, ins in enumerate(insights, 1):
        st.markdown(f"""<div style="display:flex; gap:12px; margin-bottom:10px;">
        <span class="mono" style="color:#ff9340; font-weight:700;">{i:02d}</span>
        <span style="color:#c8ceda; font-size:14px;">{ins}</span></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    row1 = st.columns(2)
    imgs = [
        ("Images/01_torque_toolwear_vs_failure.png", "Torque & Tool Wear vs Failure"),
        ("Images/02_temp_diff_distribution.png", "Temperature Differential Distribution"),
        ("Images/03_failure_type_distribution.png", "Failure Type Distribution"),
        ("Images/04_correlation_heatmap.png", "Correlation Heatmap"),
        ("Images/05_power_proxy_vs_failure.png", "Power Proxy vs Failure"),
    ]
    for i, (path, caption) in enumerate(imgs):
        with row1[i % 2]:
            st.markdown(f'<div class="panel"><div class="panel-title">{caption}</div>', unsafe_allow_html=True)
            st.image(path, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

# ===================== TAB 4: DATASET =====================
with tab_data:
    df_raw = pd.read_csv("Dataset/ai4i2020.csv")
    st.markdown(f"""<div class="panel"><div class="panel-title">AI4I 2020 Predictive Maintenance Dataset</div>
    <span class="mono" style="color:#c8ceda; font-size:13px;">
    {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns · {summary['failure_rate_pct']}% overall failure rate ·
    Failure breakdown: {", ".join(f"{k}={v}" for k, v in summary['failure_counts'].items())}
    </span></div>""", unsafe_allow_html=True)
    st.dataframe(df_raw.head(200), width='stretch', height=420)

st.markdown('<div class="footer-note">MAINT/OS · Binary + Multi-Class Predictive Maintenance Pipeline · AI4I 2020 Dataset · Built with scikit-learn, XGBoost & Streamlit</div>', unsafe_allow_html=True)
