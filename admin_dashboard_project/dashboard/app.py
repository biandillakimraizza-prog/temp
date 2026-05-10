"""
app.py — Industrial Production Admin Dashboard
Streamlit-based interactive dashboard with:
  • KPI cards
  • Production trend chart
  • Actual vs Predicted scatter
  • 3-Sigma Residual Control Chart
  • Feature coefficient bar chart
  • Q-Q residual normality plot
  • Residuals vs Fitted
  • Machine availability & efficiency charts
  • Anomaly alert system
  • What-if simulator
  • Export functionality
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize
import plotly.graph_objects as go

from components import (
    production_trend_chart,
    actual_vs_predicted_chart,
    control_chart,
    coefficient_chart,
    qq_plot,
    status_donut,
    energy_by_operation,
    machine_availability_bar,
    residuals_vs_fitted,
    efficiency_chart,
    COLORS,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IndustrOps — Production Admin Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ─────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load data & model ──────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    path = os.path.join(BASE, "data", "processed_data.csv")
    df = pd.read_csv(path)
    df["Scheduled_Start"] = pd.to_datetime(df["Scheduled_Start"])
    df["Scheduled_End"]   = pd.to_datetime(df["Scheduled_End"])
    return df

@st.cache_resource
def load_model():
    path = os.path.join(BASE, "models", "production_model.pkl")
    return joblib.load(path)

df_full  = load_data()
pkg      = load_model()
model    = pkg["model"]
scaler   = pkg["scaler"]
features = pkg["features"]
r2       = pkg["r2"]
rmse     = pkg["rmse"]
mae      = pkg["mae"]
ucl      = pkg["ucl"]
lcl      = pkg["lcl"]
mean_r   = pkg["mean_r"]
std_r    = pkg["std_r"]
ooc_pct  = pkg["ooc_pct"]
y_test   = pkg["y_test"]
preds    = pkg["preds"]
residuals = pkg["residuals"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    display:flex; align-items:center; gap:16px;
    border-bottom:1px solid #1e2530;
    padding-bottom:1rem; margin-bottom:1.5rem;">
  <div style="
    width:44px; height:44px;
    background:#f0a500;
    clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
    display:flex; align-items:center; justify-content:center;">
    <span style="font-size:18px;">⚙</span>
  </div>
  <div>
    <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.8rem;
                font-weight:900; letter-spacing:3px; color:#fff; line-height:1;">
      INDUST<span style="color:#f0a500;">OPS</span>
    </div>
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                letter-spacing:3px; color:#5a6478; text-transform:uppercase;">
      Production Admin Dashboard · Industrial Engineering
    </div>
  </div>
  <div style="margin-left:auto; display:flex; gap:16px; align-items:center;">
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                color:#3ecf8e; letter-spacing:2px;">
      ● LIVE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR — Control Panel ─────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙ Control Panel")
st.sidebar.markdown("---")

# Date range filter
min_date = df_full["Scheduled_Start"].dt.date.min()
max_date = df_full["Scheduled_Start"].dt.date.max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

st.sidebar.markdown("---")
machine_opts  = ["All"] + sorted(df_full["Machine_ID"].unique().tolist())
op_opts       = ["All"] + sorted(df_full["Operation_Type"].unique().tolist())
status_opts   = ["All"] + sorted(df_full["Job_Status"].unique().tolist())

sel_machine = st.sidebar.selectbox("Machine ID", machine_opts)
sel_op      = st.sidebar.selectbox("Operation Type", op_opts)
sel_status  = st.sidebar.selectbox("Job Status", status_opts)

st.sidebar.markdown("---")
show_diagnostics = st.sidebar.checkbox("Show Model Diagnostics", value=True)
show_whatif      = st.sidebar.checkbox("Show What-If Simulator", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-family:Share Tech Mono,monospace; font-size:0.6rem;
            color:#5a6478; letter-spacing:1px; line-height:1.8;'>
MODEL METRICS<br>
R² &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{r2:.4f}<br>
RMSE &nbsp;{rmse:.4f}<br>
MAE &nbsp;&nbsp;{mae:.4f}<br>
OOC &nbsp;&nbsp;{ooc_pct:.1f}%
</div>
""", unsafe_allow_html=True)


# ── Filter data ────────────────────────────────────────────────────────────────
df = df_full.copy()

if len(date_range) == 2:
    start_dt = pd.to_datetime(date_range[0])
    end_dt   = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df = df[(df["Scheduled_Start"] >= start_dt) & (df["Scheduled_Start"] < end_dt)]

if sel_machine != "All":
    df = df[df["Machine_ID"] == sel_machine]
if sel_op != "All":
    df = df[df["Operation_Type"] == sel_op]
if sel_status != "All":
    df = df[df["Job_Status"] == sel_status]

n_jobs      = len(df)
n_completed = (df["Job_Status"] == "Completed").sum()
n_delayed   = (df["Job_Status"] == "Delayed").sum()
n_failed    = (df["Job_Status"] == "Failed").sum()
comp_rate   = n_completed / n_jobs * 100 if n_jobs > 0 else 0
avg_output  = df["Production_Output"].mean() if n_jobs > 0 else 0
stability   = 1 - df["Production_Output"].std() / df["Production_Output"].mean() if n_jobs > 0 else 0
avg_energy  = df["Energy_Consumption"].mean() if n_jobs > 0 else 0
avg_avail   = df["Machine_Availability"].mean() if n_jobs > 0 else 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.65rem;letter-spacing:3px;color:#f0a500;text-transform:uppercase;">// Key Performance Indicators</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Jobs",       f"{n_jobs:,}",      help="Jobs matching current filters")
c2.metric("Avg Output",       f"{avg_output:.1f}", help="Mean Production_Output (filtered)")
c3.metric("Completion Rate",  f"{comp_rate:.1f}%", delta=f"{n_completed} jobs")
c4.metric("Stability Index",  f"{stability:.3f}",  help="1 − (σ/μ) of production output")
c5.metric("Avg Energy (kWh)", f"{avg_energy:.2f}", help="Mean energy consumption per job")
c6.metric("Avg Availability", f"{avg_avail:.1f}%", help="Mean machine availability %")

# ── Anomaly Alert Banner ───────────────────────────────────────────────────────
st.markdown("")
violations = np.sum((residuals > ucl) | (residuals < lcl))
if violations > 0:
    st.error(
        f"⚠ PROCESS ALERT — {int(violations)} residual(s) exceed 3-sigma control limits "
        f"({ooc_pct:.1f}% out-of-control). Investigate sensor drift or equipment degradation."
    )
else:
    st.success("✔ Process In Control — All residuals within 3-sigma limits (UCL/LCL).")

# ── Delayed / Failed quick alerts ─────────────────────────────────────────────
if n_failed > 0:
    st.warning(f"⚠  {n_failed} FAILED job(s) detected in current filter. Review machine logs.")
if n_delayed > (n_jobs * 0.25) and n_jobs > 0:
    st.warning(f"⚠  High delay rate: {n_delayed/n_jobs*100:.1f}% of jobs delayed. Check scheduling.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PRODUCTION TREND + STATUS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.65rem;letter-spacing:3px;color:#f0a500;text-transform:uppercase;">// Production Overview</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])
with col_left:
    if n_jobs > 0:
        st.plotly_chart(production_trend_chart(df), use_container_width=True, key="trend")
    else:
        st.info("No data for selected filters.")

with col_right:
    if n_jobs > 0:
        st.plotly_chart(status_donut(df), use_container_width=True, key="status")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ACTUAL vs PREDICTED + CONTROL CHART
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.65rem;letter-spacing:3px;color:#f0a500;text-transform:uppercase;">// Regression Forecast</p>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(actual_vs_predicted_chart(y_test, preds), use_container_width=True, key="avp")
with col_b:
    st.plotly_chart(control_chart(residuals, ucl, lcl, mean_r), use_container_width=True, key="spc")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FEATURE COEFFICIENTS + EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.65rem;letter-spacing:3px;color:#f0a500;text-transform:uppercase;">// Feature Impact & Efficiency</p>', unsafe_allow_html=True)

col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(
        coefficient_chart(features, model.coef_),
        use_container_width=True, key="coef"
    )
with col_d:
    if n_jobs > 0:
        st.plotly_chart(efficiency_chart(df), use_container_width=True, key="eff")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MACHINE & ENERGY CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.65rem;letter-spacing:3px;color:#f0a500;text-transform:uppercase;">// Machine & Energy Analytics</p>', unsafe_allow_html=True)

col_e, col_f = st.columns(2)
with col_e:
    if n_jobs > 0:
        st.plotly_chart(machine_availability_bar(df), use_container_width=True, key="avail")
with col_f:
    if n_jobs > 0:
        st.plotly_chart(energy_by_operation(df), use_container_width=True, key="energy")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MODEL DIAGNOSTICS (Q-Q + Residuals vs Fitted)
# ═══════════════════════════════════════════════════════════════════════════════
if show_diagnostics:
    st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.65rem;letter-spacing:3px;color:#f0a500;text-transform:uppercase;">// Model Diagnostics</p>', unsafe_allow_html=True)

    col_g, col_h = st.columns(2)
    with col_g:
        st.plotly_chart(qq_plot(residuals), use_container_width=True, key="qq")
    with col_h:
        st.plotly_chart(residuals_vs_fitted(preds, residuals), use_container_width=True, key="rvf")

    # Model metrics table
    st.markdown("##### Model Performance Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["R-Squared (R²)", "Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)",
                   "Out-of-Control %", "UCL (+3σ)", "LCL (−3σ)"],
        "Value": [f"{r2:.4f}", f"{rmse:.4f}", f"{mae:.4f}",
                  f"{ooc_pct:.2f}%", f"{ucl:.4f}", f"{lcl:.4f}"],
        "Threshold / Target": ["> 0.75", "< 5% of mean output", "Minimize",
                               "< 5%", "—", "—"],
        "Status": [
            "✔ Pass" if r2 > 0.75 else "✗ Review",
            "✔ Pass" if rmse < avg_output * 0.05 else "✗ Review",
            "—",
            "✔ Pass" if ooc_pct < 5 else "✗ Investigate",
            "—", "—",
        ],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: WHAT-IF OPTIMIZER SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
if show_whatif:
    st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.65rem;letter-spacing:3px;color:#f0a500;text-transform:uppercase;">// What-If Production Optimizer</p>', unsafe_allow_html=True)
    st.markdown("Adjust process parameters to simulate predicted production output, or run the constrained optimizer.")

    wi_col1, wi_col2 = st.columns([1, 1])

    with wi_col1:
        st.markdown("**Manual Parameter Simulator**")
        wi_material  = st.slider("Material Used (kg)",          1.0,  5.0,  3.0, 0.01)
        wi_energy    = st.slider("Energy Consumption (kWh)",    2.0,  15.0, 8.5, 0.01)
        wi_avail     = st.slider("Machine Availability (%)",    80,   99,   89)
        wi_op        = st.selectbox("Operation Type", df_full["Operation_Type"].unique().tolist())
        wi_machine   = st.selectbox("Machine ID",     df_full["Machine_ID"].unique().tolist())

        le_op   = pkg["le_op"]
        le_mach = pkg["le_mach"]

        # Build feature vector with representative lags
        lag_val  = df_full["Production_Output"].mean()
        rmean    = df_full["Production_Output"].mean()
        wi_vec   = np.array([[
            wi_material, wi_energy, wi_avail,
            le_op.transform([wi_op])[0],
            le_mach.transform([wi_machine])[0],
            9, 1,          # hour=9, day_of_week=1 (Tuesday)
            lag_val, lag_val, lag_val, rmean,
        ]])
        wi_scaled = scaler.transform(wi_vec)
        wi_pred   = model.predict(wi_scaled)[0]

        st.markdown("---")
        st.markdown(f"""
<div style="
    background:#111418; border:1px solid #1e2530;
    border-left:3px solid #f0a500;
    padding:1.25rem 1.5rem; margin-top:.5rem;">
  <div style="font-family:Share Tech Mono,monospace; font-size:0.65rem;
              letter-spacing:2px; color:#5a6478; text-transform:uppercase; margin-bottom:.3rem;">
    Predicted Production Output
  </div>
  <div style="font-family:'Barlow Condensed',sans-serif; font-size:2.8rem;
              font-weight:700; color:#f0a500;">
    {wi_pred:.2f} <span style="font-size:1rem; color:#5a6478;">units</span>
  </div>
</div>
""", unsafe_allow_html=True)

    with wi_col2:
        st.markdown("**Constrained Optimizer**")
        st.markdown(
            "Finds the *optimal* material, energy, and availability settings to **maximize** "
            "predicted production output within feasible industrial bounds."
        )

        run_opt = st.button("🔍 Run Optimizer (SLSQP)")
        if run_opt:
            with st.spinner("Running constrained optimization…"):
                op_enc_val   = le_op.transform([wi_op])[0]
                mach_enc_val = le_mach.transform([wi_machine])[0]

                def objective(x):
                    vec = np.array([[x[0], x[1], x[2],
                                     op_enc_val, mach_enc_val,
                                     9, 1, lag_val, lag_val, lag_val, rmean]])
                    return -model.predict(scaler.transform(vec))[0]

                bounds = [(1.0, 5.0), (2.0, 15.0), (80.0, 99.0)]
                x0     = [wi_material, wi_energy, wi_avail]
                result = minimize(objective, x0, bounds=bounds, method="SLSQP")
                opt_pred = -result.fun

            st.markdown(f"""
<div style="background:#111418; border:1px solid #1e2530;
            border-left:3px solid #3ecf8e; padding:1.25rem 1.5rem; margin-top:.5rem;">
  <div style="font-family:Share Tech Mono,monospace; font-size:0.6rem;
              letter-spacing:2px; color:#5a6478; text-transform:uppercase; margin-bottom:.5rem;">
    Optimal Settings Found
  </div>
  <table style="width:100%; font-family:Share Tech Mono,monospace; font-size:0.75rem; color:#d4dbe8;">
    <tr><td style="color:#5a6478;">Material Used</td><td style="color:#3ecf8e;">{result.x[0]:.3f} kg</td></tr>
    <tr><td style="color:#5a6478;">Energy Consumption</td><td style="color:#3ecf8e;">{result.x[1]:.3f} kWh</td></tr>
    <tr><td style="color:#5a6478;">Machine Availability</td><td style="color:#3ecf8e;">{result.x[2]:.1f}%</td></tr>
    <tr><td style="color:#5a6478;">Max Predicted Output</td>
        <td style="color:#f0a500; font-size:1.1rem; font-weight:700;">{opt_pred:.2f} units</td></tr>
    <tr><td style="color:#5a6478;">Gain vs Manual</td>
        <td style="color:#3ecf8e;">+{opt_pred - wi_pred:.2f} units</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: JOB DATA TABLE + EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.65rem;letter-spacing:3px;color:#f0a500;text-transform:uppercase;">// Job Records</p>', unsafe_allow_html=True)

display_cols = [
    "Job_ID", "Machine_ID", "Operation_Type", "Material_Used",
    "Processing_Time", "Energy_Consumption", "Machine_Availability",
    "Job_Status", "Optimization_Category", "Production_Output",
]

st.dataframe(
    df[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=320,
)

col_exp1, col_exp2 = st.columns([1, 4])
with col_exp1:
    csv_data = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Export CSV",
        data=csv_data,
        file_name="industrial_report_filtered.csv",
        mime="text/csv",
    )

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="font-family:Share Tech Mono,monospace; font-size:0.6rem;
            letter-spacing:1px; color:#5a6478; text-align:center;
            border-top:1px solid #1e2530; padding-top:1rem; margin-top:1rem;">
  INDUSTROPS PRODUCTION ADMIN DASHBOARD &nbsp;·&nbsp;
  {n_jobs:,} JOBS LOADED &nbsp;·&nbsp;
  MODEL R²={r2:.4f} &nbsp;·&nbsp;
  INDUSTRIAL ENGINEERING ANALYTICS
</div>
""", unsafe_allow_html=True)
