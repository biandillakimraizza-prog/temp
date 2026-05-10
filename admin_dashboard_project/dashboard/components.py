"""
components.py — Reusable chart & diagnostic components
for the Industrial Production Admin Dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats


# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#0a0c0f",
    "surface":  "#111418",
    "border":   "#1e2530",
    "accent":   "#f0a500",
    "success":  "#3ecf8e",
    "warning":  "#f0a500",
    "danger":   "#e05a5a",
    "info":     "#4da6ff",
    "text":     "#d4dbe8",
    "muted":    "#5a6478",
    "accent2":  "#00c4a7",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,20,24,1)",
    font=dict(family="Share Tech Mono, monospace", color=COLORS["text"], size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], tickfont=dict(size=10)),
    yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"], borderwidth=1),
)


def _apply_layout(fig, title="", height=380):
    """Apply standard dark industrial layout to a figure."""
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(
            text=f"<b>{title.upper()}</b>",
            font=dict(family="Barlow Condensed, sans-serif", size=14, color="#fff"),
            x=0, xanchor="left",
        ),
        height=height,
    )
    return fig


# ── 1. Production Trend Chart ──────────────────────────────────────────────────
def production_trend_chart(df: pd.DataFrame, target_col: str = "Production_Output") -> go.Figure:
    """Line chart of production output over time with 7-day rolling average."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Scheduled_Start"], y=df[target_col],
        mode="lines",
        name="Actual Output",
        line=dict(color=COLORS["info"], width=1.2),
        opacity=0.7,
    ))

    if "rolling_mean_7" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Scheduled_Start"], y=df["rolling_mean_7"],
            mode="lines",
            name="7-Job Rolling Avg",
            line=dict(color=COLORS["accent"], width=2.5),
        ))

    _apply_layout(fig, "Production Output Trend", height=340)
    fig.update_xaxes(title_text="Scheduled Start")
    fig.update_yaxes(title_text="Production Output (units)")
    return fig


# ── 2. Actual vs Predicted Scatter ─────────────────────────────────────────────
def actual_vs_predicted_chart(y_actual: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Scatter of actual vs predicted with perfect-fit diagonal."""
    lo = min(y_actual.min(), y_pred.min()) - 2
    hi = max(y_actual.max(), y_pred.max()) + 2

    fig = go.Figure()

    # Perfect-fit line
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        name="Perfect Fit",
        line=dict(color=COLORS["muted"], width=1.5, dash="dash"),
    ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_actual, y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(
            color=COLORS["accent"],
            size=5,
            opacity=0.75,
            line=dict(color=COLORS["bg"], width=0.5),
        ),
    ))

    _apply_layout(fig, "Actual vs Predicted Output", height=380)
    fig.update_xaxes(title_text="Actual Output")
    fig.update_yaxes(title_text="Predicted Output")
    return fig


# ── 3. Residual Control Chart (Shewhart 3-sigma) ──────────────────────────────
def control_chart(residuals: np.ndarray, ucl: float, lcl: float, mean_r: float) -> go.Figure:
    """Shewhart control chart with UCL/LCL and highlighted out-of-control points."""
    idx = list(range(len(residuals)))
    violations = np.where((residuals > ucl) | (residuals < lcl))[0]

    fig = go.Figure()

    # Residual line
    fig.add_trace(go.Scatter(
        x=idx, y=residuals,
        mode="lines+markers",
        name="Residuals",
        line=dict(color=COLORS["info"], width=1.2),
        marker=dict(size=3, color=COLORS["info"]),
    ))

    # Violations
    if len(violations) > 0:
        fig.add_trace(go.Scatter(
            x=violations.tolist(),
            y=residuals[violations].tolist(),
            mode="markers",
            name="Out-of-Control",
            marker=dict(color=COLORS["danger"], size=8, symbol="x", line=dict(width=2, color=COLORS["danger"])),
        ))

    # Control lines
    for y_val, label, color in [
        (ucl, "UCL (+3σ)", COLORS["danger"]),
        (lcl, "LCL (−3σ)", COLORS["danger"]),
        (mean_r, "CL (mean)", COLORS["success"]),
    ]:
        fig.add_hline(
            y=y_val,
            line_dash="dash" if "σ" in label else "dot",
            line_color=color,
            line_width=1.5,
            annotation_text=f" {label}",
            annotation_font_color=color,
            annotation_font_size=10,
        )

    # UCL/LCL band
    fig.add_hrect(y0=lcl, y1=ucl, fillcolor=COLORS["success"], opacity=0.03, line_width=0)

    _apply_layout(fig, "Residual Control Chart (3-Sigma SPC)", height=360)
    fig.update_xaxes(title_text="Observation Index")
    fig.update_yaxes(title_text="Residual (Actual − Predicted)")
    return fig


# ── 4. Feature Coefficient Bar Chart ──────────────────────────────────────────
def coefficient_chart(features: list, coefficients: np.ndarray) -> go.Figure:
    """Horizontal bar chart of linear regression coefficients."""
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefficients})
    coef_df = coef_df.sort_values("Coefficient")
    colors = [COLORS["success"] if c > 0 else COLORS["danger"] for c in coef_df["Coefficient"]]

    fig = go.Figure(go.Bar(
        x=coef_df["Coefficient"],
        y=coef_df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in coef_df["Coefficient"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
    ))

    _apply_layout(fig, "Feature Coefficients — Impact on Production Output", height=380)
    fig.update_xaxes(title_text="Coefficient Value", zeroline=True, zerolinecolor=COLORS["border"], zerolinewidth=2)
    fig.update_yaxes(title_text="")
    return fig


# ── 5. Q-Q Residual Normality Plot ────────────────────────────────────────────
def qq_plot(residuals: np.ndarray) -> go.Figure:
    """Q-Q plot to assess normality of residuals."""
    (theoretical_q, sample_q), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    fit_line = [slope * q + intercept for q in theoretical_q]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(theoretical_q), y=list(sample_q),
        mode="markers",
        name="Residuals",
        marker=dict(color=COLORS["accent"], size=5, opacity=0.8),
    ))
    fig.add_trace(go.Scatter(
        x=list(theoretical_q), y=fit_line,
        mode="lines",
        name="Normal Line",
        line=dict(color=COLORS["danger"], width=1.5, dash="dash"),
    ))

    _apply_layout(fig, "Q-Q Plot — Residual Normality Check", height=340)
    fig.update_xaxes(title_text="Theoretical Quantiles")
    fig.update_yaxes(title_text="Sample Quantiles")
    return fig


# ── 6. Job Status Pie Chart ────────────────────────────────────────────────────
def status_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["Job_Status"].value_counts()
    colors_map = {
        "Completed": COLORS["success"],
        "Delayed":   COLORS["warning"],
        "Failed":    COLORS["danger"],
    }
    clrs = [colors_map.get(s, COLORS["muted"]) for s in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.55,
        marker=dict(colors=clrs, line=dict(color=COLORS["bg"], width=3)),
        textinfo="label+percent",
        textfont=dict(size=11),
    ))
    _apply_layout(fig, "Job Status Distribution", height=300)
    fig.update_layout(showlegend=False)
    return fig


# ── 7. Energy by Operation Type ───────────────────────────────────────────────
def energy_by_operation(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("Operation_Type")["Energy_Consumption"].mean().sort_values()
    clrs = [COLORS["accent"] if v == grp.max() else COLORS["info"] for v in grp.values]

    fig = go.Figure(go.Bar(
        x=grp.index.tolist(),
        y=grp.values.tolist(),
        marker_color=clrs,
        text=[f"{v:.2f}" for v in grp.values],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=10),
    ))
    _apply_layout(fig, "Avg Energy Consumption by Operation", height=300)
    fig.update_yaxes(title_text="Energy (kWh)")
    return fig


# ── 8. Machine Availability Gauge ─────────────────────────────────────────────
def machine_availability_bar(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("Machine_ID")["Machine_Availability"].mean().sort_index()
    colors = []
    for v in grp.values:
        if v >= 95:
            colors.append(COLORS["success"])
        elif v >= 88:
            colors.append(COLORS["info"])
        else:
            colors.append(COLORS["warning"])

    fig = go.Figure(go.Bar(
        x=grp.index.tolist(),
        y=grp.values.tolist(),
        marker_color=colors,
        text=[f"{v:.1f}%" for v in grp.values],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
    ))
    fig.add_hline(y=90, line_dash="dash", line_color=COLORS["muted"],
                  annotation_text=" 90% target", annotation_font_color=COLORS["muted"])
    _apply_layout(fig, "Machine Availability by ID", height=300)
    fig.update_yaxes(title_text="Availability (%)", range=[75, 105])
    return fig


# ── 9. Residuals vs Fitted ────────────────────────────────────────────────────
def residuals_vs_fitted(y_pred: np.ndarray, residuals: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        name="Residuals",
        marker=dict(color=COLORS["accent2"], size=5, opacity=0.7),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["danger"], line_width=1.5)
    _apply_layout(fig, "Residuals vs Fitted Values", height=340)
    fig.update_xaxes(title_text="Fitted (Predicted) Values")
    fig.update_yaxes(title_text="Residuals")
    return fig


# ── 10. Optimization Category Distribution ─────────────────────────────────────
def efficiency_chart(df: pd.DataFrame) -> go.Figure:
    order  = ["Optimal Efficiency", "High Efficiency", "Moderate Efficiency", "Low Efficiency"]
    counts = df["Optimization_Category"].value_counts().reindex(order, fill_value=0)
    clrs   = [COLORS["accent2"], COLORS["success"], COLORS["info"], COLORS["danger"]]

    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=clrs,
        text=counts.values.tolist(),
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
    ))
    _apply_layout(fig, "Optimization Efficiency Distribution", height=300)
    fig.update_yaxes(title_text="Job Count")
    return fig
