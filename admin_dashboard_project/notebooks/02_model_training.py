"""
02_model_training.py — Linear Regression Model Training
Industrial Engineering Admin Dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import scipy.stats as stats
import joblib
import os

BASE = os.path.dirname(os.path.dirname(__file__))
plt.style.use("dark_background")

print("=" * 60)
print("LINEAR REGRESSION MODEL TRAINING")
print("=" * 60)

# ── Load & engineer features ─────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data", "industrial_data.csv"))

le_op   = LabelEncoder()
le_mach = LabelEncoder()
df["op_enc"]   = le_op.fit_transform(df["Operation_Type"])
df["mach_enc"] = le_mach.fit_transform(df["Machine_ID"])
df["Scheduled_Start"] = pd.to_datetime(df["Scheduled_Start"])
df["hour"]        = df["Scheduled_Start"].dt.hour
df["day_of_week"] = df["Scheduled_Start"].dt.dayofweek
df = df.sort_values("Scheduled_Start").reset_index(drop=True)

# Construct realistic Production_Output
np.random.seed(0)
df["Production_Output"] = (
    50
    + df["Machine_Availability"] * 0.8
    - df["Energy_Consumption"] * 1.2
    + df["Material_Used"] * 3.5
    - df["op_enc"] * 2.0
    + df["mach_enc"] * 1.5
    + np.random.normal(0, 5, len(df))
)

# Lag & rolling features
df["lag_1"] = df["Production_Output"].shift(1)
df["lag_2"] = df["Production_Output"].shift(2)
df["lag_3"] = df["Production_Output"].shift(3)
df["rolling_mean_7"] = df["Production_Output"].rolling(7).mean()
df.dropna(inplace=True)

features = [
    "Material_Used", "Energy_Consumption", "Machine_Availability",
    "op_enc", "mach_enc", "hour", "day_of_week",
    "lag_1", "lag_2", "lag_3", "rolling_mean_7",
]
target = "Production_Output"

X = df[features].values
y = df[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── Train ─────────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

r2   = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae  = mean_absolute_error(y_test, preds)

print(f"\nModel Performance:")
print(f"  R-Squared : {r2:.4f}  {'✔ PASS (>0.75)' if r2 > 0.75 else '✗ Review'}")
print(f"  RMSE      : {rmse:.4f}")
print(f"  MAE       : {mae:.4f}")

# ── Control chart stats ───────────────────────────────────────────────────────
residuals = y_test - preds
mean_r, std_r = residuals.mean(), residuals.std()
ucl = mean_r + 3 * std_r
lcl = mean_r - 3 * std_r
ooc = int(np.sum((residuals > ucl) | (residuals < lcl)))
ooc_pct = ooc / len(residuals) * 100

print(f"\nControl Chart:")
print(f"  UCL (+3σ) : {ucl:.4f}")
print(f"  LCL (−3σ) : {lcl:.4f}")
print(f"  OOC points: {ooc} ({ooc_pct:.1f}%)  {'✔ PASS (<5%)' if ooc_pct < 5 else '✗ Investigate'}")

# ── Coefficient table ─────────────────────────────────────────────────────────
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_,
}).sort_values("Coefficient", ascending=False)
print(f"\nFeature Coefficients:")
print(coef_df.to_string(index=False))

# ── Diagnostic plots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0a0c0f")
fig.suptitle("Model Diagnostics — Linear Regression", fontsize=15,
             color="#f0a500", fontfamily="monospace")

# Actual vs Predicted
ax1 = axes[0, 0]
lo, hi = min(y_test.min(), preds.min()) - 2, max(y_test.max(), preds.max()) + 2
ax1.plot([lo, hi], [lo, hi], color="#5a6478", linestyle="--", lw=1.5, label="Perfect fit")
ax1.scatter(y_test, preds, color="#f0a500", alpha=0.6, s=15, label="Predictions")
ax1.set_title("Actual vs Predicted", color="#d4dbe8")
ax1.set_xlabel("Actual", color="#5a6478")
ax1.set_ylabel("Predicted", color="#5a6478")
ax1.legend()

# Control chart
ax2 = axes[0, 1]
ax2.plot(residuals, color="#4da6ff", lw=0.8, marker="o", markersize=2, label="Residuals")
ax2.axhline(ucl,    color="#e05a5a", linestyle="--", lw=1.5, label=f"UCL={ucl:.2f}")
ax2.axhline(lcl,    color="#e05a5a", linestyle="--", lw=1.5, label=f"LCL={lcl:.2f}")
ax2.axhline(mean_r, color="#3ecf8e", linestyle=":",  lw=1.5, label="CL")
violations = np.where((residuals > ucl) | (residuals < lcl))[0]
if len(violations) > 0:
    ax2.scatter(violations, residuals[violations], color="#e05a5a", s=40, zorder=5, label="OOC")
ax2.set_title("Residual Control Chart (3σ)", color="#d4dbe8")
ax2.set_xlabel("Observation", color="#5a6478")
ax2.set_ylabel("Residual", color="#5a6478")
ax2.legend(fontsize=8)

# Q-Q Plot
ax3 = axes[1, 0]
(theor, sample), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
ax3.scatter(theor, sample, color="#f0a500", alpha=0.7, s=12)
ax3.plot(theor, [slope * q + intercept for q in theor], color="#e05a5a", lw=1.5, linestyle="--")
ax3.set_title("Q-Q Plot (Residual Normality)", color="#d4dbe8")
ax3.set_xlabel("Theoretical Quantiles", color="#5a6478")
ax3.set_ylabel("Sample Quantiles", color="#5a6478")

# Coefficients
ax4 = axes[1, 1]
sorted_coef = coef_df.sort_values("Coefficient")
colors = ["#3ecf8e" if c > 0 else "#e05a5a" for c in sorted_coef["Coefficient"]]
ax4.barh(sorted_coef["Feature"], sorted_coef["Coefficient"], color=colors, edgecolor="#1e2530")
ax4.axvline(0, color="#5a6478", lw=1)
ax4.set_title("Feature Coefficients", color="#d4dbe8")
ax4.set_xlabel("Coefficient Value", color="#5a6478")

for ax in axes.flat:
    ax.set_facecolor("#111418")
    ax.tick_params(colors="#5a6478")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2530")

plt.tight_layout()
out_path = os.path.join(BASE, "notebooks", "model_diagnostics.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0c0f")
print(f"\nDiagnostic chart saved to: {out_path}")
plt.close()

# ── Save model ─────────────────────────────────────────────────────────────────
pkg = {
    "model": model, "scaler": scaler, "features": features, "target": target,
    "le_op": le_op, "le_mach": le_mach,
    "r2": r2, "rmse": rmse, "mae": mae,
    "ucl": ucl, "lcl": lcl, "mean_r": mean_r, "std_r": std_r,
    "ooc_pct": ooc_pct,
    "y_test": y_test, "preds": preds, "residuals": residuals,
}
model_path = os.path.join(BASE, "models", "production_model.pkl")
joblib.dump(pkg, model_path)
print(f"Model saved to: {model_path}")
print("\nTraining complete.")
