"""
01_eda_analysis.py — Exploratory Data Analysis
Industrial Engineering Admin Dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ─────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
sns.set_theme(style="darkgrid", palette="muted")
BASE = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(BASE, "data", "industrial_data.csv"))

print("=" * 60)
print("INDUSTRIAL DATA — EXPLORATORY ANALYSIS")
print("=" * 60)
print(df.info())
print()
print("Descriptive Statistics:")
print(df.describe().round(3))
print()

# ── Missing values ─────────────────────────────────────────────────────────────
print("Missing Values:")
print(df.isnull().sum())
print()

# ── Outlier detection via IQR ──────────────────────────────────────────────────
numeric_cols = ["Material_Used", "Processing_Time", "Energy_Consumption", "Machine_Availability"]
print("Outlier Detection (IQR Method):")
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"  {col}: {len(outliers)} outliers (bounds: [{lower:.2f}, {upper:.2f}])")

print()

# ── Correlation heatmap ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0a0c0f")
fig.suptitle("Industrial Data — EDA Analysis", fontsize=16, color="#f0a500",
             fontfamily="monospace", y=0.98)

# Heatmap
ax = axes[0, 0]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            linewidths=0.5, linecolor="#1e2530", annot_kws={"size": 11})
ax.set_title("Correlation Matrix", color="#d4dbe8", fontsize=12)
ax.tick_params(colors="#5a6478")

# Job Status bar
ax2 = axes[0, 1]
status_counts = df["Job_Status"].value_counts()
colors = ["#3ecf8e", "#f0a500", "#e05a5a"]
bars = ax2.bar(status_counts.index, status_counts.values, color=colors, edgecolor="#1e2530")
ax2.set_title("Job Status Distribution", color="#d4dbe8", fontsize=12)
ax2.set_ylabel("Count", color="#5a6478")
ax2.tick_params(colors="#5a6478")
for bar, v in zip(bars, status_counts.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
             str(v), ha="center", color="#d4dbe8", fontsize=10)

# Processing Time distribution
ax3 = axes[1, 0]
ax3.hist(df["Processing_Time"], bins=20, color="#4da6ff", edgecolor="#0a0c0f", alpha=0.85)
ax3.set_title("Processing Time Distribution", color="#d4dbe8", fontsize=12)
ax3.set_xlabel("Processing Time (min)", color="#5a6478")
ax3.set_ylabel("Frequency", color="#5a6478")
ax3.tick_params(colors="#5a6478")

# Energy by Operation
ax4 = axes[1, 1]
energy_by_op = df.groupby("Operation_Type")["Energy_Consumption"].mean().sort_values()
bars4 = ax4.barh(energy_by_op.index, energy_by_op.values, color="#f0a500", edgecolor="#1e2530")
ax4.set_title("Avg Energy by Operation Type", color="#d4dbe8", fontsize=12)
ax4.set_xlabel("Energy (kWh)", color="#5a6478")
ax4.tick_params(colors="#5a6478")

for ax in axes.flat:
    ax.set_facecolor("#111418")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2530")

plt.tight_layout()
out_path = os.path.join(BASE, "notebooks", "eda_output.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0c0f")
print(f"EDA chart saved to: {out_path}")
plt.close()

# ── Summary insights ──────────────────────────────────────────────────────────
print()
print("KEY INSIGHTS:")
print(f"  • Total records: {len(df):,}")
print(f"  • Completion rate: {(df['Job_Status']=='Completed').mean()*100:.1f}%")
print(f"  • Avg processing time: {df['Processing_Time'].mean():.1f} min")
print(f"  • Avg energy consumption: {df['Energy_Consumption'].mean():.2f} kWh")
print(f"  • Avg machine availability: {df['Machine_Availability'].mean():.1f}%")
print(f"  • Most common operation: {df['Operation_Type'].mode()[0]}")
print(f"  • Most active machine: {df['Machine_ID'].mode()[0]}")
