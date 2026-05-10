# IndustrOps — Industrial Production Admin Dashboard

> A production intelligence dashboard combining **Linear Regression Forecasting**, **Statistical Process Control (SPC)**, and **KPI Tracking** — built for plant managers and process engineers.

---

## Project Overview

This dashboard transforms raw industrial sensor data into actionable production intelligence:

| Feature | Description |
|---|---|
| 📊 Real-time KPIs | Avg output, completion rate, stability index, energy, availability |
| 📈 Production Trend | Time-series with 7-job rolling average |
| 🔮 Forecast vs Actual | Scatter plot with perfect-fit line |
| 🔴 Control Chart | 3-sigma Shewhart SPC with out-of-control highlighting |
| 📉 Feature Coefficients | Horizontal bar chart showing each variable's impact |
| 🧪 Model Diagnostics | Q-Q plot + Residuals vs Fitted |
| ⚙ What-If Simulator | Adjust parameters → instant predicted output |
| 🧠 Optimizer | SLSQP-based constrained optimizer for max production |
| ⚠ Anomaly Alerts | Auto-detects SPC violations, failures, high delay rates |
| ⬇ Export CSV | Download filtered job records |

---

## Deliverables Checklist

- [x] Kaggle dataset downloaded and loaded (`data/industrial_data.csv`)
- [x] EDA notebook with insights and outlier analysis (`notebooks/01_eda_analysis.py`)
- [x] Trained linear regression model with coefficient interpretation (`models/production_model.pkl`)
- [x] Residual diagnostics — normality Q-Q, residuals vs fitted (`dashboard/components.py`)
- [x] Interactive dashboard with at least 5 interactive widgets (date range, machine, operation, status, sliders, optimizer button)
- [x] Production trend line chart and Actual vs Predicted scatter plot
- [x] Control chart (3-sigma) with out-of-control highlighting
- [x] Feature importance / coefficient plot (horizontal bar)
- [x] Auto-refresh anomaly alert / warning system
- [x] Export functionality (CSV / filtered data)
- [x] README with setup and run instructions ← **you are here**

---

## Project Structure

```
admin_dashboard_project/
│
├── data/
│   ├── industrial_data.csv        ← raw Kaggle dataset
│   └── processed_data.csv         ← feature-engineered data (auto-generated)
│
├── notebooks/
│   ├── 01_eda_analysis.py         ← EDA: correlation, distributions, outliers
│   └── 02_model_training.py       ← Model training + diagnostic plots
│
├── dashboard/
│   ├── app.py                     ← Main Streamlit dashboard
│   ├── components.py              ← Reusable Plotly chart functions
│   └── assets/
│       └── style.css              ← Custom dark industrial styling
│
├── models/
│   └── production_model.pkl       ← Trained model + scaler + metrics (auto-generated)
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone / set up the project

```bash
cd admin_dashboard_project
```

### 2. Create a virtual environment

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run EDA (optional — generates `notebooks/eda_output.png`)

```bash
python notebooks/01_eda_analysis.py
```

### 5. Train / retrain the model (optional — model already pre-trained)

```bash
python notebooks/02_model_training.py
```

### 6. Launch the dashboard

```bash
cd dashboard
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**

---

## Technology Stack

| Layer | Library |
|---|---|
| Data | `pandas`, `numpy` |
| ML Model | `scikit-learn` (LinearRegression, StandardScaler) |
| Statistics | `scipy.stats` (Q-Q, probplot) |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Model Serialization | `joblib` |
| Optimization | `scipy.optimize.minimize` (SLSQP) |

---

## Model Details

- **Algorithm**: Ordinary Least Squares Linear Regression
- **Target**: `Production_Output` (constructed from process parameters)
- **Features**: `Material_Used`, `Energy_Consumption`, `Machine_Availability`, operation/machine encodings, time features, lag_1/2/3, rolling_mean_7
- **Train/Test Split**: 80% / 20%
- **Scaler**: `StandardScaler`

### Performance Thresholds

| Metric | Result | Target | Status |
|---|---|---|---|
| R-Squared | 0.7590 | > 0.75 | ✔ Pass |
| RMSE | 4.74 | < 5% of mean | ✔ Pass |
| Out-of-Control % | 0.0% | < 5% | ✔ Pass |

---

## Interpretation Guide for Engineers

- **Positive coefficient** → increasing that feature increases production output
- **Negative coefficient** → increasing that feature decreases production output
- **Out-of-control points** on control chart → possible sensor drift, raw material change, or equipment degradation
- **What-If Simulator** → test "what happens if I increase machine availability to 98%?"
- **Optimizer** → automatically finds the combination of material, energy, availability that maximizes predicted output within feasible bounds

---

## Executive Summary

This dashboard helps plant managers **reduce downtime** and **maximize yield** by:

1. **Detecting anomalies early** via the 3-sigma SPC control chart — out-of-control residuals signal process shifts before they become failures
2. **Forecasting production output** from process parameters using a validated regression model (R² = 0.76)
3. **Identifying key drivers** of output — machine availability and material usage show the highest positive impact
4. **Simulating process changes** in real-time using the What-If Simulator without stopping the production line
5. **Tracking KPIs continuously** with filtered, exportable job records for shift-level accountability

---

*IndustrOps — Industrial Engineering Analytics · Version 1.0*
