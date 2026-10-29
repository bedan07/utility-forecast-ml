import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.ensemble import GradientBoostingRegressor

# 1. Load the updated dataset (49 months)
df = pd.read_csv("monthly_invoices.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

ts_df = df.set_index("Date").asfreq("MS")
ts_df["Amount_Clean"] = ts_df["Amount_Clean"].ffill()

# 2. Fit SARIMA on all 49 months
sarima_model = auto_arima(
    ts_df["Amount_Clean"],
    seasonal=True,
    m=12,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)

sarima_preds, conf_int = sarima_model.predict(n_periods=6, return_conf_int=True)

# 3. Prepare Supervised Features for Gradient Boosting
data = pd.DataFrame(index=ts_df.index)
data["y"] = ts_df["Amount_Clean"]
data["lag_1"] = data["y"].shift(1)
data["lag_2"] = data["y"].shift(2)
data["lag_12"] = data["y"].shift(12)
data["rolling_mean_3"] = data["y"].shift(1).rolling(window=3).mean()
data["month"] = data.index.month
data["quarter"] = data.index.quarter

train_data = data.dropna()
features = ["lag_1", "lag_2", "lag_12", "rolling_mean_3", "month", "quarter"]
X_train = train_data[features]
y_train = train_data["y"]

# Fit GBM on all available historical data
gbm = GradientBoostingRegressor(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.05,
    random_state=42
)
gbm.fit(X_train, y_train)

# 4. Recursive Multi-Step Forecasting for GBM
history = list(ts_df["Amount_Clean"].values)
gbm_preds = []

future_dates = pd.date_range(
    start=ts_df.index[-1] + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)

for dt in future_dates:
    lag_1 = history[-1]
    lag_2 = history[-2]
    lag_12 = history[-12]
    rolling_3 = np.mean(history[-3:])
    month = dt.month
    quarter = dt.quarter

    feat_row = pd.DataFrame([{
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_12": lag_12,
        "rolling_mean_3": rolling_3,
        "month": month,
        "quarter": quarter
    }])

    pred = gbm.predict(feat_row)[0]
    gbm_preds.append(pred)
    history.append(pred)  # Feed prediction back into history for next step

# 5. Display Side-by-Side Projection Table
comparison_df = pd.DataFrame({
    "SARIMA_Forecast": np.round(sarima_preds, 2),
    "GBM_Forecast": np.round(gbm_preds, 2),
    "Difference (GBM - SARIMA)": np.round(np.array(gbm_preds) - sarima_preds, 2),
    "SARIMA_Lower_95%": np.round(conf_int[:, 0], 2),
    "SARIMA_Upper_95%": np.round(conf_int[:, 1], 2)
}, index=future_dates.strftime("%Y-%m"))

print("\n" + "=" * 70)
print("     6-MONTH ROLLING FUTURE FORECAST: SARIMA vs. GRADIENT BOOSTING")
print("=" * 70)
print(comparison_df.to_string())
print("=" * 70 + "\n")