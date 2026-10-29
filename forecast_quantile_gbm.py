import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.ensemble import GradientBoostingRegressor

# 1. Load the updated 49-month dataset
df = pd.read_csv("monthly_invoices.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

ts_df = df.set_index("Date").asfreq("MS")
ts_df["Amount_Clean"] = ts_df["Amount_Clean"].ffill()

# 2. SARIMA Forecast (Benchmark)
sarima_model = auto_arima(
    ts_df["Amount_Clean"],
    seasonal=True,
    m=12,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)
sarima_preds, sarima_conf = sarima_model.predict(n_periods=6, return_conf_int=True)

# 3. Supervised Feature Engineering
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

# 4. Train 3 Quantile Gradient Boosting Regressors
params = dict(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)

gbm_lower = GradientBoostingRegressor(loss="quantile", alpha=0.025, **params).fit(X_train, y_train)
gbm_median = GradientBoostingRegressor(loss="quantile", alpha=0.50, **params).fit(X_train, y_train)
gbm_upper = GradientBoostingRegressor(loss="quantile", alpha=0.975, **params).fit(X_train, y_train)

# 5. Recursive Multi-Step Projections
history = list(ts_df["Amount_Clean"].values)
gbm_low_preds, gbm_med_preds, gbm_high_preds = [], [], []

future_dates = pd.date_range(start=ts_df.index[-1] + pd.DateOffset(months=1), periods=6, freq="MS")

for dt in future_dates:
    feat_row = pd.DataFrame([{
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_12": history[-12],
        "rolling_mean_3": np.mean(history[-3:]),
        "month": dt.month,
        "quarter": dt.quarter
    }])
    
    pred_low = gbm_lower.predict(feat_row)[0]
    pred_med = gbm_median.predict(feat_row)[0]
    pred_high = gbm_upper.predict(feat_row)[0]
    
    gbm_low_preds.append(pred_low)
    gbm_med_preds.append(pred_med)
    gbm_high_preds.append(pred_high)
    
    # Recursive step feeds the median projection back into history
    history.append(pred_med)

# 6. Combined Comparison Table
comparison_df = pd.DataFrame({
    "SARIMA_Mid": np.round(sarima_preds, 2),
    "SARIMA_95%_CI": [f"${l:.1f} - ${u:.1f}" for l, u in sarima_conf],
    "GBM_Median": np.round(gbm_med_preds, 2),
    "GBM_95%_Interval": [f"${l:.1f} - ${u:.1f}" for l, u in zip(gbm_low_preds, gbm_high_preds)]
}, index=future_dates.strftime("%Y-%m"))

print("\n" + "=" * 78)
print("     6-MONTH FORECAST & 95% PREDICTION INTERVALS: SARIMA vs QUANTILE GBM")
print("=" * 78)
print(comparison_df.to_string())
print("=" * 78 + "\n")