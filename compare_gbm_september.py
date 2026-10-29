import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# 1. Load data
df = pd.read_csv("monthly_invoices.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

ts_df = df.set_index("Date").asfreq("MS")
ts_df["Amount_Clean"] = ts_df["Amount_Clean"].ffill()

# 2. Build feature matrix
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

# 3. Train GBM on all historical data
gbm = GradientBoostingRegressor(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.05,
    random_state=42
)
gbm.fit(X_train, y_train)

# 4. Construct features for September 2026
aug_2026 = ts_df.loc["2026-08-01", "Amount_Clean"]
jul_2026 = ts_df.loc["2026-07-01", "Amount_Clean"]
jun_2026 = ts_df.loc["2026-06-01", "Amount_Clean"]
sep_2025 = ts_df.loc["2025-09-01", "Amount_Clean"]
rolling_3 = np.mean([jun_2026, jul_2026, aug_2026])

sept_features = pd.DataFrame([{
    "lag_1": aug_2026,
    "lag_2": jul_2026,
    "lag_12": sep_2025,
    "rolling_mean_3": rolling_3,
    "month": 9,
    "quarter": 3
}])

# 5. Predict & Compare
gbm_pred = gbm.predict(sept_features)[0]
actual_bill = 116.95
sarima_pred = 102.33

print("\n" + "=" * 52)
print("        SEPTEMBER 2026 BENCHMARK VERDICT")
print("=" * 52)
print(f"Actual Utility Bill:         ${actual_bill:.2f}")
print("-" * 52)
print(f"SARIMA Prediction:           ${sarima_pred:.2f}")
print(f"  SARIMA Error:              ${abs(actual_bill - sarima_pred):.2f} ({abs(actual_bill - sarima_pred)/actual_bill*100:.2f}%)")
print("-" * 52)
print(f"Gradient Boosting (GBM):     ${gbm_pred:.2f}")
print(f"  GBM Error:                 ${abs(actual_bill - gbm_pred):.2f} ({abs(actual_bill - gbm_pred)/actual_bill*100:.2f}%)")
print("=" * 52 + "\n")