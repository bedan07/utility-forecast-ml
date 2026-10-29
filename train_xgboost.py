import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load clean monthly data
df = pd.read_csv("monthly_invoices.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Ensure full chronological monthly index
ts_df = df.set_index("Date").asfreq("MS")
ts_df["Amount_Clean"] = ts_df["Amount_Clean"].ffill()

# 2. Feature Engineering
data = pd.DataFrame(index=ts_df.index)
data["y"] = ts_df["Amount_Clean"]

# Lag features
data["lag_1"] = data["y"].shift(1)
data["lag_2"] = data["y"].shift(2)
data["lag_12"] = data["y"].shift(12)

# Rolling features
data["rolling_mean_3"] = data["y"].shift(1).rolling(window=3).mean()

# Calendar features
data["month"] = data.index.month
data["quarter"] = data.index.quarter

# Drop the first 12 rows that contain NaN values from lag_12
data = data.dropna()

# 3. Train / Test Split (Hold out the last 6 months)
features = ["lag_1", "lag_2", "lag_12", "rolling_mean_3", "month", "quarter"]
X = data[features]
y = data["y"]

X_train, X_test = X.iloc[:-6], X.iloc[-6:]
y_train, y_test = y.iloc[:-6], y.iloc[-6:]

print(f"XGBoost/GBM Training samples: {len(X_train)}")
print(f"XGBoost/GBM Testing samples:  {len(X_test)}")

# 4. Train Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- Gradient Boosting Test Evaluation ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# 6. Feature Importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Feature Importances ---")
print(importances.round(3))

# 7. Plot Actual vs Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_train.index, y_train, label="Train", color="#1f77b4")
plt.plot(y_test.index, y_test, label="Actual (Ground Truth)", color="#2ca02c", marker="o")
plt.plot(y_test.index, y_pred, label=f"GBM Forecast (MAE: ${mae:.2f})", color="#ff7f0e", linestyle="--", marker="s")
plt.title(f"Gradient Boosting Regressor Out-of-Sample Evaluation")
plt.xlabel("Date")
plt.ylabel("Amount ($)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()