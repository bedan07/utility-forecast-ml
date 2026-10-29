import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load data and set standard monthly frequency
df = pd.read_csv("monthly_invoices.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Resample/ensure regular monthly start index
ts = df.set_index("Date")["Amount_Clean"].asfreq("MS")

# If any single month was missing in index, forward fill baseline
ts = ts.ffill()

# 2. Train / Test Split (Hold out the last 6 months for evaluation)
train = ts.iloc[:-6]
test = ts.iloc[-6:]

print(f"Training months: {len(train)} ({train.index[0].date()} to {train.index[-1].date()})")
print(f"Testing months:  {len(test)}  ({test.index[0].date()} to {test.index[-1].date()})")

# 3. Fit SARIMA with auto-tuning (m=12 for annual seasonality)
model = auto_arima(
    train,
    seasonal=True,
    m=12,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)

print("\n--- Best Model Summary ---")
print(model.summary())

# 4. Generate out-of-sample forecast
forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
forecast = pd.Series(forecast, index=test.index)

# 5. Evaluate Metrics
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print(f"\n--- Evaluation on Test Split ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# 6. Plot Forecast vs Actual
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label="Training Data", color="#1f77b4")
plt.plot(test.index, test, label="Actual (Ground Truth)", color="#2ca02c", marker="o")
plt.plot(forecast.index, forecast, label="SARIMA Forecast", color="#d62728", linestyle="--", marker="x")
plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color="#d62728", alpha=0.15, label="95% Confidence Interval")
plt.title(f"SARIMA Out-of-Sample Forecast | MAE: ${mae:.2f}")
plt.xlabel("Date")
plt.ylabel("Amount ($)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()