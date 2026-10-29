import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# 1. Load the full clean dataset
df = pd.read_csv("monthly_invoices.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Resample to full monthly frequency and fill any gaps
ts = df.set_index("Date")["Amount_Clean"].asfreq("MS").ffill()

print(f"Training on all {len(ts)} historical months ({ts.index[0].date()} to {ts.index[-1].date()})")

# 2. Fit the best model architecture on all data
# (SARIMA with 12-month annual seasonality)
model = auto_arima(
    ts,
    seasonal=True,
    m=12,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)

print("\nModel trained successfully:")
print(model)

# 3. Forecast the next 6 future months
n_months = 6
forecast_values, conf_int = model.predict(n_periods=n_months, return_conf_int=True)

# 4. Generate future monthly timestamps
future_dates = pd.date_range(
    start=ts.index[-1] + pd.DateOffset(months=1),
    periods=n_months,
    freq="MS"
)

# Build a clean results dataframe
forecast_df = pd.DataFrame({
    "Forecast_Amount": forecast_values.round(2),
    "Lower_95%_Bound": conf_int[:, 0].round(2),
    "Upper_95%_Bound": conf_int[:, 1].round(2)
}, index=future_dates)

print(f"\n--- Next {n_months} Months Forecast ---")
print(forecast_df)

# 5. Plot Historical + Future Projection
plt.figure(figsize=(11, 5))
plt.plot(ts.index, ts, label="Historical Actuals", color="#1f77b4", linewidth=2)
plt.plot(forecast_df.index, forecast_df["Forecast_Amount"], label="Future Forecast", color="#d62728", marker="o", linestyle="--")
plt.fill_between(
    forecast_df.index,
    forecast_df["Lower_95%_Bound"],
    forecast_df["Upper_95%_Bound"],
    color="#d62728",
    alpha=0.15,
    label="95% Confidence Interval"
)

plt.title(f"Utility Invoice Projection (Next {n_months} Months)", fontsize=12, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Amount ($)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()