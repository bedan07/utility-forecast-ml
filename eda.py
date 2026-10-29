import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the clean monthly dataset
df = pd.read_csv("monthly_invoices.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month_name()
df["Year"] = df["Date"].dt.year

# 2. Print Summary Statistics
print("--- Summary Statistics ---")
print(df["Amount_Clean"].describe().round(2))

# 3. Visualization: Time Series, Seasonality, and Distribution
fig, axes = plt.subplots(3, 1, figsize=(11, 12))

# Plot A: Chronological Time Series
axes[0].plot(df["Date"], df["Amount_Clean"], marker="o", color="#1f77b4", linewidth=2)
axes[0].set_title("1. Chronological Monthly Invoices (2022 - 2026)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Bill Amount ($)")
axes[0].grid(True, linestyle="--", alpha=0.5)

# Plot B: Monthly Seasonality (Grouped by Month across all years)
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
sns.boxplot(ax=axes[1], x="Month", y="Amount_Clean", data=df, order=month_order, palette="Blues")
axes[1].set_title("2. Seasonal Distribution by Calendar Month", fontsize=12, fontweight="bold")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)
axes[1].set_ylabel("Amount ($)")
axes[1].grid(True, linestyle="--", alpha=0.5)

# Plot C: Distribution / Density (Detecting Multimodal vs Normal Distribution)
sns.histplot(data=df, x="Amount_Clean", ax=axes[2], kde=True, color="#2ca02c", bins=12)
axes[2].set_title("3. Target Variable Distribution (Amount)", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Bill Amount ($)")
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()