import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the raw data
csv_file = "utility_transactions_clean.csv"  # change name if yours is different
df = pd.read_csv(csv_file)

# Strip any whitespace around headers
df.columns = df.columns.str.strip()

print("--- Transaction Types Found ---")
print(df["Type"].value_counts())

# 2. Currency cleaner: handles '$', commas, and negative credits like '($106.40)'
def clean_amount(val):
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace("$", "").replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    return float(s)

df["Amount_Clean"] = df["Amount"].apply(clean_amount)
df["Date"] = pd.to_datetime(df["Date"], format="mixed")

# 3. Filter strictly for Invoices (actual monthly utility usage charges)
invoices = df[df["Type"].str.lower() == "invoice"].copy()
invoices = invoices.sort_values("Date").reset_index(drop=True)

# 4. Aggregate by month (in case two charges hit the same month)
invoices["YearMonth"] = invoices["Date"].dt.to_period("M")
monthly = invoices.groupby("YearMonth")["Amount_Clean"].sum().reset_index()
monthly["Date"] = monthly["YearMonth"].dt.to_timestamp()

print("\n--- Cleaned Monthly Data (First 5 Rows) ---")
print(monthly.head())
print(f"\nTotal monthly billing points: {len(monthly)}")

# 5. Save the processed data for modeling
monthly.to_csv("monthly_invoices.csv", index=False)
print("Saved clean data to 'monthly_invoices.csv'")

# 6. Plot the time-series history
plt.figure(figsize=(10, 5))
plt.plot(monthly["Date"], monthly["Amount_Clean"], marker="o", linestyle="-", color="#1f77b4")
plt.title("Monthly Utility Invoice Charges (2022 - 2026)")
plt.xlabel("Date")
plt.ylabel("Amount ($)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()