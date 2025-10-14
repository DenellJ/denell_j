# build_static_history.py
# One-time converter: Excel (wide months) -> long monthly CSV.GZ (static, “hard-coded” asset)
# Input (unchanged): your original 5-year history Excel
# Output: static_history.csv.gz (bundled with the app and used forever)

import re
import pandas as pd
from pathlib import Path

# === EDIT THIS TO YOUR REAL FILE ONCE ===
HIST_XLSX = r"C:\Users\Denell\Desktop\Demand Forecasting and Inventory Management\Static Sales Quantity Jan 2020 to July 2025\Sales Quantity - 5 Years.xlsx"

OUT_DIR = Path(__file__).parent
OUT_FILE = OUT_DIR / "static_history.csv.gz"

print(f"Reading: {HIST_XLSX}")
df = pd.read_excel(HIST_XLSX, sheet_name="Sales Quantity - 5 Years")

# Expect columns: ["Item No.", "Item Description", "January  (2020) - Quantity", ..., "July  (2025) - Quantity"]
id_cols = ["Item No.", "Item Description"]
month_cols = [c for c in df.columns if c.endswith("- Quantity")]

long_df = df.melt(id_vars=id_cols, value_vars=month_cols, var_name="MonthLabel", value_name="Quantity")

# Parse “January  (2020) - Quantity”
m = long_df["MonthLabel"].str.extract(r"^(?P<mon>[A-Za-z]+)\s+\((?P<year>\d{4})\)\s+-\s+Quantity$")
month_to_num = {m: i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], start=1)}

long_df["Year"]  = pd.to_numeric(m["year"], errors="coerce")
long_df["Month"] = m["mon"].map(month_to_num)
long_df = long_df.dropna(subset=["Year","Month"]).copy()
long_df["Year"]  = long_df["Year"].astype(int)
long_df["Month"] = long_df["Month"].astype(int)
long_df["Date"]  = pd.to_datetime(dict(year=long_df["Year"], month=long_df["Month"], day=1))

long_df["Quantity"] = pd.to_numeric(long_df["Quantity"], errors="coerce").fillna(0)

long_df = (long_df
           .rename(columns={"Item No.":"ItemCode", "Item Description":"ItemName"})
           .loc[:, ["ItemCode","ItemName","Date","Quantity"]]
           .sort_values(["ItemCode","Date"])
          )

OUT_DIR.mkdir(parents=True, exist_ok=True)
long_df.to_csv(OUT_FILE, index=False, compression="gzip")
print(f"Wrote static asset: {OUT_FILE} rows={len(long_df):,}")
