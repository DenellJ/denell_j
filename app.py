# app.py — Inventory Planner & SARIMA Forecaster (per SKU) with robust matching
# Uses static hard-coded history (static_history.csv.gz) built once from your 5-year Excel.
# Denell-ready: per-SKU SARIMA, Min/Max/ROP/EOQ, 1–120 mo horizon, batch Excel, per-item PDF.

import io
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import norm

# Optional: Auto-ARIMA (pmdarima). App works fine without it.
try:
    from pmdarima import auto_arima
    HAS_PM = True
except Exception:
    HAS_PM = False

# ====== CONFIG ======
APP_DIR = Path(__file__).parent
STATIC_HISTORY = APP_DIR / "static_history.csv.gz"  # created by build_static_history.py

DEFAULT_REORDER = r"C:\Users\Denell\Desktop\Demand Forecasting and Inventory Management\Reordering Report\Reordering Report by Item Range formatted for Excel.xlsx"

# ====== UI ======
st.set_page_config(page_title="Inventory Planner & Forecast (Static History)", layout="wide")
st.title("📦 Inventory Planner + 📈 SARIMA Forecaster (Static History)")

left, right = st.columns([2, 1])

with right:
    st.subheader("Settings")
    debug = st.toggle("🔎 Debug mode", value=False)
    use_auto_arima = st.toggle("Auto-ARIMA (pmdarima)", value=(HAS_PM and True))

    seasonal_periods = st.number_input("Seasonal Periods (months)", min_value=0, value=12, step=1)
    diff_order = st.number_input("Difference order (d)", min_value=0, value=1, step=1)
    seasonal_diff = st.number_input("Seasonal difference (D)", min_value=0, value=1, step=1)
    horizon = st.slider("Forecast horizon (months)", 1, 120, 24, 1)

    st.markdown("---")
    st.subheader("Inventory Parameters")
    lead_time_months = st.number_input("Lead Time (months)", min_value=0.0, value=2.0, step=0.5)
    review_period_months = st.number_input("Review Period (months)", min_value=0.0, value=1.0, step=0.5)
    service_level = st.slider("Service Level (%)", 50, 99, 95, 1)
    ordering_cost = st.number_input("Ordering Cost per Order (S)", min_value=0.0, value=250.0, step=50.0)
    holding_cost = st.number_input("Holding Cost per Unit per Year (H)", min_value=0.0, value=15.0, step=1.0)

with left:
    st.subheader("Data Sources")
    if STATIC_HISTORY.exists():
        st.success(f"Static history found: {STATIC_HISTORY.name}")
    else:
        st.error("static_history.csv.gz not found next to app.py. Run build_static_history.py first.")
        st.stop()

    use_default_reorder = st.toggle("Use default Reorder Report path", value=True)
    if use_default_reorder:
        reorder_path = st.text_input("Reorder Excel path", value=DEFAULT_REORDER)
        upload_file = None
    else:
        upload_file = st.file_uploader("Upload Reorder Report (Excel)", type=["xlsx", "xls"])
        reorder_path = None

# ====== LOADERS ======
@st.cache_data(show_spinner=True)
def load_static_history(static_csv_gz: Path) -> pd.DataFrame:
    df = pd.read_csv(static_csv_gz, parse_dates=["Date"])
    # enforce schema
    return df.loc[:, ["ItemCode", "ItemName", "Date", "Quantity"]]

@st.cache_data(show_spinner=True)
def load_reorder(reorder_path: str = None, upload=None) -> pd.DataFrame:
    if upload is not None:
        xl = pd.ExcelFile(upload)
    else:
        p = Path(reorder_path)
        if not p.exists():
            raise FileNotFoundError(f"Reorder path not found: {reorder_path}")
        xl = pd.ExcelFile(p)
    df = xl.parse(xl.sheet_names[0])
    return df

# Load the static, hard-coded history
hist_long = load_static_history(STATIC_HISTORY)

# Load the reorder report (path or upload)
try:
    reorder_df = load_reorder(reorder_path=reorder_path, upload=upload_file)
    st.success(f"Reorder report loaded. Rows: {len(reorder_df):,}")
except Exception as e:
    st.error(f"Failed to load reorder report: {e}")
    st.stop()

# ====== READY-TO-PASTE MATCHING BLOCK (robust; fixes InvalidIndexError) ======
import re as _re

# 1) Normalize headers: collapse spaces, strip
reorder_df.columns = reorder_df.columns.map(lambda c: _re.sub(r"\s+", " ", str(c)).strip())

# 1b) Ensure UNIQUE headers (suffix duplicates instead of dropping them)
def _ensure_unique_columns(cols):
    seen, out = {}, []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

if reorder_df.columns.duplicated().any():
    reorder_df.columns = _ensure_unique_columns(reorder_df.columns)

# 2) Robust detection of Item Code / Name columns
def _pick_code_col(cols):
    pri = ["item code", "item no.", "item number", "itemno", "item"]
    lower = {c: c.lower() for c in cols}
    # exact priorities
    for target in pri:
        for c in cols:
            if lower[c] == target:
                return c
    # contains-based fallback
    for c in cols:
        lc = lower[c]
        if ("item" in lc and "code" in lc) or ("item" in lc and "no" in lc):
            return c
    # last resort: first column
    return cols[0]

code_col = _pick_code_col(list(reorder_df.columns))
name_candidates = [c for c in reorder_df.columns if ("name" in c.lower() or "desc" in c.lower())]
name_col = name_candidates[0] if name_candidates else None

# 3) Build keys and initial match on Item Code
def _norm_code(x):
    return (str(x) if x is not None else "").strip().upper()

reorder_df["_ItemCodeKey"] = reorder_df[code_col].map(_norm_code)
# drop empty keys
reorder_df = reorder_df[reorder_df["_ItemCodeKey"] != ""].copy()

hist_codes = set(hist_long["ItemCode"].astype(str).str.upper().unique())

matched = reorder_df[reorder_df["_ItemCodeKey"].isin(hist_codes)].copy()
unmatched = reorder_df[~reorder_df["_ItemCodeKey"].isin(hist_codes)].copy()

# 4) Optional name-based fallback (simple normalized-name join)
def _norm_name(x: str) -> str:
    s = "".join(ch for ch in (str(x).upper()) if ch.isalnum() or ch.isspace()).strip()
    return _re.sub(r"\s+", " ", s)  # collapse multiple spaces

if len(unmatched) and name_col:
    hist_name_map = (
        hist_long.groupby("ItemCode")["ItemName"].first()
        .reset_index()
        .assign(_ItemNameKey=lambda d: d["ItemName"].map(_norm_name))
    )
    left = unmatched.copy()
    left["_ItemNameKey"] = left[name_col].map(_norm_name)

    fallback = left.merge(hist_name_map, on="_ItemNameKey", how="left", suffixes=("", "_H"))

    got = fallback[~fallback["ItemCode"].isna()].copy()
    if len(got):
        got["_ItemCodeKey"] = got["ItemCode"].astype(str).str.upper().str.strip()

        # Align columns BEFORE concat to avoid InvalidIndexError
        cols_for_concat = list(reorder_df.columns)  # includes "_ItemCodeKey"
        left_block = matched.reindex(columns=cols_for_concat)
        right_block = got.reindex(columns=cols_for_concat)

        matched = pd.concat([left_block, right_block], ignore_index=True, sort=False)

        # recompute unmatched
        unmatched = fallback[fallback["ItemCode"].isna()].reindex(columns=reorder_df.columns).copy()

st.write(f"Matched by code/name: {len(matched):,} | Still unmatched: {len(unmatched):,}")
if debug and len(unmatched):
    st.warning("Sample unmatched rows (first 10):")
    st.dataframe(unmatched.head(10), use_container_width=True)

# Build SKU dropdown list
sku_list = sorted(matched["_ItemCodeKey"].unique().tolist())

# ====== FORECAST / METRICS ======
def forecast_series(y: pd.Series, horizon: int, seasonal: int, d: int, D: int, use_auto: bool):
    """Return (mean, ci_df) for next horizon months."""
    y = y.clip(lower=0).astype("float64")
    model_fit = None
    if use_auto and HAS_PM:
        try:
            m = max(1, int(seasonal))
            model = auto_arima(
                y,
                seasonal=(seasonal >= 2),
                m=m,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=3, max_q=3, max_P=2, max_Q=2, max_order=None,
                d=None if d is not None else None,
                D=None if D is not None else None,
            )
            order, seas_order = model.order, model.seasonal_order
            model_fit = SARIMAX(
                y, order=order, seasonal_order=seas_order,
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)
        except Exception:
            model_fit = None

    if model_fit is None:
        # Seasonal default or non-seasonal fallback
        if seasonal and seasonal >= 2:
            order = (1, max(1, int(d)), 1)
            seas = (1, max(1, int(D)), 1, int(seasonal))
        else:
            order = (1, 1, 1)
            seas = (0, 0, 0, 0)
        model_fit = SARIMAX(
            y, order=order, seasonal_order=seas,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)

    fc = model_fit.get_forecast(steps=int(horizon))
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    ci.columns = ["lower", "upper"]

    mean = mean.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").clip(lower=0)
    ci = ci.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    return mean, ci

def inventory_metrics(ts: pd.Series, Z: float, lead_m: float, review_m: float, S: float, H: float):
    mavg = float(max(ts.mean(), 0.0))
    mstd = float(max(ts.std(ddof=1), 0.0))
    annual = mavg * 12.0
    safety = max(0.0, Z * mstd * math.sqrt(max(lead_m, 0.0)))
    expected_dlt = mavg * max(lead_m, 0.0)
    rop = expected_dlt + safety
    EOQ = math.sqrt(2.0 * max(annual, 0.0) * max(S, 0.0) / max(H, 1e-9)) if (annual > 0 and S > 0 and H > 0) else 0.0
    max_lvl = rop + EOQ + (mavg * max(review_m, 0.0))
    return {
        "AvgMonthly": mavg,
        "MonthlyStd": mstd,
        "AnnualDemand": annual,
        "SafetyStock": safety,
        "ROP_Min": rop,
        "EOQ": EOQ,
        "MaxLevel": max_lvl,
    }

Z = float(norm.ppf(service_level / 100.0))

# ====== PER-ITEM ANALYSIS ======
st.subheader("Per-item analysis")
if not sku_list:
    st.warning("No matched SKUs. Check Item Code/Name columns in your Reorder Report.")
else:
    sku = st.selectbox("Select an Item Code", sku_list)

    # Pull its 5-year history (hard-coded static base)
    sku_hist = (
        hist_long.loc[hist_long["ItemCode"].astype(str).str.upper() == sku, ["Date", "Quantity"]]
        .set_index("Date")
        .sort_index()
        .asfreq("MS")
    )["Quantity"].interpolate("linear").fillna(method="bfill").fillna(method="ffill")

    enough = len(sku_hist) >= max(24, seasonal_periods * 2 if seasonal_periods else 24)
    fc_mean, fc_ci, im = None, None, None

    if not enough:
        st.warning(f"Not enough history for stable SARIMA (points={len(sku_hist)}). Try a smaller seasonal period or pick another item.")
    else:
        fc_mean, fc_ci = forecast_series(sku_hist, horizon, seasonal_periods, diff_order, seasonal_diff, use_auto_arima)
        im = inventory_metrics(sku_hist, Z, lead_time_months, review_period_months, ordering_cost, holding_cost)

        st.write(f"**Item:** {sku}")
        # Chart: history + forecast + CI
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sku_hist.index, y=sku_hist.values, name="History", mode="lines"))
        fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, name="Forecast", mode="lines"))
        fig.add_trace(
            go.Scatter(
                x=fc_ci.index.tolist() + fc_ci.index[::-1].tolist(),
                y=fc_ci["upper"].tolist() + fc_ci["lower"][::-1].tolist(),
                fill="toself",
                name="95% CI",
                hoverinfo="skip",
                line=dict(width=0),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Chart: policy levels
        fig2 = go.Figure()
        recent = pd.concat([sku_hist.tail(24), fc_mean.head(min(horizon, 24))], axis=0)
        fig2.add_trace(go.Scatter(x=recent.index, y=recent.values, name="Demand", mode="lines"))
        fig2.add_hline(y=im["ROP_Min"], annotation_text=f"Min/ROP ≈ {im['ROP_Min']:.0f}", annotation_position="top left")
        fig2.add_hline(y=im["MaxLevel"], annotation_text=f"Max ≈ {im['MaxLevel']:.0f}", annotation_position="bottom left")
        st.plotly_chart(fig2, use_container_width=True)

        # Metrics table
        met_df = pd.DataFrame(
            [
                ["Avg Monthly Demand", im["AvgMonthly"]],
                ["Monthly Std Dev", im["MonthlyStd"]],
                ["Safety Stock", im["SafetyStock"]],
                ["EOQ", im["EOQ"]],
                ["Reorder Point (Min)", im["ROP_Min"]],
                ["Max Level", im["MaxLevel"]],
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(met_df, use_container_width=True)

# ====== BATCH (entire matched list) ======
st.subheader("Batch: metrics for all matched items")
run_batch = st.button("Run batch")
if run_batch and sku_list:
    results = []
    need_points = max(24, seasonal_periods * 2 if seasonal_periods else 24)
    for code in sku_list:
        series = (
            hist_long.loc[hist_long["ItemCode"].astype(str).str.upper() == code, ["Date", "Quantity"]]
            .set_index("Date")
            .sort_index()
            .asfreq("MS")
        )["Quantity"].interpolate("linear").fillna(method="bfill").fillna(method="ffill")

        if len(series) < need_points:
            results.append(
                {"ItemCode": code, "AvgMonthly": np.nan, "MonthlyStd": np.nan, "SafetyStock": np.nan,
                 "EOQ": np.nan, "ROP_Min": np.nan, "MaxLevel": np.nan}
            )
            continue

        imx = inventory_metrics(series, Z, lead_time_months, review_period_months, ordering_cost, holding_cost)
        results.append({"ItemCode": code, **imx})

    res_df = pd.DataFrame(results)

    # Keep original reorder context + append computed fields
    keep_cols = [c for c in reorder_df.columns if c != "_ItemCodeKey"]
    out = matched.merge(res_df, left_on="_ItemCodeKey", right_on="ItemCode", how="left")
    out = out[keep_cols + ["ItemCode", "AvgMonthly", "MonthlyStd", "SafetyStock", "EOQ", "ROP_Min", "MaxLevel"]]

    st.success(f"Batch complete: {len(out):,} rows.")
    st.dataframe(out.head(50), use_container_width=True)

    # Excel export
    def to_excel_bytes(df: pd.DataFrame):
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)
        bio.seek(0)
        return bio

    xls = to_excel_bytes(out)
    st.download_button(
        "⬇️ Download batch results (Excel)",
        data=xls,
        file_name=f"Inventory_Batch_{datetime.now():%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ====== PDF for selected item ======
st.subheader("PDF report (selected item)")
if 'sku' in locals() and 'im' in locals() and im is not None:
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    def fig_to_png_bytes(make_fig_fn):
        buf = io.BytesIO()
        fig = make_fig_fn()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    def fig_hist_fc():
        f, ax = plt.subplots(figsize=(8, 3))
        ax.plot(sku_hist.index, sku_hist.values, label="History")
        if 'fc_mean' in locals() and fc_mean is not None:
            ax.plot(fc_mean.index, fc_mean.values, label="Forecast")
        ax.legend()
        ax.set_title(f"Item {sku} — History & Forecast")
        return f

    def fig_minmax():
        f, ax = plt.subplots(figsize=(8, 3))
        if 'fc_mean' in locals() and fc_mean is not None:
            recent = pd.concat([sku_hist.tail(24), fc_mean.head(min(horizon, 24))], axis=0)
        else:
            recent = sku_hist.tail(24)
        ax.plot(recent.index, recent.values, label="Demand")
        ax.axhline(im["ROP_Min"], ls="--", label=f"Min/ROP ≈ {im['ROP_Min']:.0f}")
        ax.axhline(im["MaxLevel"], ls="--", label=f"Max ≈ {im['MaxLevel']:.0f}")
        ax.legend()
        ax.set_title("Inventory Policy Levels")
        return f

    img1 = ImageReader(fig_to_png_bytes(fig_hist_fc))
    img2 = ImageReader(fig_to_png_bytes(fig_minmax))

    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    W, H = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, H - 50, f"Inventory Report — {sku}")
    c.setFont("Helvetica", 10)

    y = H - 80
    table = [
        ("Avg Monthly Demand", f"{im['AvgMonthly']:.2f}"),
        ("Monthly Std Dev", f"{im['MonthlyStd']:.2f}"),
        ("Safety Stock", f"{im['SafetyStock']:.0f}"),
        ("EOQ", f"{im['EOQ']:.0f}"),
        ("Reorder Point (Min)", f"{im['ROP_Min']:.0f}"),
        ("Max Level", f"{im['MaxLevel']:.0f}"),
    ]
    for label, val in table:
        c.drawString(40, y, f"{label}: {val}")
        y -= 14

    c.drawImage(img1, 40, y - 220, width=W - 80, height=180, preserveAspectRatio=True, mask="auto")
    y -= 240
    c.drawImage(img2, 40, y - 220, width=W - 80, height=180, preserveAspectRatio=True, mask="auto")

    c.showPage()
    c.save()
    pdf_buf.seek(0)

    st.download_button(
        "⬇️ Download PDF (this item)",
        data=pdf_buf.getvalue(),
        file_name=f"Inventory_{sku}_{datetime.now():%Y%m%d}.pdf",
        mime="application/pdf",
    )

# ====== Notes ======
with st.expander("Notes & Assumptions"):
    st.markdown(
        """
- **Static history** (5 years) is read from `static_history.csv.gz` next to this script. Build it once with `build_static_history.py`.
- **Per-SKU** forecast uses that SKU’s own monthly history (aggregated at month-start).
- **SARIMA**: Auto-ARIMA when available; otherwise defaults to a robust seasonal or non-seasonal model.
- **EOQ**: `sqrt(2*D*S/H)` where D is annualized demand from history.
- **Safety Stock**: `Z × σ_monthly × √(LeadTimeMonths)`.
- **Min/Max policy**: `Min=ROP = AvgMonthly×LeadTime + Safety; Max = ROP + EOQ + AvgMonthly×ReviewPeriod`.
- Adjust **service level, lead time, costs, and horizon** in the right panel.
        """
    )
# End of app.py