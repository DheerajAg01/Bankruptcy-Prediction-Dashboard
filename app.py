# app.py — Bankruptcy Prediction Dashboard (Single File, Fixed & Polished UI)
# Run:
#    pip install --upgrade pip
#    pip install streamlit yfinance scikit-learn plotly xgboost pandas numpy alpha_vantage pandas-datareader
#    streamlit run app.py

import re
import time # Re-added time, as it's used in fetch_yahoo_financials
import math
from datetime import datetime, timedelta
from io import StringIO
import os

import streamlit as st
import pandas as pd
import numpy as np

# 3rd party libs (optional guards)
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from alpha_vantage.timeseries import TimeSeries
except Exception:
    TimeSeries = None

try:
    import pandas_datareader.data as web  # optional tertiary fallback (Stooq)
except Exception:
    web = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# -------------------------------
# PAGE CONFIG (must be first UI call)
# -------------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# GLOBAL THEME / CSS (clean dark)
# -------------------------------
PRO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root{
  --primary-bg:#0b0f2a; --secondary-bg:#12183a; --card-bg:rgba(20,27,61,.88);
  --card-border:rgba(99,102,241,.25); --text:#f8fafc; --muted:#a3aed0;
  --accent:#6366f1; --good:#10b981; --warn:#f59e0b; --bad:#ef4444;
}
*{font-family:'Inter',sans-serif}
html,body,[data-testid="stAppViewContainer"]{
  background:radial-gradient(1200px 800px at 15% 10%, #101744 0%, #0b0f2a 35%, #080b1e 100%) fixed;
}

/* tighten base spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }

/* Tabs spacing */
[data-baseweb="tab-list"] { gap: 6px; }

/* Cards */
.card{background:var(--card-bg);border:1px solid var(--card-border);border-radius:16px;padding:16px}
.card-tight{background:var(--card-bg);border:1px solid var(--card-border);border-radius:14px;padding:12px}
.metric-card{
  background:linear-gradient(135deg,rgba(99,102,241,.12),rgba(15,23,42,.85));
  border-radius:12px;padding:12px;text-align:center;border:1px solid rgba(99,102,241,.18)
}
.metric-label{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.metric-value{font-size:1.6rem;font-weight:700;color:var(--text)}
.small{font-size:.86rem;color:var(--muted)}
.badge{display:inline-block;padding:6px 10px;border-radius:8px;color:#fff;font-weight:600}
.badge.safe{background:var(--good)}.badge.gray{background:var(--warn)}.badge.distress{background:var(--bad)}
.progress-container{height:12px;background:rgba(255,255,255,.06);border-radius:8px;overflow:hidden}
.progress-bar.safe{background:var(--good);height:100%}.progress-bar.warning{background:var(--warn);height:100%}.progress-bar.danger{background:var(--bad);height:100%}

/* Buttons polish */
button[kind="secondary"] { border:1px solid rgba(99,102,241,.35) !important }
.stButton>button { border-radius:10px; padding:.55rem .8rem }

/* DataFrame dark styling */
[data-testid="stDataFrame"] div, [data-testid="stTable"] div { color: var(--text) !important; }
[data-testid="stDataFrame"] { background: transparent !important; }
[data-testid="stDataFrame"] [class*="row_heading"], 
[data-testid="stDataFrame"] [class*="blank"] { background: rgba(15,23,42,.5) !important; }
[data-testid="stDataFrame"] [class*="column_heading"] { background: rgba(15,23,42,.7) !important; }

/* Form labels alignment */
.css-ocqkz7, .stTextInput label, .stSelectbox label { color: var(--muted) !important }
</style>
"""
st.markdown(PRO_CSS, unsafe_allow_html=True)

# -------------------------------
# SESSION
# -------------------------------
if "industry" not in st.session_state:
    st.session_state.industry = {}
if "history" not in st.session_state:
    st.session_state.history = []

# Alpha Vantage key (env var supported)
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY", "J9EGBEZLO30NBACU")

# -------------------------------
# UTILITIES
# -------------------------------
def normalize_tickers(raw: str):
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    for p in parts:
        t = re.sub(r"[^A-Za-z0-9\.\-]", "", p)
        if "." in t:
            a, b = t.split(".", 1)
            t = a.upper() + "." + b.upper()
        else:
            t = t.upper()
        out.append(t)
    return out

def fmt_curr(v):
    try:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "-"
        v = float(v)
        if abs(v) >= 1e12:
            return f"${v/1e12:.2f}T"
        if abs(v) >= 1e9:
            return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v/1e6:.2f}M"
        if abs(v) >= 1e3:
            return f"${v/1e3:.2f}K"
        return f"${v:.2f}"
    except Exception:
        return str(v)

def _safe_den(v, floor=1.0):
    try:
        x = float(v)
        return max(abs(x), floor)
    except Exception:
        return floor

def _first_col(df: pd.DataFrame) -> pd.Series:
    try:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df.iloc[:, 0]
    except Exception:
        return pd.Series(dtype=float)

def _get_with_candidates(s: pd.Series, names) -> float:
    if s is None or len(s) == 0:
        return 0.0
    for nm in names:
        if nm in s.index and pd.notna(s.loc[nm]):
            try:
                return float(s.loc[nm])
            except Exception:
                pass
    return 0.0

# -------------------------------
# YFINANCE FETCHERS (robust)
# -------------------------------
def _first_nonempty_df(obj, attr_names):
    for nm in attr_names:
        try:
            df = getattr(obj, nm, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60 * 15)
def fetch_yahoo_financials(ticker: str, prefer: str = "auto", retries: int = 2, pause: float = 0.8):
    if yf is None:
        raise RuntimeError("yfinance is not installed. Add it to requirements and redeploy.")
    tkr = yf.Ticker(ticker)
    last_err = None
    for _ in range(retries + 1):
        try:
            try:
                info = tkr.info or {}
            except Exception:
                info = {}
            inc_annual = _first_nonempty_df(tkr, ["income_stmt", "financials"])
            bs_annual = _first_nonempty_df(tkr, ["balance_sheet"])
            cf_annual = _first_nonempty_df(tkr, ["cashflow"])
            inc_quarter = _first_nonempty_df(tkr, ["quarterly_income_stmt", "quarterly_financials"])
            bs_quarter = _first_nonempty_df(tkr, ["quarterly_balance_sheet"])
            cf_quarter = _first_nonempty_df(tkr, ["quarterly_cashflow"])

            if prefer == "annual":
                inc_df = inc_annual if not inc_annual.empty else inc_quarter
                bs_df = bs_annual if not bs_annual.empty else bs_quarter
                cf_df = cf_annual if not cf_annual.empty else cf_quarter
            elif prefer == "quarterly":
                inc_df = inc_quarter if not inc_quarter.empty else inc_annual
                bs_df = bs_quarter if not bs_quarter.empty else bs_annual
                cf_df = cf_quarter if not cf_quarter.empty else cf_annual
            else:
                def pick_first(a, q): return a if (isinstance(a, pd.DataFrame) and not a.empty) else q
                inc_df = pick_first(inc_annual, inc_quarter)
                bs_df = pick_first(bs_annual, bs_quarter)
                cf_df = pick_first(cf_annual, cf_quarter)

            if (inc_df is None or inc_df.empty) and (bs_df is None or bs_df.empty):
                raise ValueError("No financial statements available from Yahoo for this ticker.")

            return {"info": info, "bs": _first_col(bs_df), "inc": _first_col(inc_df), "cf": _first_col(cf_df)}
        except Exception as e:
            last_err = e
            time.sleep(pause)
    raise RuntimeError(f"Yahoo Finance fetch failed for {ticker}: {last_err}")

@st.cache_data(show_spinner=False, ttl=60 * 15)
def fetch_price_history(ticker: str, years: int = 5) -> pd.DataFrame:
    """Robust price fetch using Yahoo Finance with Alpha Vantage fallback (and optional Stooq)."""
    end = datetime.today()
    start = end - timedelta(days=365 * years)

    # 1) Try Yahoo (Primary Source)
    try:
        if yf is not None:
            # IMPORTANT FIX: Added multi_level_index=False for greater yfinance stability
            df = yf.download(
                ticker, 
                period=f"{years}y", 
                interval="1d", 
                auto_adjust=True, 
                progress=False,
                multi_level_index=False
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.reset_index()
                if "Date" not in df.columns and "Datetime" in df.columns:
                    df.rename(columns={"Datetime": "Date"}, inplace=True)
                elif "Date" not in df.columns:
                    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                if df["Close"].dropna().shape[0] >= 5:
                    return df
    except Exception:
        pass

    # 2) Fallback Alpha Vantage (Requires API Key)
    try:
        if TimeSeries is not None and ALPHA_KEY:
            ts = TimeSeries(key=ALPHA_KEY, output_format="pandas")
            df, _ = ts.get_daily_adjusted(symbol=ticker, outputsize="full")

            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",
                "6. volume": "Volume"
            })
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.sort_index().reset_index().rename(columns={"index": "Date"})
            for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[df["Date"].between(start, end)]
            if ("Close" in df and df["Close"].dropna().shape[0] >= 5) or \
               ("Adj Close" in df and df["Adj Close"].dropna().shape[0] >= 5):
                return df
    except Exception:
        pass

    # 3) Optional Stooq fallback (no key)
    try:
        if web is not None:
            df = web.DataReader(ticker, "stooq", start, end)
            df = df.sort_index().reset_index()
            df.rename(columns={"Close": "Adj Close", "Date": "Date"}, inplace=True)
            for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
    except Exception:
        pass

    st.warning(f"📊 Price history not available for this ticker via Yahoo/AlphaVantage/Stooq in this environment.")
    return pd.DataFrame()

# -------------------------------
# DATA & RATIOS
# -------------------------------
def collect_financial_data(ticker: str, prefer="auto") -> dict:
    raw = fetch_yahoo_financials(ticker, prefer=prefer)
    info, bs, inc, cf = raw["info"], raw["bs"], raw["inc"], raw["cf"]

    company_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    market_cap = info.get("marketCap", 0.0)
    current_price = info.get("currentPrice", info.get("previousClose", 0.0))

    return {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "market_cap": float(market_cap or 0.0),
        "stock_price": float(current_price or 0.0),

        "total_assets": _get_with_candidates(bs, ["Total Assets", "TotalAssets"]),
        "current_assets": _get_with_candidates(bs, ["Current Assets", "CurrentAssets"]),
        "cash": _get_with_candidates(bs, ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash"]),
        "accounts_receivable": _get_with_candidates(bs, ["Accounts Receivable", "AccountsReceivable", "Receivables"]),
        "inventory": _get_with_candidates(bs, ["Inventory"]),
        "total_liabilities": _get_with_candidates(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilitiesNetMinorityInterest"]),
        "current_liabilities": _get_with_candidates(bs, ["Current Liabilities", "CurrentLiabilities"]),
        "long_term_debt": _get_with_candidates(bs, ["Long Term Debt", "LongTermDebt"]),
        "total_debt": _get_with_candidates(bs, ["Total Debt", "TotalDebt"]),
        "retained_earnings": _get_with_candidates(bs, ["Retained Earnings", "RetainedEarnings"]),
        "stockholders_equity": _get_with_candidates(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest", "StockholdersEquity", "TotalEquityGrossMinorityInterest"]),

        "total_revenue": _get_with_candidates(inc, ["Total Revenue", "TotalRevenue"]),
        "gross_profit": _get_with_candidates(inc, ["Gross Profit", "GrossProfit"]),
        "operating_income": _get_with_candidates(inc, ["Operating Income", "OperatingIncome"]),
        "ebit": _get_with_candidates(inc, ["EBIT", "Operating Income", "OperatingIncome"]),
        "ebitda": _get_with_candidates(inc, ["EBITDA"]),
        "net_income": _get_with_candidates(inc, ["Net Income", "NetIncome"]),
        "interest_expense": abs(_get_with_candidates(inc, ["Interest Expense", "InterestExpense"])) or 1.0,

        "operating_cashflow": _get_with_candidates(cf, ["Operating Cash Flow", "OperatingCashFlow", "Total Cash From Operating Activities"]),
        "capex": abs(_get_with_candidates(cf, ["Capital Expenditure", "CapitalExpenditures", "Capital Expenditures"])),
        "free_cashflow": _get_with_candidates(cf, ["Free Cash Flow", "FreeCashFlow"]),
    }

def compute_ratios(d: dict) -> dict:
    TA = _safe_den(d.get("total_assets", 0.0))
    CL = _safe_den(d.get("current_liabilities", 0.0))
    TE = _safe_den(d.get("stockholders_equity", 0.0))
    TL = _safe_den(d.get("total_liabilities", 0.0))
    TR = _safe_den(d.get("total_revenue", 0.0))
    WC = (d.get("current_assets", 0.0) or 0.0) - (d.get("current_liabilities", 0.0) or 0.0)

    return {
        "working_capital_to_assets": WC / TA,
        "retained_earnings_to_assets": (d.get("retained_earnings", 0.0) or 0.0) / TA,
        "ebit_to_assets": (d.get("ebit", 0.0) or 0.0) / TA,
        "market_cap_to_liabilities": _safe_den(d.get("market_cap", 0.0)) / TL,
        "sales_to_assets": (d.get("total_revenue", 0.0) or 0.0) / TA,

        "current_ratio": (d.get("current_assets", 0.0) or 0.0) / CL,
        "quick_ratio": ((d.get("current_assets", 0.0) or 0.0) - (d.get("inventory", 0.0) or 0.0)) / CL,
        "cash_ratio": (d.get("cash", 0.0) or 0.0) / CL,

        "debt_to_equity": TL / TE,
        "debt_to_assets": TL / TA,
        "long_term_debt_to_equity": (d.get("long_term_debt", 0.0) or 0.0) / TE,
        "interest_coverage": (d.get("ebit", 0.0) or 0.0) / _safe_den(d.get("interest_expense", 1.0)),

        "return_on_assets": ((d.get("net_income", 0.0) or 0.0) / TA) * 100,
        "return_on_equity": ((d.get("net_income", 0.0) or 0.0) / TE) * 100,
        "profit_margin": ((d.get("net_income", 0.0) or 0.0) / TR) * 100,
        "gross_margin": ((d.get("gross_profit", 0.0) or 0.0) / TR) * 100,
        "operating_margin": ((d.get("operating_income", 0.0) or 0.0) / TR) * 100,
        "ebitda_margin": ((d.get("ebitda", 0.0) or 0.0) / TR) * 100,

        "asset_turnover": (d.get("total_revenue", 0.0) or 0.0) / TA,
        "receivables_turnover": (d.get("total_revenue", 0.0) or 0.0) / _safe_den(d.get("accounts_receivable", 0.0)),
        "inventory_turnover": (d.get("total_revenue", 0.0) or 0.0) / _safe_den(d.get("inventory", 0.0)),

        "operating_cashflow_to_sales": (d.get("operating_cashflow", 0.0) or 0.0) / TR,
        "free_cashflow_to_equity": (d.get("free_cashflow", 0.0) or 0.0) / TE,
        "capex_to_revenue": (d.get("capex", 0.0) or 0.0) / TR,
    }

def altman_z(r: dict):
    x1 = 1.2 * r["working_capital_to_assets"]
    x2 = 1.4 * r["retained_earnings_to_assets"]
    x3 = 3.3 * r["ebit_to_assets"]
    x4 = 0.6 * r["market_cap_to_liabilities"]
    x5 = 1.0 * r["sales_to_assets"]
    z = x1 + x2 + x3 + x4 + x5
    components = {
        "X1 Working Capital": round(x1, 3),
        "X2 Retained Earnings": round(x2, 3),
        "X3 EBIT": round(x3, 3),
        "X4 Market Value": round(x4, 3),
        "X5 Sales": round(x5, 3),
    }
    return float(z), components

def z_classify(z):
    if z > 2.99:
        return "Safe", "Low", 0
    if z >= 1.81:
        return "Gray Zone", "Medium", 1
    return "Distress", "High", 2

def z_probability(z):
    if z > 2.99:
        p = max(1, min(15, 100 - (z * 20)))
    elif z >= 1.81:
        p = 20 + ((2.99 - z) / 1.18 * 40)
    else:
        p = 60 + ((1.81 - max(z, 0)) * 20)
    return round(min(99, max(1, p)), 2)

# -------------------------------
# ML (synthetic training)
# -------------------------------
FEATURES = [
    "working_capital_to_assets", "retained_earnings_to_assets", "ebit_to_assets", "market_cap_to_liabilities", "sales_to_assets",
    "current_ratio", "debt_to_equity", "return_on_assets", "profit_margin", "asset_turnover", "interest_coverage",
    "quick_ratio", "operating_margin", "free_cashflow_to_equity", "debt_to_assets"
]

@st.cache_resource(show_spinner=False)
def build_models():
    np.random.seed(42)
    n = 1000
    nb = n // 3

    bankX = np.random.normal(
        loc=[0.1, 0.05, 0.02, 0.5, 0.8, 0.9, 2.5, -5, -10, 0.5, 1.0, 0.6, -5, -0.1, 0.8],
        scale=[0.1, 0.1, 0.05, 0.3, 0.3, 0.3, 1.0, 5, 10, 0.3, 2.0, 0.3, 5, 0.2, 0.2], size=(nb, 15))
    bankY = np.ones(nb)

    ns = n - nb
    safeX = np.random.normal(
        loc=[0.4, 0.3, 0.15, 2.5, 1.5, 2.0, 1.0, 8, 12, 1.2, 5.0, 1.5, 10, 0.15, 0.4],
        scale=[0.15, 0.15, 0.08, 1.0, 0.5, 0.5, 0.5, 4, 8, 0.4, 3.0, 0.5, 5, 0.1, 0.15], size=(ns, 15))
    safeY = np.zeros(ns)

    X = np.vstack([bankX, safeX])
    y = np.hstack([bankY, safeY])
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    models = {}
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr_s, ytr)
    models["logistic_regression"] = lr

    rf = RandomForestClassifier(n_estimators=180, max_depth=10, random_state=42)
    rf.fit(Xtr_s, ytr)
    models["random_forest"] = rf

    gb = GradientBoostingClassifier(n_estimators=160, random_state=42)
    gb.fit(Xtr_s, ytr)
    models["gradient_boosting"] = gb

    if XGBOOST_AVAILABLE:
        xg = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            eval_metric="logloss"
        )
        xg.fit(Xtr_s, ytr)
        models["xgboost"] = xg

    metrics = {}
    for name, m in models.items():
        yp = m.predict(Xte_s)
        ypr = m.predict_proba(Xte_s)[:, 1]
        metrics[name] = {
            "accuracy": float(accuracy_score(yte, yp)),
            "precision": float(precision_score(yte, yp, zero_division=0)),
            "recall": float(recall_score(yte, yp, zero_division=0)),
            "f1": float(f1_score(yte, yp, zero_division=0)),
            "roc_auc": float(roc_auc_score(yte, ypr)),
            "cm": confusion_matrix(yte, yp).tolist()
        }
    return {"scaler": scaler, "models": models, "metrics": metrics}

def features_from_ratios(r): 
    return np.array([r.get(k, 0.0) for k in FEATURES], dtype=float).reshape(1, -1)

def predict_all(r):
    pack = build_models()
    scaler, models = pack["scaler"], pack["models"]
    Xs = scaler.transform(features_from_ratios(r))
    preds, votes, probs = {}, [], []
    for name, m in models.items():
        p, pr = int(m.predict(Xs)[0]), float(m.predict_proba(Xs)[0, 1])
        preds[name] = {
            "prediction": p,
            "probability_bankrupt": round(pr * 100, 2),
            "probability_safe": round((1 - pr) * 100, 2),
            "risk_label": "High Risk" if p == 1 else "Low Risk"
        }
        votes.append(p)
        probs.append(pr * 100)
    avg = float(np.mean(probs)) if probs else 0.0
    maj = 1 if sum(votes) > len(votes) / 2 else 0
    return {
        "ensemble_prediction": "High Risk" if maj == 1 else "Low Risk",
        "probability_bankrupt": round(avg, 2),
        "probability_safe": round(100 - avg, 2),
        "confidence": round(max(avg, 100 - avg), 2),
        "individual_models": preds,
        "rf_features": models["random_forest"].feature_importances_.tolist() if "random_forest" in models else None
    }

def combined_assessment(z, ml):
    zp = z_probability(z)
    mp = ml["probability_bankrupt"]
    comb = (0.6 * mp) + (0.4 * zp)
    if comb < 20:
        risk, status = ("Low", "Safe")
        rec = "Strong financial health. Both traditional and ML models indicate low bankruptcy risk."
    elif comb < 50:
        risk, status = ("Medium", "Gray Zone")
        rec = "Moderate risk detected. Monitor metrics and consider risk management strategies."
    else:
        risk, status = ("High", "Distress")
        rec = "High bankruptcy risk. Indicators point to distress. Immediate attention required."
    return {
        "combined_probability": round(comb, 2),
        "risk_level": risk,
        "status": status,
        "recommendation": rec,
        "model_agreement": abs(zp - mp) < 20
    }

def record_industry(sector, tk, z, mlp):
    sector = sector or "Unknown"
    st.session_state.industry.setdefault(sector, [])
    st.session_state.industry[sector].append({
        "ticker": tk, "z": float(z), "ml": float(mlp),
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def sector_stats(sector):
    rows = st.session_state.industry.get(sector, [])
    if not rows:
        return None
    z = [r["z"] for r in rows]
    return {
        "n": len(rows),
        "z_mean": round(float(np.mean(z)), 2),
        "z_med": round(float(np.median(z)), 2),
        "z_std": round(float(np.std(z)), 2),
        "z_min": round(float(np.min(z)), 2),
        "z_max": round(float(np.max(z)), 2),
        "risk_dist": {
            "safe": int(sum(1 for x in z if x > 2.99)),
            "gray": int(sum(1 for x in z if 1.81 <= x <= 2.99)),
            "distress": int(sum(1 for x in z if x < 1.81))
        },
        "rows": rows
    }

# -------------------------------
# CSV HELPERS
# -------------------------------
def _csv_from_result(tk, base, ratios, z, zprob, ml, comb):
    out = StringIO()
    out.write("BANKRUPTCY ANALYSIS REPORT\n")
    out.write(f"Company,{base['company_name']} ({tk})\n")
    out.write(f"Sector,{base['sector']}\n")
    out.write(f"Generated,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    out.write("ALTMAN Z-SCORE\n")
    out.write(f"Z-Score,{round(z, 2)}\n")
    out.write(f"Status,{z_classify(z)[0]}\n")
    out.write(f"Bankruptcy Probability,{zprob}%\n\n")
    out.write("ML ENSEMBLE\n")
    out.write(f"Ensemble Prediction,{ml['ensemble_prediction']}\n")
    out.write(f"Bankruptcy Probability,{ml['probability_bankrupt']}%\n")
    out.write(f"Safe Probability,{ml['probability_safe']}%\n")
    out.write(f"Confidence,{ml['confidence']}%\n\n")
    out.write("COMBINED ASSESSMENT\n")
    out.write(f"Combined Probability,{comb['combined_probability']}%\n")
    out.write(f"Risk Level,{comb['risk_level']}\n")
    out.write(f"Status,{comb['status']}\n")
    out.write(f"Model Agreement,{comb['model_agreement']}\n\n")
    out.write("FINANCIAL RATIOS\n")
    rfmt = {k: (round(v, 4) if abs(v) < 100 else round(v, 2)) for k, v in ratios.items()}
    for k, v in rfmt.items():
        out.write(f"{k},{v}\n")
    return out.getvalue().encode("utf-8")

def _csv_from_history_last():
    last = st.session_state.history[-1]
    base = {"company_name": last["company_name"], "sector": last["sector"]}
    return _csv_from_result(last["ticker"], base, last["ratios"],
                            last["altman"]["score"], last["altman"]["prob"],
                            last["ml"], last["combined"])

# -------------------------------
# CHARTS
# -------------------------------
def chart_price(df: pd.DataFrame, ticker: str, mode: str = "auto"):
    if df is None or df.empty or "Date" not in df.columns:
        return None
    df = df.copy().sort_values("Date").reset_index(drop=True)

    for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    try:
        have_ohlc = all(c in df.columns for c in ("Open", "High", "Low", "Close")) and \
                    df[["Open", "High", "Low", "Close"]].dropna().shape[0] >= 5
        if mode == "line":
            have_ohlc = False
        if mode == "candle" and not have_ohlc:
            mode = "line"

        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05,
                            subplot_titles=(f"{ticker} Stock Price", "Volume"))

        if have_ohlc and mode in ("auto", "candle"):
            valid = df.dropna(subset=["Open", "High", "Low", "Close"])
            fig.add_trace(go.Candlestick(
                x=valid["Date"], open=valid["Open"], high=valid["High"],
                low=valid["Low"], close=valid["Close"],
                name="Price", increasing_line_color="#10b981", decreasing_line_color="#ef4444"
            ), row=1, col=1)
        else:
            price_col = "Close" if ("Close" in df.columns and df["Close"].notna().any()) \
                       else "Adj Close" if ("Adj Close" in df.columns and df["Adj Close"].notna().any()) \
                       else None
            if not price_col:
                return None
            valid = df.dropna(subset=[price_col])
            if valid.empty:
                return None
            fig.add_trace(go.Scatter(
                x=valid["Date"], y=valid[price_col], mode="lines", name=price_col,
                line=dict(width=2), fill="tozeroy", fillcolor="rgba(99,102,241,0.12)"
            ), row=1, col=1)

        if "Volume" in df.columns and df["Volume"].notna().any():
            fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", opacity=0.55),
                          row=2, col=1)

        fig.update_layout(template="plotly_dark", height=520, margin=dict(l=20, r=20, t=60, b=20),
                          xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.1)",
                         rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.1)")
        return fig

    except KeyError:
        price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if not price_col or df[price_col].dropna().empty:
            return None
        fig = go.Figure(go.Scatter(x=df["Date"], y=df[price_col], mode="lines", name=price_col))
        fig.update_layout(template="plotly_dark", height=520, margin=dict(l=20, r=20, t=60, b=20),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

def chart_z_components(components: dict):
    df = pd.DataFrame({"Component": list(components.keys()), "Value": list(components.values())})
    colors = ['#6366f1' if v >= 0 else '#ef4444' for v in df['Value']]
    vmax, vmin = float(df["Value"].max()), float(df["Value"].min())
    headroom_top = 1.3 if vmax > 0 else 0.2
    headroom_bot = 0.2 if vmin < 0 else 0.0

    fig = go.Figure([
        go.Bar(
            x=df["Component"], y=df["Value"], marker_color=colors,
            text=df["Value"].round(3), textposition="outside",
            cliponaxis=False
        )
    ])
    fig.update_layout(
        title="Altman Z-Score Components", template="plotly_dark", height=440,
        margin=dict(l=20, r=20, t=100, b=90),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"), showlegend=False,
        uniformtext_minsize=10, uniformtext_mode="hide"
    )
    fig.update_xaxes(showgrid=False, tickangle=-45)
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.1)",
        automargin=True, range=[min(0, vmin) - headroom_bot, vmax * headroom_top]
    )
    return fig

def chart_ratio_radar(r: dict):
    keys = ["current_ratio", "quick_ratio", "cash_ratio", "debt_to_equity", "interest_coverage", "profit_margin", "operating_margin", "asset_turnover"]
    labels = ["Current Ratio", "Quick Ratio", "Cash Ratio", "Debt/Equity", "Interest Coverage", "Profit Margin", "Operating Margin", "Asset Turnover"]
    vals = [float(r.get(k, 0) or 0) for k in keys]
    clipped = []
    for k, v in zip(keys, vals):
        if k in ("debt_to_equity", "interest_coverage"):
            clipped.append(min(v, 10.0))
        elif k in ("profit_margin", "operating_margin"):
            clipped.append(min(max(v, -50), 50))
        else:
            clipped.append(min(v, 5.0))
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=clipped, theta=labels, fill="toself", name="Ratios",
                                  line=dict(color="#6366f1", width=2), fillcolor="rgba(99,102,241,0.3)"))
    fig.update_layout(template="plotly_dark", title="Key Financial Ratios Overview", height=450,
                      polar=dict(radialaxis=dict(visible=True, gridcolor="rgba(99,102,241,0.2)"),
                                 angularaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
                                 bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=80, r=80, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
    return fig

def chart_model_probs(ml: dict):
    rows = [{"Model": k.replace("_", " ").title(),
             "Bankrupt": v["probability_bankrupt"],
             "Safe": v["probability_safe"]}
            for k, v in ml["individual_models"].items()]
    df = pd.DataFrame(rows)
    ymax = float(max(df["Bankrupt"].max(), df["Safe"].max()))
    headroom = max(105.0, ymax * 1.12)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Bankruptcy Risk", x=df["Model"], y=df["Bankrupt"],
        marker_color="#ef4444",
        text=(df["Bankrupt"].round(1).astype(str) + "%"),
        textposition="outside", cliponaxis=False
    ))
    fig.add_trace(go.Bar(
        name="Safe", x=df["Model"], y=df["Safe"],
        marker_color="#10b981",
        text=(df["Safe"].round(1).astype(str) + "%"),
        textposition="outside", cliponaxis=False
    ))

    fig.update_layout(
        title="Model Predictions Comparison", template="plotly_dark", height=420,
        barmode="group",
        margin=dict(l=20, r=20, t=80, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uniformtext_minsize=10, uniformtext_mode="hide"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.1)",
                     range=[0, headroom], automargin=True, title="%")
    return fig

def chart_rf_importance(importances):
    if not importances:
        return None
    df = pd.DataFrame({"Feature": [f.replace("_", " ").title() for f in FEATURES],
                       "Importance": (np.array(importances) * 100.0)}).sort_values("Importance").tail(12)
    colors = ['#6366f1' if i % 2 == 0 else '#8b5cf6' for i in range(len(df))]
    fig = go.Figure([go.Bar(x=df["Importance"], y=df["Feature"], orientation="h", marker_color=colors,
                            text=df["Importance"].round(1).astype(str) + "%", texttemplate="%{text}", textposition="outside",
                            cliponaxis=False)])
    fig.update_layout(title="Random Forest Feature Importance (Top 12)", template="plotly_dark", height=500,
                      margin=dict(l=20, r=20, t=60, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#cbd5e1"), showlegend=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.1)", title="Importance (%)",
                     range=[0, max(df["Importance"]) * 1.2])
    fig.update_yaxes(showgrid=False)
    return fig

def chart_z_gauge(z_score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=z_score, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Altman Z-Score", 'font': {'size': 20, 'color': '#cbd5e1'}},
        number={'font': {'size': 42, 'color': '#f8fafc'}},
        gauge={'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
               'bar': {'color': "#6366f1"},
               'bgcolor': "rgba(20,27,61,.5)", 'borderwidth': 2, 'bordercolor': "rgba(99,102,241,.3)",
               'steps': [{'range': [0, 1.81], 'color': 'rgba(239,68,68,.3)'},
                         {'range': [1.81, 2.99], 'color': 'rgba(245,158,11,.3)'},
                         {'range': [2.99, 5], 'color': 'rgba(16,185,129,.3)'}],
               'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': z_score}}
    ))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=60, b=20),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
    return fig

def chart_sector_risk(stats: dict):
    if not stats or "risk_dist" not in stats:
        return None
    dist = stats["risk_dist"]
    labels = ["Safe", "Gray Zone", "Distress"]
    values = [dist.get("safe", 0), dist.get("gray", 0), dist.get("distress", 0)]
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4,
                                 marker_colors=colors, textinfo='label+percent', textposition='outside')])
    fig.update_layout(title=f"Risk Distribution ({stats['n']} Companies)", template="plotly_dark", height=400,
                      margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#cbd5e1"), showlegend=True)
    return fig

# -------------------------------
# INSIGHTS (compact bullets)
# -------------------------------
def make_insights(base, ratios, z, z_status, z_prob, ml, comb):
    bullets = []
    bullets.append(f"**Overall:** {comb['status']} ({comb['combined_probability']}% risk, confidence {ml['confidence']}%).")
    agree = "aligned" if comb["model_agreement"] else "divergent"
    bullets.append(f"**Models {agree}:** Z-Score ~{z_prob}% vs ML ~{ml['probability_bankrupt']}% bankruptcy probability.")
    cr = ratios.get("current_ratio", 0)
    if cr < 1:
        bullets.append("**Liquidity watch:** Current ratio < 1 indicates short-term pressure.")
    elif cr >= 1.5:
        bullets.append("**Solid liquidity:** Current ratio ≥ 1.5.")
    dte = ratios.get("debt_to_equity", 0)
    if dte > 2:
        bullets.append("**High leverage:** Debt/Equity > 2; monitor refinancing risk.")
    elif dte < 1:
        bullets.append("**Conservative leverage:** Debt/Equity < 1.")
    pm = ratios.get("profit_margin", 0)
    if pm < 0:
        bullets.append("**Negative margin:** Profit margin < 0; profitability drag.")
    elif pm > 10:
        bullets.append("**Healthy margin:** Profit margin > 10%.")
    return bullets[:5]

# =============================================
# ================== UI =======================
# =============================================

# Sidebar (models are ready here)
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    prefer = st.selectbox("📊 Financials Preference", ["auto", "annual", "quarterly"], index=0,
                          help="Choose whether to prefer annual or quarterly statements.")
    price_years = st.select_slider("📈 Price History (years)", options=[1, 2, 3, 5, 10], value=5)
    price_mode = st.radio("📉 Price Chart Style", options=["auto", "candle", "line"], horizontal=True, index=0)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    try:
        pack = build_models()
        st.metric("Active Models", len(pack["models"]))
        st.metric("Training Samples", "1,000")
    except Exception:
        st.warning("Models loading...")

    if st.session_state.history:
        st.markdown("---")
        st.metric("Analyses Run", len(st.session_state.history))
    st.markdown("---")
    st.markdown("<div class='small'>Tip: use exchange suffixes for non-US tickers (e.g., RELIANCE.NS, SHOP.TO, VOD.L).</div>", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="display:flex;flex-direction:column;gap:4px;margin-bottom:10px;">
  <div style="font-size:20px;color:#f8fafc;font-weight:700">🏛️ Bankruptcy Prediction Dashboard</div>
  <div style="color:#cbd5e1">Advanced Financial Analysis Using ML & Altman Z-Score</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🔍 Analyze", "⚖️ Compare", "🏢 Industry", "🤖 Models", "📜 History", "📥 Export"])

# -------------------------------
# ANALYZE TAB
# -------------------------------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Enter Stock Ticker")

    with st.form(key="analyze_form", clear_on_submit=False):
        col_inp, col_btn = st.columns([4, 1])
        with col_inp:
            user_raw = st.text_input(
                "Stock Ticker",
                value="",
                placeholder="e.g., AAPL, RELIANCE.NS, SHOP.TO",
                label_visibility="collapsed",
                help="Enter a single ticker to analyze. Use the Compare tab for multiple tickers."
            )
        with col_btn:
            submitted = st.form_submit_button("🔍 Analyze", use_container_width=True, type="primary")
    st.markdown('<div class="small" style="margin-top:8px;">Quick Select:</div>', unsafe_allow_html=True)
    qcols = st.columns(8)
    selected_quick = None
    for t, col in zip(["AAPL", "MSFT", "TSLA", "AMZN", "META", "GOOGL", "JPM", "BAC"], qcols):
        with col:
            if st.button(t, key=f"quick_{t}", use_container_width=True):
                selected_quick = t

    ticker = selected_quick if selected_quick else (normalize_tickers(user_raw)[0] if user_raw else None)
    st.markdown("</div>", unsafe_allow_html=True)

    if (submitted or selected_quick):
        if not ticker:
            st.warning("Please enter a valid ticker.")
        else:
            with st.spinner(f"🔄 Analyzing {ticker}..."):
                try:
                    base = collect_financial_data(ticker, prefer=prefer)
                    ratios = compute_ratios(base)
                    z, zcomp = altman_z(ratios)
                    z_status, z_risk, _ = z_classify(z)
                    z_prob = z_probability(z)
                    ml = predict_all(ratios)
                    comb = combined_assessment(z, ml)
                    record_industry(base["sector"], ticker, z, ml["probability_bankrupt"])

                    result = {
                        "ticker": ticker,
                        "company_name": base["company_name"],
                        "sector": base["sector"],
                        "industry": base["industry"],
                        "altman": {"score": round(z, 2), "status": z_status, "risk": z_risk, "prob": z_prob, "components": zcomp},
                        "ml": ml,
                        "combined": comb,
                        "ratios": {k: (round(v, 4) if abs(v) < 100 else round(v, 2)) for k, v in ratios.items()},
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.history.append(result)

                    st.success(f"✅ Successfully analyzed {base['company_name']} ({ticker})")
                    st.markdown("---")

                    # Header block
                    c1, c2 = st.columns([1.4, 1])
                    with c1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        left, right = st.columns([3, 1])
                        with left:
                            st.markdown(f"## {base['company_name']}")
                            st.markdown(f"**Ticker:** {ticker} &nbsp;&nbsp; **Sector:** {base['sector']}")
                        with right:
                            csv_bytes = _csv_from_result(
                                ticker,
                                {"company_name": base["company_name"], "sector": base["sector"]},
                                ratios, z, z_prob, ml, comb
                            )
                            st.download_button("📥 Export", data=csv_bytes,
                                               file_name=f"{ticker}_analysis.csv",
                                               mime="text/csv", use_container_width=True)
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f"""<div class="metric-card"><div class="metric-label">Market Cap</div><div class="metric-value">{fmt_curr(base.get('market_cap', 0))}</div></div>""", unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"""<div class="metric-card"><div class="metric-label">Stock Price</div><div class="metric-value">{fmt_curr(base.get('stock_price', 0))}</div></div>""", unsafe_allow_html=True)
                        with m3:
                            ind = base['industry'] if base['industry'] else "—"
                            st.markdown(f"""<div class="metric-card"><div class="metric-label">Industry</div><div class="metric-value" style="font-size:1.0rem">{ind[:20]}</div></div>""", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with c2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("## 🎯 Risk Assessment")
                        badge = "safe" if comb["status"] == "Safe" else ("gray" if comb["status"] == "Gray Zone" else "distress")
                        st.markdown(f'<div class="badge {badge}" style="margin:12px 0;">{comb["status"]}</div>', unsafe_allow_html=True)
                        b1, b2 = st.columns(2)
                        with b1:
                            st.metric("Risk Level", comb["risk_level"])
                            st.metric("Agreement", "✓ Yes" if comb["model_agreement"] else "✗ No")
                        with b2:
                            st.metric("Combined Risk", f"{comb['combined_probability']}%")
                            st.metric("Confidence", f"{ml['confidence']}%")
                        bar = "safe" if comb["combined_probability"] < 20 else ("warning" if comb["combined_probability"] < 50 else "danger")
                        st.markdown(f"""<div class="progress-container" style="margin-top:12px;"><div class="progress-bar {bar}" style="width:{int(comb['combined_probability'])}%"></div></div>""", unsafe_allow_html=True)
                        st.markdown(f"<div class='small' style='margin-top:8px'>{comb['recommendation']}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("---")

                    # Z + Price row
                    m1, m2 = st.columns([1, 2], vertical_alignment="top")
                    with m1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.plotly_chart(chart_z_gauge(z), use_container_width=True, config={"displayModeBar": False})
                        st.markdown(f"<div class='small'><strong>Status:</strong> {z_status}<br><strong>Risk:</strong> {z_risk}<br><strong>Probability:</strong> {z_prob}%</div>", unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("#### ✨ Insights")
                        insights = make_insights(base, ratios, z, z_status, z_prob, ml, comb)
                        for b in insights:
                            st.markdown(f"- {b}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with m2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        price_df = fetch_price_history(ticker, years=price_years)
                        if not price_df.empty:
                            fig_price = chart_price(price_df, ticker, mode=price_mode)
                            if fig_price:
                                st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})
                            else:
                                st.warning("⚠️ Price data available but chart rendering failed.")
                        else:
                            st.info("📊 Price history not available for this ticker via data providers in this environment.")
                            st.markdown("<div class='small'>Tip: try a different exchange suffix (e.g., RELIANCE.NS, SHOP.TO, VOD.L).</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("---")

                    ch1, ch2 = st.columns(2)
                    with ch1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.plotly_chart(chart_z_components(zcomp), use_container_width=True, config={"displayModeBar": False})
                        st.markdown("</div>", unsafe_allow_html=True)
                    with ch2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.plotly_chart(chart_ratio_radar(ratios), use_container_width=True, config={"displayModeBar": False})
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("---")

                    # ML + Feature importance
                    bt1, bt2 = st.columns(2)
                    with bt1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("### 🤖 ML Model Predictions")
                        st.plotly_chart(chart_model_probs(ml), use_container_width=True, config={"displayModeBar": False})
                        mdl_df = pd.DataFrame([{
                            "Model": k.replace("_", " ").title(),
                            "Risk %": f"{v['probability_bankrupt']:.2f}%",
                            "Safe %": f"{v['probability_safe']:.2f}%",
                            "Prediction": v['risk_label']
                        } for k, v in ml["individual_models"].items()])
                        st.dataframe(mdl_df, use_container_width=True, hide_index=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with bt2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        if ml.get("rf_features"):
                            fi_fig = chart_rf_importance(ml["rf_features"])
                            if fi_fig:
                                st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})
                        else:
                            st.info("Random Forest feature importance not available.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("---")

                    # Ratios table stack
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Complete Financial Ratios")
                    ratio_categories = {
                        "Altman Z Components": ["working_capital_to_assets", "retained_earnings_to_assets", "ebit_to_assets", "market_cap_to_liabilities", "sales_to_assets"],
                        "Liquidity Ratios": ["current_ratio", "quick_ratio", "cash_ratio"],
                        "Leverage Ratios": ["debt_to_equity", "debt_to_assets", "long_term_debt_to_equity", "interest_coverage"],
                        "Profitability Ratios": ["return_on_assets", "return_on_equity", "profit_margin", "gross_margin", "operating_margin", "ebitda_margin"],
                        "Efficiency Ratios": ["asset_turnover", "receivables_turnover", "inventory_turnover"],
                        "Cash Flow Ratios": ["operating_cashflow_to_sales", "free_cashflow_to_equity", "capex_to_revenue"]
                    }
                    cols = st.columns(len(ratio_categories))
                    for idx, (category, keys) in enumerate(ratio_categories.items()):
                        with cols[idx]:
                            st.markdown(f"**{category}**")
                            data = [{"Metric": k.replace("_", " ").title(), "Value": result["ratios"][k]} for k in keys if k in result["ratios"]]
                            if data:
                                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True, height=240)
                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.error(f"❌ Analysis failed for '{ticker}'")
                    st.exception(e)
                    st.markdown("<div class='small'>Troubleshooting: check ticker spelling, add exchange suffix (.NS, .TO, .L), ensure network and that yfinance/alpha_vantage are installed.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# COMPARE TAB
# -------------------------------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚖️ Compare Multiple Companies")
    line = st.text_input("Enter tickers (comma separated)", value="AAPL, MSFT, TSLA, GOOGL", help="Enter up to 12 tickers separated by commas")
    if st.button("🔍 Compare All", type="primary"):
        tks = normalize_tickers(line)[:12]
        if not tks:
            st.warning("Please enter at least one valid ticker.")
        else:
            with st.spinner(f"🔄 Analyzing {len(tks)} companies..."):
                rows = []
                progress_bar = st.progress(0)
                for i, tk in enumerate(tks):
                    try:
                        base = collect_financial_data(tk, prefer=prefer)
                        ratios = compute_ratios(base)
                        z, _ = altman_z(ratios)
                        ml = predict_all(ratios)
                        comb = combined_assessment(z, ml)
                        record_industry(base["sector"], tk, z, ml["probability_bankrupt"])
                        rows.append({"Ticker": tk, "Company": base["company_name"], "Sector": base["sector"],
                                     "Z-Score": round(z, 2), "ML Risk %": ml["probability_bankrupt"],
                                     "Combined %": comb["combined_probability"], "Status": comb["status"], "Risk Level": comb["risk_level"]})
                    except Exception as e:
                        rows.append({"Ticker": tk, "Company": "Error", "Sector": "-", "Z-Score": None, "ML Risk %": None,
                                     "Combined %": None, "Status": f"Failed: {str(e)[:30]}", "Risk Level": "-"})
                    progress_bar.progress((i + 1) / max(len(tks), 1))
                progress_bar.empty()
                if rows:
                    dfc = pd.DataFrame(rows)
                    st.dataframe(dfc, use_container_width=True, hide_index=True)
                    valid = dfc[dfc["Combined %"].notna()].copy()
                    if not valid.empty:
                        colors = ["#10b981" if s == "Safe" else "#f59e0b" if s == "Gray Zone" else "#ef4444" for s in valid["Status"]]
                        fig = go.Figure([go.Bar(x=valid["Ticker"], y=valid["Combined %"], marker_color=colors,
                                                text=valid["Combined %"].round(1), texttemplate="%{text}%", textposition="outside")])
                        fig.update_layout(title="Combined Bankruptcy Risk Comparison", template="plotly_dark", height=450,
                                          margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font=dict(color="#cbd5e1"), yaxis_title="Bankruptcy Risk %")
                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.1)", range=[0, 100])
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# INDUSTRY TAB
# -------------------------------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏢 Industry Analysis")
    sectors = sorted(list(st.session_state.industry.keys()))
    if sectors:
        sec = st.selectbox("Select sector", [""] + sectors)
        if sec:
            stats = sector_stats(sec)
            if stats:
                m1, m2, m3, m4, m5 = st.columns(5)
                labels = ["Companies", "Avg Z-Score", "Median Z", "Min Z", "Max Z"]
                vals = [stats['n'], stats['z_mean'], stats['z_med'], stats['z_min'], stats['z_max']]
                for col, label, val in zip([m1, m2, m3, m4, m5], labels, vals):
                    with col:
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>""", unsafe_allow_html=True)
                fig_sector = chart_sector_risk(stats)
                if fig_sector:
                    st.plotly_chart(fig_sector, use_container_width=True, config={"displayModeBar": False})
                st.markdown("#### Sector Companies")
                st.dataframe(pd.DataFrame(stats["rows"]), use_container_width=True, hide_index=True)
    else:
        st.info("📊 No sector data available yet. Run some analyses to populate industry statistics.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# MODELS TAB
# -------------------------------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Machine Learning Models Performance")
    m = build_models()["metrics"]
    rows = [{"Model": name.replace("_", " ").title(),
             "Accuracy": f"{met['accuracy']*100:.2f}%", "Precision": f"{met['precision']*100:.2f}%",
             "Recall": f"{met['recall']*100:.2f}%", "F1-Score": f"{met['f1']*100:.2f}%", "ROC-AUC": f"{met['roc_auc']*100:.2f}%"}
            for name, met in m.items()]
    df_rows = pd.DataFrame(rows)
    st.dataframe(df_rows, use_container_width=True, hide_index=True, height=250)
    fig = go.Figure()
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    color_map = {"Accuracy": "#6366f1", "Precision": "#8b5cf6", "Recall": "#ec4899", "F1-Score": "#10b981", "ROC-AUC": "#f59e0b"}
    for metric in metrics_to_plot:
        vals = [float(v.strip("%")) for v in df_rows[metric]]
        fig.add_trace(go.Bar(name=metric, x=df_rows["Model"], y=vals, marker_color=color_map[metric]))
    fig.update_layout(title="Model Performance Comparison", template="plotly_dark", height=450, barmode="group",
                      margin=dict(l=20, r=20, t=60, b=80), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#cbd5e1"), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                      yaxis_title="Score %")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.1)", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("<div class='small'>Training data: 1,000 synthetic samples (≈33% bankrupt). Ensemble: majority vote with probability averaging.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# HISTORY TAB
# -------------------------------
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📜 Analysis History")
    hist = st.session_state.history[-50:]
    if hist:
        tidy = [{"Timestamp": h["ts"], "Ticker": h["ticker"], "Company": h["company_name"], "Sector": h["sector"],
                 "Z-Score": h["altman"]["score"], "Z-Status": h["altman"]["status"],
                 "ML Risk %": h["ml"]["probability_bankrupt"], "Combined %": h["combined"]["combined_probability"],
                 "Final Status": h["combined"]["status"]} for h in hist]
        history_df = pd.DataFrame(tidy)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if len(hist) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(hist))),
                                     y=[h["combined"]["combined_probability"] for h in hist],
                                     mode="lines+markers", name="Combined Risk",
                                     line=dict(color="#6366f1", width=3),
                                     marker=dict(size=8, color="#6366f1"),
                                     text=[h["ticker"] for h in hist],
                                     hovertemplate="<b>%{text}</b><br>Risk: %{y:.2f}%<extra></extra>"))
            fig.update_layout(title="Analysis Timeline - Risk Progression", template="plotly_dark", height=400,
                              margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#cbd5e1"), xaxis_title="Analysis #", yaxis_title="Bankruptcy Risk %")
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.1)")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.1)", range=[0, 100])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("📊 No analysis history yet. Start by analyzing a company in the Analyze tab.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# EXPORT TAB
# -------------------------------
with tabs[5]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📥 Export Analysis Results")
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.markdown(f"<div class='small'><strong>Latest Analysis:</strong><br><strong>Company:</strong> {last['company_name']}<br><strong>Ticker:</strong> {last['ticker']}<br><strong>Status:</strong> {last['combined']['status']}<br><strong>Risk:</strong> {last['combined']['combined_probability']}%</div>", unsafe_allow_html=True)
        out = _csv_from_history_last()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Latest Analysis (CSV)", data=out,
                file_name=f"{last['ticker']}_bankruptcy_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True, type="primary"
            )
        with col2:
            if len(st.session_state.history) > 1:
                all_hist = [{"Timestamp": h["ts"], "Ticker": h["ticker"], "Company": h["company_name"], "Sector": h["sector"],
                             "Z-Score": h["altman"]["score"], "Z-Status": h["altman"]["status"],
                             "ML_Risk_%": h["ml"]["probability_bankrupt"], "Combined_%": h["combined"]["combined_probability"],
                             "Status": h["combined"]["status"], "Risk_Level": h["combined"]["risk_level"]} for h in st.session_state.history]
                csv_all = pd.DataFrame(all_hist).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Full History (CSV)", data=csv_all,
                    file_name=f"bankruptcy_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv", use_container_width=True
                )
    else:
        st.info("📊 No analysis results to export. Run an analysis first.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div style="text-align:center;margin-top:24px;padding:16px;border-top:1px solid rgba(99,102,241,.2);color:#94a3b8">
  <p><strong>Powered by:</strong> Altman Z-Score | ML (LR, RF, GB, XGBoost)</p>
  <p><strong>Data Sources:</strong> Yahoo Finance (yfinance), Alpha Vantage (fallback), Stooq (optional)</p>
  <p><strong>Disclaimer:</strong> Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

