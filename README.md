# 🏛 Bankruptcy Prediction Dashboard

A **financial distress intelligence dashboard** that evaluates corporate bankruptcy risk using:

- **Altman Z-Score**
- **Machine Learning Ensemble** (Logistic Regression, Random Forest, Gradient Boosting, XGBoost*)
- **Comprehensive Financial Ratios**
- **Stock Price & Volume Visualization**
- **Industry & Multi-Company Comparison Tools**

All data is sourced from **Yahoo Finance via `yfinance`**, so **any global ticker** can be analyzed.

> **Purpose:** Built for financial analysis, investment research, and academic coursework.  
> **Note:** This tool is for **educational use only**, not investment advice.

---

## 🚀 Quick Start

```bash
# 1) Create & activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
## 📦 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
