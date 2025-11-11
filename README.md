# Bankruptcy Prediction Dashboard

A Streamlit app that computes Altman Z‑Score, key financial ratios, and an ML ensemble risk estimate using Yahoo Finance data (`yfinance`).

## Quick start

```bash
# 1) Create & activate a virtual env (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Open the URL that Streamlit prints (usually http://localhost:8501).

## Common ticker suffixes
- US: `AAPL`, `MSFT`
- India (NSE): `RELIANCE.NS`
- Canada (TSX): `SHOP.TO`
- UK (LSE): `VOD.L`

## Deploy (optional)
You can deploy on Streamlit Community Cloud:
1. Push this repo to GitHub (instructions below).
2. Go to **share.streamlit.io** → **New App** → select this repo and `app.py`.
3. Add secrets if needed; otherwise just deploy.

## Push to GitHub (first time)

```bash
# from your project folder
git init

# Set your identity once (required by Git)
git config --global user.name "Dheeraj Agarwal"
git config --global user.email "da62@buffalo.edu"

git add .
git commit -m "Initial commit: Bankruptcy Prediction Dashboard"

# Make sure the branch is named main
git branch -M main

# Replace the URL below with your repo's URL
git remote add origin https://github.com/DheerajAg01/Bankruptcy-Prediction-Dashboard.git
git push -u origin main
```

### Fixing common Git errors
- **`Author identity unknown`** → run the two `git config --global ...` commands above, then commit again.
- **`src refspec main does not match any`** → you haven’t committed yet. Run `git add .` then `git commit -m "message"`.
- **`remote origin already exists`** → run `git remote remove origin` and then add it again.
- **Permission denied/warnings while `git add .`** → ensure you are inside the project folder *only* (not your user home).

## Project structure
```
Bankruptcy Dashboard/
├─ app.py
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## License
For educational use.
