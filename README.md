# HDB Resale Price Predictor — L06 (Better & Trustworthy Models)

A teaching project that upgrades a simple 3-feature HDB price model into a more
accurate, honestly-evaluated one — and deploys it as a Streamlit web app.

Built as a 2-hour coaching add-on for **Module 3 · Machine Learning & GenAI**,
continuing from the L01–L05 predictor.

## 🚀 Live demo / run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app **trains itself on first run** (downloads a live HDB resale dataset,
compares two models, keeps the better one, and caches it). No need to train
manually — but you can if you want to see the scores:

```bash
python model.py
```

## 🧠 What it does
- Predicts an HDB flat's resale price from floor area, lease year, floor level,
  **flat type**, and **town**.
- Reports its own accuracy (MAE — average dollar error) tested on unseen flats.
- Compares Linear Regression vs Random Forest and deploys the better one.

## 📂 Repository contents
```
app.py                        Streamlit web app (auto-trains on first run)
model.py                      Training + load_or_train logic (run standalone too)
requirements.txt              Python dependencies
runtime.txt                   Python version pin for Streamlit Cloud
facilitator_run_sheet.md      2-hour session plan (Kolb's cycle)
explainer.html                Interactive concept aid (MAE / features / overfitting)
notebooks/L06_practice.ipynb  Self-runnable hands-on notebook (incl. model stacking)
```

## ☁️ Deploy to Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo (see steps below).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app**, pick your repo, set **Main file path** to `app.py`, deploy.
4. First load takes ~1 minute while it trains; after that it's instant.

> The trained model file (`house_model.pkl`) is intentionally **not** committed
> (see `.gitignore`) — the app regenerates it, so deployments always train fresh
> against the latest data.

## 📤 Push this to GitHub
This folder is already a git repository with an initial commit. To publish it:

```bash
# 1. Create an EMPTY repo on github.com (no README/license), copy its URL.
# 2. In this folder:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

(If you prefer the GitHub CLI: `gh repo create <repo-name> --public --source=. --push`.)

## 📚 Learning concepts covered
MAE · train/test split · feature engineering · model selection · overfitting · model stacking.

## License
MIT — see `LICENSE`.
