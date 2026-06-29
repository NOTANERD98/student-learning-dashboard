# 🎓 Student Learning Patterns & Dropout-Risk Dashboard

An interactive **Streamlit** dashboard (Hebrew / RTL) for a Data-Mining final project
at **Kinneret College**. It explores student learning patterns, compares supervised
ML models, predicts course success, clusters students, and analyzes model fairness.

> The app is **self-contained** — if `data/students_data.csv` is missing it
> generates a realistic synthetic dataset (1,000 students) automatically, so it
> runs anywhere with zero setup.

---

## ✨ Features

| Page | What it shows |
|------|----------------|
| 🏠 Overview | Dataset summary, research question, variable dictionary |
| 📊 EDA | Distributions, correlation matrix, boxplots |
| 🤖 Model Comparison | 7 models (Decision Tree, Random Forest, KNN, Logistic Regression, Naive Bayes, Neural Net, Gradient Boosting) with Accuracy / Precision / Recall / F1 / AUC, confusion matrix, ROC, feature importance |
| 🔍 Predict | Predict success for a new student + "what-if" suggestions |
| 🗂️ Clustering | K-Means in PCA space with silhouette score |
| ⚖️ Fairness | Per-group performance across prior-grade bands |

**Stack:** Streamlit · scikit-learn · pandas · numpy · matplotlib · seaborn

---

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Then open http://localhost:8501 . (Windows users can also double-click `run_dashboard.bat`.)

---

## ☁️ Deploy publicly (Streamlit Community Cloud — recommended, free)

1. Push this folder to a **public GitHub repository**.
2. Go to **https://share.streamlit.io** → sign in with GitHub.
3. **Create app** → pick the repo, branch `main`, main file `dashboard.py`.
4. *(Advanced settings)* set **Python version 3.12**.
5. **Deploy.** Your app goes live at:

   ```
   https://<your-app-name>.streamlit.app
   ```

### Update the live app later
Just push to GitHub — Streamlit Cloud redeploys automatically:

```bash
git add -A
git commit -m "Update dashboard"
git push
```

---

## 📁 Project structure

```
.
├── dashboard.py            # the Streamlit app
├── requirements.txt        # pinned Python dependencies
├── packages.txt            # apt packages (Hebrew fonts) for Streamlit Cloud
├── runtime.txt             # Python version (Render / Railway / Heroku)
├── Procfile                # start command (Render / Railway / Heroku)
├── render.yaml             # one-click Render blueprint (optional)
├── Dockerfile              # container image (Render / Fly.io / VPS / Docker)
├── .dockerignore
├── .streamlit/config.toml  # theme + server settings
├── .gitignore
├── data/                   # optional: drop a real students_data.csv here
└── run_dashboard.bat       # local Windows launcher
```

---

## 🔄 Alternative hosts

| Host | Free tier | Notes |
|------|-----------|-------|
| **Streamlit Community Cloud** | ✅ | Best fit — native Streamlit, auto-redeploy on push |
| Hugging Face Spaces | ✅ | Choose the *Streamlit* SDK; great uptime |
| Render | ✅* | Uses `Procfile` / `render.yaml`; free tier sleeps when idle |
| Railway / Fly.io | trial/paid | Use the `Dockerfile` |
| Docker / VPS | n/a | `docker build -t dashboard . && docker run -p 8501:8501 dashboard` |

\* Free web services on Render sleep after inactivity and cold-start on the next visit.

---

*Built for the Kinneret College Data-Mining course. Claude AI + ChatGPT assisted development.*
