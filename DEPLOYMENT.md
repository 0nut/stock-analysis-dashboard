# Deploy to Streamlit Cloud (Free)

Follow these steps to deploy this project on Streamlit Community Cloud using a GitHub repository.

## 1) Create a GitHub repository

1. Go to GitHub and create a new repository (public repos are fully supported on the free tier).
2. Clone the new repository to your machine (or use this existing local project folder).
3. Copy this project into the repository if needed.

## 2) Commit and push your code

From the project root, run:

```bash
git init
git add .
git commit -m "Initial stock dashboard app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If your repo already exists and is initialized, just run:

```bash
git add .
git commit -m "Prepare Streamlit Cloud deployment"
git push
```

## 3) Confirm required files exist

Make sure these files are present in the repo root before deploying:

- `streamlit_app.py` (Streamlit Cloud entrypoint)
- `requirements.txt` (Python dependencies)
- `.streamlit/config.toml` (default Streamlit settings)

## 4) Sign in to Streamlit Community Cloud

1. Open: https://share.streamlit.io/
2. Click **Sign in with GitHub**.
3. Authorize Streamlit to access your repository.

## 5) Create a new app deployment

1. Click **New app**.
2. Select your GitHub repository.
3. Choose the branch (usually `main`).
4. Set **Main file path** to:

   `streamlit_app.py`

5. Click **Deploy**.

## 6) Wait for first build

- Streamlit Cloud installs dependencies from `requirements.txt`.
- The initial deploy can take a few minutes.
- Once complete, you get a public app URL.

## 7) Update the app later

Whenever you push new commits to the selected branch, Streamlit Cloud can redeploy automatically.

If auto-redeploy is off, open the app in Streamlit Cloud and click **Reboot** or **Rerun**.

## Troubleshooting

- **Module not found errors**: confirm dependency is listed in `requirements.txt`.
- **Wrong file path**: make sure the app path is exactly `streamlit_app.py`.
- **App crashes on startup**: open **Manage app -> Logs** in Streamlit Cloud to inspect tracebacks.
- **Private repo access issues**: verify GitHub authorization for Streamlit Cloud.
