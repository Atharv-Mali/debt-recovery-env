# Submission Guide

## 1. Create the repo

Run these commands inside `debt-recovery-env`:

```powershell
git init
git add .
git commit -m "Prepare OpenEnv submission"
```

If Git shows a `dubious ownership` warning on Windows, run:

```powershell
git config --global --add safe.directory C:/Users/SEJAL/Downloads/Meta1/debt-recovery-env
```

Create a GitHub repository named `debt-recovery-env`, then push:

```powershell
git branch -M main
git remote add origin https://github.com/<your-username>/debt-recovery-env.git
git push -u origin main
```

## 2. Create the Hugging Face Space

Create a Docker Space with the same project files, then push the repo there.

Set these Space secrets or variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `ENV_BASE_URL` optional, defaults to `http://127.0.0.1:7860`
- `LOCAL_IMAGE_NAME` optional and only needed for a docker-image based inference flow

## 3. What gets submitted

- GitHub URL: `https://github.com/<your-username>/debt-recovery-env`
- HF Space URL: `https://huggingface.co/spaces/<your-username>/debt-recovery-env`

## 4. Final checks

- `inference.py` is in the repo root.
- `inference.py` defines `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.
- All LLM calls in `inference.py` use `OpenAI(...)`.
- The script prints `[START]`, `[STEP]`, and `[END]` structured logs.
- `openenv.yaml` has your real Space URL before submission.
