# SMS Shield — React + FastAPI

A deployable split frontend/backend version of the SMS Shield research demo.

## Project structure

- `frontend/` — React + Vite UI
- `backend/` — FastAPI REST API and model inference
- Your trained `.joblib` models are intentionally not included in this ZIP. Copy them from your existing research project.

## 1. Copy model artifacts

From the existing `sms_spam_project_starter` folder, copy these directories into `sms-shield-fullstack/backend/artifacts/`:

- `artifacts/obfuscation/word_char_normalized_model.joblib`
- `artifacts/paraphrase_augmentation/word_char_normalized_paraphrase_augmented_model.joblib`

Preserve the folder structure exactly. Do not commit datasets or unrelated research files.

Expected paths:

```text
backend/artifacts/obfuscation/word_char_normalized_model.joblib
backend/artifacts/paraphrase_augmentation/word_char_normalized_paraphrase_augmented_model.joblib
```

## 2. Run locally on Windows

Open two PowerShell terminals in the extracted `sms-shield-fullstack` folder.

### Terminal A — FastAPI backend

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Test API docs at `http://127.0.0.1:8000/docs` and health at `http://127.0.0.1:8000/health`.

### Terminal B — React frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`.

## 3. Deploy the API to Render

1. Push the project to a GitHub repository.
2. In Render, create a **Web Service** from the repository; set root directory to `backend`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add `CORS_ORIGINS` with your Vercel URL, e.g. `https://your-sms-shield.vercel.app`.
6. Ensure model files are available to the service at the expected artifact paths. If deploying from Git, model files must be included in the repository or downloaded securely at build/runtime from private storage. Do not commit sensitive or licensed datasets. Large model artifacts may need Git LFS or a private object store depending on host limits.

## 4. Deploy React to Vercel

1. Import the GitHub repository into Vercel.
2. Set Root Directory to `frontend`.
3. Build command: `npm run build`; output directory: `dist`.
4. Set environment variable `VITE_API_URL` to your deployed Render API URL, e.g. `https://your-sms-shield-api.onrender.com`.
5. Redeploy after adding the variable.
6. Add the deployed Vercel origin to Render's `CORS_ORIGINS`, then redeploy the API.

## API endpoints

- `GET /health` — checks model file presence
- `GET /api/models` — model metadata
- `POST /api/predict` — compare selected models
- `POST /api/robustness` — compare both available models
- `GET /docs` — interactive Swagger API documentation

## Notes

- Predictions use the exact normalization pattern and word/character vectorizer workflow from the current Streamlit demo.
- SVM decision scores are margins, not probabilities.
- The research dataset has no timestamps, so the current experiments do not constitute a temporal evaluation.
- Held-out paraphrase candidates need independent human validation before they are described as verified labels.
