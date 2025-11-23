# Fake News Detection (Django)

A Django-based web application for detecting fake vs. real news using multiple NLP/ML models. It provides a browser UI with authentication and a JSON API for programmatic predictions. Model assets are loaded from the `data/models` directory, and searches are logged to `data/searches/searches.csv`.

## Features
- Text classification across multiple model options: `best`, `nb`, `rf`, `svm`, `lstm`, `bert`.
- Fast startup with lazy-loading for heavy models (BERT, LSTM) on first use.
- Authentication (signup, login, logout) and protected home page.
- Simple API endpoint for predictions (`POST /api/predict`).
- Search logging to CSV for analysis.

## Tech Stack
- Backend: Django 5.x
- ML: scikit-learn (TF‑IDF + classic models), TensorFlow/Keras (LSTM), Hugging Face Transformers (BERT)
- Data handling: pandas, joblib

## Project Structure
- `fnd/` — Django project config (`settings.py`, `urls.py`, `wsgi.py`)
- `detector/` — App with views, URLs, and services
  - `services/predict.py` — Predictor loader and prediction logic
  - `apps.py` — AppConfig with startup hook to warm up lightweight models
  - `views.py` — UI, auth, and `/api/predict` endpoint
- `data/` — Local data assets
  - `models/` — Serialized model artifacts
    - `tfidf_vectorizer.joblib`
    - `nb_model.joblib`, `rf_model.joblib`, `svm_model.joblib`, `best_model.joblib`
    - `lstm_model.keras`, `lstm_tokenizer.joblib`
    - `bert/` — Hugging Face format (`config.json`, tokenizer files, etc.)
  - `searches/searches.csv` — Logged requests and results
- `templates/` and `static/` — UI templates and assets

## Setup
1. Create and activate a virtual environment.
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure database in `fnd/settings.py`.
   - Default: PostgreSQL (`NAME`, `USER`, `PASSWORD`, `HOST`, `PORT`)
   - Optional development fallback: SQLite (see commented section in `settings.py`).
4. Apply migrations and create a superuser:
   - `python manage.py migrate`
   - `python manage.py createsuperuser`

## Running
- Start the development server:
  - `python manage.py runserver`
- App loads lightweight assets on startup; heavy models initialize on first use of `bert` or `lstm`.

## Prediction API
- Endpoint: `POST /api/predict/`
- Content-Type: `application/json`
- Send JSON with `text` and optional `model` (default `best`).
- Response returns `label`, `score`, and `model`.
- Notes:
  - `model` options: `best`, `nb`, `rf`, `svm`, `lstm`, `bert`
  - First call with `bert` or `lstm` may take longer due to lazy load; subsequent calls are fast.

## Web UI
- Auth pages:
  - `GET /auth/signup/` — JSON signup endpoint
  - `GET /auth/login/` — login page
  - `POST /auth/login-submit/` — JSON login endpoint
  - `GET /auth/logout/` — logout
- Home page:
  - `GET /` — text input and model selection, submits to `/api/predict`

## Models and Data
- Place trained artifacts under `data/models/` using the filenames listed above.
- BERT directory must contain Hugging Face files (e.g. `config.json`, tokenizer files, model weights if applicable).
- Searches are appended to `data/searches/searches.csv` with columns: `text,model,label,score`.

## Training

### 1) Prepare data
- Download `Fake.csv` and `True.csv` to `data/raw/`.
- Create a labeled DataFrame with columns: `text` (article text) and `label` (`fake` or `real`).
- Split into train/validation sets.

### 2) Train TF‑IDF + classic ML
Produces: `tfidf_vectorizer.joblib`, `nb_model.joblib`, `rf_model.joblib`, `svm_model.joblib`, `best_model.joblib`.
- Run: `python train_ml.py`

### 3) Train LSTM
Produces: `lstm_model.keras`, `lstm_tokenizer.joblib`.

Option A: Management command (if available):
- `python manage.py train_lstm`

Option B: Script example: (use your own training script)
- Run: `python train_lstm.py`

### 4) Fine‑tune BERT (optional)
Produces Hugging Face directory under `data/models/bert/`.

Steps:
- Run: `python train_bert.py`

After training, restart the Django server to ensure updated models are loaded.

## Performance
- Startup optimized by loading TF‑IDF and classic models only.
- Heavy models (BERT/LSTM) load on first use inside the prediction path.
- If you need BERT/LSTM preloaded at startup, this can be enabled in `detector/apps.py` and `services/predict.py`.

## Troubleshooting
- If BERT import errors mention `regex` conflicts, reinstall `regex`:
  - `.venv\Scripts\python.exe -m pip uninstall regex -y`
  - `.venv\Scripts\python.exe -m pip install regex`
- Ensure model files exist in `data/models/`. Missing artifacts return informative error JSON.
- PostgreSQL connection issues: verify credentials and server running, or switch to SQLite for local testing.

## Development Tips
- Keep `.venv/` and large raw model files out of version control.
- Use the API for automation and log analysis from `data/searches/searches.csv`.

## Training Scripts
- Place scripts at project root.
- Run classic ML: `python train_ml.py`
- Run LSTM: `python train_lstm.py`
- Run BERT: `python train_bert.py`
- Restart the server after training.

## Screenshots
- Home page: `static/screenshots/home.png`
- Auth page (login): `static/screenshots/login.png`
- Prediction result: `static/screenshots/predict.png`

Place screenshots under `static/screenshots/` and reference them in your documentation.

## Dataset Sources
- Fake and Real News Dataset (CSV): https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  - Files used: `Fake.csv`, `True.csv`
  - Recommended location: `data/raw/`
- Alternative sources:
  - LIAR dataset (short statements): https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
  - BuzzFeed/PolitiFact curated datasets (various articles) — search Kaggle for additional variants.

Ensure you preprocess and train models to produce artifacts under `data/models/` using the filenames expected by the app.

## Deployment

### Docker (Recommended for portability)
1. Build image:
   - Create a `Dockerfile` based on `python:3.11-slim`, install system deps, copy project, `pip install -r requirements.txt`, run `collectstatic`.
2. Run container:
   - `docker run -e DJANGO_SETTINGS_MODULE=fnd.settings -p 8000:8000 <image> python manage.py migrate && gunicorn fnd.wsgi:application --bind 0.0.0.0:8000`
3. Persist data/models:
   - Mount volumes: `-v ./data:/app/data`

### Gunicorn + Nginx (Typical Linux deployment)
1. Environment
   - Set `DEBUG=False`, secure `SECRET_KEY`, configure database credentials.
2. Prepare app
   - `pip install -r requirements.txt`
   - `python manage.py migrate`
   - `python manage.py collectstatic`
3. Start Gunicorn
   - `gunicorn fnd.wsgi:application --bind 0.0.0.0:8000 --workers 3`
4. Nginx reverse proxy
   - Proxy `server_name` to `http://127.0.0.1:8000`, serve `/static/` from `STATIC_ROOT`.
5. Systemd service (optional)
   - Create a unit to manage Gunicorn as a service and enable on boot.

### Windows (Development/Staging)
- Use `python manage.py runserver` or `waitress` for a simple production-like server:
  - `pip install waitress`
  - `waitress-serve --port=8000 fnd.wsgi:application`

### Security and Ops
- Rotate secrets, configure HTTPS via Nginx/Cloud provider, restrict allowed hosts.
- Monitor logs and model loading times; preload heavy models if needed for your workload.