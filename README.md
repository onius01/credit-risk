# Credit Risk Assessment

Predicts the probability that a loan applicant defaults, **explains why**, and serves it through a
FastAPI backend with a small web page.

## Run it
```bash
pip install -r requirements-train.txt
```
1. Put `credit_risk_dataset.csv` (Kaggle "Credit Risk Dataset") in `data/`.
2. Train: `python train.py` (about a minute; add `--n-iter 150` for the full search from the notebook).
   This writes `model/credit_risk_model.joblib`, `model/metrics.json` and charts in `reports/`.
3. Serve: `uvicorn main:app --reload`, then open http://127.0.0.1:8000
   (interactive API docs at `/docs`).

## API
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "person_age": 35, "person_income": 85000, "person_home_ownership": "MORTGAGE",
  "person_emp_length": 10, "loan_intent": "PERSONAL", "loan_grade": "B",
  "loan_amnt": 8000, "loan_int_rate": 10.5,
  "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 10 }'
```
Returns the probability, a risk band, a suggested action, the top reasons (each in percentage points
of risk versus a typical applicant) and any warnings. `GET /meta` returns model metrics; `GET /health` for uptime checks.

## What changed from the notebook version
**Bugs fixed**
- The threshold formula was `2*(p - r)/(p + r)` (minus instead of times), so it never found the F1 optimum.
- The threshold was tuned on the *test set* and on a *different, uncalibrated* model than the one deployed.
  Now: train / validation / test split. Calibration and the threshold use the validation split, and
  the test split is only used for the final report.
- The cleaning step `emp_length <= age` silently deleted every row with a missing employment length.
- `n_estimator` (typo) was ignored by XGBoost; the zero/negative income filter was only a comment.
- `loan_percent_income` was a user input and could contradict income and loan amount. It is now computed.
- Categories were free text, so a typo was silently ignored by the encoder. They are validated now.
- File paths were relative to wherever you launched the app; they are now relative to the project.

**Improvements**
- Per-applicant explanations: exact Shapley values over the 10 inputs, in percentage points, summing
  precisely to (this applicant's risk - typical applicant's risk). No extra packages needed.
- Calibrated probabilities (Platt scaling), with before/after Brier score and a calibration chart.
- Three risk bands, out-of-range input warnings, optional fields (missing values are imputed).
- Cleaner web page with examples, validation messages and dark mode.

## Deploying on Render (free)
1. Push this folder to GitHub (commit `model/` and `reports/`; `data/` is git-ignored).
2. On render.com create a **Web Service** from the repo.
   Build command: `pip install -r requirements.txt`
   Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Make versions match: `model/metrics.json` lists the scikit-learn and xgboost versions used for
   training. Pin them in `requirements.txt` (for example `scikit-learn==1.7.2`) and set the
   `PYTHON_VERSION` environment variable to your local Python (for example `3.12.7`).

## Limits worth knowing
- `loan_grade` and `loan_int_rate` are decided by the lender *after* assessing risk, so using them
  as inputs is partly circular. Real systems score applicants before those are assigned.
- Age is a legally sensitive input in many places. This is a learning project and has not been
  checked for fairness or regulatory compliance. Do not use it for real lending decisions.
- The model reflects one public dataset; the explanations describe the model, not cause and effect.
