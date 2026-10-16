# Credit Risk Assessment

Predicts the probability that a loan applicant defaults, explains why, and serves predictions through a FastAPI backend with a simple web interface.

## Run It Locally

Install the training dependencies:

```bash
pip install -r requirements-train.txt
```

1. Place `credit_risk_dataset.csv` (Kaggle Credit Risk Dataset) in the `data/` directory.
2. Train the model:

   ```bash
   python train.py
   ```

   Training generates `model/credit_risk_model.joblib`, `model/metrics.json`, and charts in `reports/`.

3. Start the API:

   ```bash
   uvicorn main:app --reload
   ```

4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Interactive API documentation is available at `/docs`.

## API Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 35,
    "person_income": 85000,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 10,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 8000,
    "loan_int_rate": 10.5,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 10
  }'
```

The API returns an estimated default probability, a risk band, a suggested action, applicant-level explanations, and warnings where applicable.

Additional endpoints:
- `GET /meta` — model metrics.
- `GET /health` — health check.

## Key Improvements

### Model and data quality
- Corrected the F1 threshold optimization formula.
- Introduced separate training, validation, and test splits.
- Calibrated predicted probabilities using Platt scaling.
- Fixed employment-length cleaning so missing values are not silently discarded.
- Corrected the XGBoost estimator parameter and implemented income validation.
- Calculate `loan_percent_income` from income and loan amount instead of accepting contradictory input.
- Validate categorical inputs and resolve file paths relative to the project.

### Explainability and interface
- Applicant-level Shapley-value explanations across the ten input features.
- Calibrated probabilities with Brier-score comparisons and a calibration chart.
- Three risk bands, input-range warnings, and optional-field imputation.
- Web interface with example inputs, validation messages, and dark mode.

## Deployment on Render

1. Push the project to GitHub. Include the trained model and evaluation reports; exclude datasets and unnecessary generated files.
2. Create a Web Service on [Render](https://render.com/) connected to your GitHub repository.
3. Configure the commands:

   **Build command**
   ```bash
   pip install -r requirements.txt
   ```

   **Start command**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

4. Ensure the Python, scikit-learn, and XGBoost versions match the environment used to train the model.

## Limitations and Responsible Use

- `loan_grade` and `loan_int_rate` may be assigned by lenders after assessing risk. Using them as inputs can introduce target leakage or circularity.
- Age and other applicant attributes may be subject to legal restrictions or fairness concerns.
- The model is trained on a public dataset and has not been established as fair, compliant, or suitable for real lending decisions.
- Feature explanations describe the model's behavior; they do not establish causation.

**This project is for educational purposes and should not be used to make real lending decisions without rigorous validation, fairness testing, and regulatory review.**