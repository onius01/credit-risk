"""
Shared logic for training (train.py) and serving (main.py).
Everything that must behave identically in both places lives here.
"""
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

NUMERIC_COLS = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CATEGORICAL_COLS = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

# What the applicant actually types in. loan_percent_income is NOT an input:
# it is computed as loan_amnt / person_income so the numbers can never disagree.
INPUT_FIELDS = [c for c in FEATURES if c != "loan_percent_income"]

LABELS = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_home_ownership": "Home ownership",
    "person_emp_length": "Employment length",
    "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "cb_person_default_on_file": "Previous default on file",
    "cb_person_cred_hist_length": "Credit history length",
}


# ------------------------------------------------------------------ cleaning
def clean_data(df):
    """
    Remove impossible rows. Unlike the notebook version, rows with a MISSING
    employment length are kept (NaN <= 60 is False, so the old filter silently
    deleted every one of them). The pipeline imputes them instead.
    """
    df = df.drop_duplicates().copy()
    emp = df["person_emp_length"]
    keep = (
        df["person_age"].between(18, 100)
        & (emp.isna() | ((emp <= 60) & (emp <= df["person_age"])))
        & (df["loan_amnt"] > 0)
        & (df["person_income"] > 0)
    )
    df = df[keep].copy()
    df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]
    return df.reset_index(drop=True)


def add_derived(df):
    df = df.copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]
    return df


def make_frame(rows):
    """List of applicant dicts (None = missing) -> model-ready DataFrame."""
    return add_derived(pd.DataFrame(rows, columns=INPUT_FIELDS))


# --------------------------------------------------------------- calibration
class PlattCalibrator:
    """Sigmoid calibration: maps raw model scores to honest probabilities."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_p, y):
        self.lr.fit(self._logit(raw_p), y)
        return self

    def predict(self, raw_p):
        return self.lr.predict_proba(self._logit(raw_p))[:, 1]


# ------------------------------------------------------------------- scoring
class Scorer:
    """Wraps the saved artifact: score an application and explain it."""

    def __init__(self, artifact):
        self.pipe = artifact["pipeline"]
        self.cal = artifact["calibrator"]
        self.threshold = float(artifact["threshold"])
        self.reference = artifact["reference"]     # a "typical applicant"
        self.ranges = artifact["ranges"]           # training range per numeric feature
        self.categories = artifact["categories"]
        self.metrics = artifact.get("metrics", {})

    def _proba(self, df):
        return self.cal.predict(self.pipe.predict_proba(df)[:, 1])

    # exact Shapley values (in probability points) against the typical applicant
    def shapley(self, x):
        """
        Fair credit assignment across the 10 input fields: how many percentage
        points of risk each field adds or removes compared with the typical
        applicant. Exact (all 2^10 coalitions, one batch), model-agnostic, and
        consistent with the probability shown to the user. loan_percent_income is
        recomputed inside every coalition so income and loan amount stay coherent.
        Guarantee: sum(phi) == p(applicant) - p(typical applicant).
        """
        fields = INPUT_FIELDS
        n = len(fields)
        masks = np.arange(2 ** n)
        bits = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)

        cols = {}
        for j, f in enumerate(fields):
            xv = np.nan if x.get(f) is None else x[f]
            cols[f] = np.where(bits[:, j], xv, self.reference[f])
        v = self._proba(add_derived(pd.DataFrame(cols)))

        fact = [math.factorial(k) for k in range(n + 1)]
        popcount = bits.sum(axis=1)
        phi = np.zeros(n)
        for j in range(n):
            without = masks[~bits[:, j]]
            w = np.array([fact[k] * fact[n - k - 1] / fact[n] for k in popcount[without]])
            phi[j] = np.sum(w * (v[without | (1 << j)] - v[without]))
        return phi, float(v[-1]), float(v[0])   # all-applicant, all-reference

    def band(self, p):
        if p >= self.threshold:
            return "High risk", "Decline or refer for senior review"
        if p >= self.threshold / 2:
            return "Medium risk", "Manual review recommended"
        return "Low risk", "Eligible for approval"

    def range_warnings(self, df):
        out = []
        for col, (lo, hi) in self.ranges.items():
            val = df[col].iloc[0]
            if pd.notna(val) and (val < lo or val > hi):
                out.append(
                    f"{LABELS.get(col, 'Loan as % of income')} ({val:,.2f}) is outside the range "
                    f"the model was trained on ({lo:,.2f} to {hi:,.2f}); the estimate may be less reliable."
                )
        return out

    def assess(self, app, top_k=5):
        df = make_frame([app])
        p = float(self._proba(df)[0])
        band, action = self.band(p)

        phi, p_x, p_ref = self.shapley(app)
        order = np.argsort(-np.abs(phi))
        reasons = []
        for j in order:
            impact = float(phi[j] * 100)
            if abs(impact) < 0.1:
                continue
            f = INPUT_FIELDS[j]
            reasons.append({
                "feature": f,
                "label": LABELS[f],
                "value": app.get(f),
                "impact_pp": round(impact, 1),
                "direction": "increases risk" if impact > 0 else "decreases risk",
            })
            if len(reasons) == top_k:
                break

        return {
            "default_probability": round(p, 4),
            "default_prediction": int(p >= self.threshold),
            "threshold": round(self.threshold, 4),
            "risk_band": band,
            "suggested_action": action,
            "loan_percent_income": round(float(df["loan_percent_income"].iloc[0]), 4),
            "typical_applicant_risk": round(p_ref, 4),
            "reasons": reasons,
            "warnings": self.range_warnings(df),
        }
