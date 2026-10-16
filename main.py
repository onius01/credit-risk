from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from credit_utils import Scorer

# Paths relative to THIS file, so the app works no matter where it is launched from
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "credit_risk_model.joblib"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Run `python train.py` first.")
    state["scorer"] = Scorer(joblib.load(MODEL_PATH))
    yield
    state.clear()


app = FastAPI(title="Credit Risk Assessment API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])


class LoanApplication(BaseModel):
    """What the applicant provides. Categories and ranges are validated, so a typo
    like 'RENTT' is rejected instead of being silently ignored by the model."""
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0, le=10_000_000, description="Annual income")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: Optional[float] = Field(default=None, ge=0, le=60,
                                               description="Years employed (optional)")
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                         "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0, le=1_000_000)
    loan_int_rate: Optional[float] = Field(default=None, ge=0, le=40,
                                           description="Interest rate % (optional)")
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(ge=0, le=80)
    # NOTE: loan_percent_income is no longer an input. It is computed as
    # loan_amnt / person_income so it can never contradict the other two numbers.

    @model_validator(mode="after")
    def check_consistency(self):
        if self.person_emp_length is not None and self.person_emp_length > self.person_age:
            raise ValueError("Employment length cannot exceed age")
        if self.cb_person_cred_hist_length > self.person_age:
            raise ValueError("Credit history length cannot exceed age")
        return self


class Reason(BaseModel):
    feature: str
    label: str
    value: Optional[float | str]
    impact_pp: float          # percentage points of default risk vs a typical applicant
    direction: str


class Assessment(BaseModel):
    default_probability: float
    default_prediction: int
    threshold: float
    risk_band: str
    suggested_action: str
    loan_percent_income: float
    typical_applicant_risk: float
    reasons: List[Reason]
    warnings: List[str]


@app.post("/predict", response_model=Assessment)
def predict(data: LoanApplication):
    return state["scorer"].assess(data.model_dump())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    s = state["scorer"]
    m = s.metrics
    return {
        "categories": s.categories,
        "threshold": s.threshold,
        "trained_at": m.get("trained_at"),
        "default_rate": m.get("default_rate"),
        "test": m.get("test"),
        "global_importance_pp": m.get("global_importance_pp"),
    }


# must come LAST so it doesn't shadow the API routes
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
