"""
Train the credit risk model.

Usage:
    1. Put credit_risk_dataset.csv in ./data
    2. python train.py                (add --n-iter 150 for the full search from your notebook)

Outputs: model/credit_risk_model.joblib, model/metrics.json, reports/*.png
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from credit_utils import (CATEGORICAL_COLS, FEATURES, INPUT_FIELDS, NUMERIC_COLS,
                          PlattCalibrator, Scorer, clean_data)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default=str(BASE_DIR / "data" / "credit_risk_dataset.csv"))
parser.add_argument("--n-iter", type=int, default=40, help="random search iterations (notebook used 150)")
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--beta", type=float, default=1.0,
                    help="F-beta for the threshold. 1 = balance precision/recall; 2 = care more about "
                         "catching defaulters (more false alarms).")
args = parser.parse_args()


def preprocessor(scale):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLS),
    ])


def best_threshold(y_true, p, beta):
    """F-beta maximising cut-off. (The notebook used 2*(p - r)/(p + r): a minus sign
    instead of a times, so its 'best' threshold was not the F1 optimum.)"""
    prec, rec, thr = precision_recall_curve(y_true, p)
    prec, rec = prec[:-1], rec[:-1]
    fbeta = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
    return float(thr[np.argmax(fbeta)])


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(args.csv)
    df = clean_data(raw)
    print(f"Rows: {len(raw)} raw -> {len(df)} after cleaning; defaults = {df['loan_status'].mean():.1%}")

    X, y = df[FEATURES], df["loan_status"]

    # 60% train / 20% validation (calibration + threshold) / 20% test (final report only)
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.25,
                                                      stratify=y_rest, random_state=42)
    neg, pos = np.bincount(y_train)
    spw = neg / pos

    # ---- 1) compare models with 5-fold CV (training data only)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
    lr_pipe = Pipeline([("preprocessor", preprocessor(True)),
                        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    xgb_pipe = Pipeline([("preprocessor", preprocessor(False)),
                         ("classifier", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                                      scale_pos_weight=spw, eval_metric="logloss",
                                                      random_state=42))])
    cv_results = {}
    for name, pipe in [("Logistic Regression", lr_pipe), ("XGBoost (default)", xgb_pipe)]:
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=args.n_jobs)
        cv_results[name] = {k: float(s[f"test_{k}"].mean()) for k in scoring}
        print(f"{name:22s} ROC-AUC={cv_results[name]['roc_auc']:.4f}  PR-AUC={cv_results[name]['pr_auc']:.4f}")

    # ---- 2) tune XGBoost
    space = {
        "classifier__n_estimators": randint(150, 600),
        "classifier__max_depth": randint(3, 9),
        "classifier__learning_rate": uniform(0.01, 0.3),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__min_child_weight": randint(1, 10),
        "classifier__gamma": uniform(0, 5),
    }
    search = RandomizedSearchCV(xgb_pipe, space, n_iter=args.n_iter, cv=cv, scoring="average_precision",
                                n_jobs=args.n_jobs, random_state=42, verbose=1)
    search.fit(X_train, y_train)
    model = search.best_estimator_            # already refit on the full training split
    print(f"Tuned CV PR-AUC: {search.best_score_:.4f}")

    # ---- 3) calibrate on the validation split, then pick the threshold on CALIBRATED probabilities
    calibrator = PlattCalibrator().fit(model.predict_proba(X_val)[:, 1], y_val)
    p_val = calibrator.predict(model.predict_proba(X_val)[:, 1])
    threshold = best_threshold(y_val, p_val, args.beta)
    print(f"Threshold (F{args.beta:g}, validation): {threshold:.3f}")

    # ---- 4) honest evaluation on the untouched test split
    p_raw = model.predict_proba(X_test)[:, 1]
    p_cal = calibrator.predict(p_raw)
    pred = (p_cal >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    lr_pipe.fit(X_train, y_train)
    p_lr = lr_pipe.predict_proba(X_test)[:, 1]

    test = {
        "roc_auc": roc_auc_score(y_test, p_cal),
        "pr_auc": average_precision_score(y_test, p_cal),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "brier_uncalibrated": brier_score_loss(y_test, p_raw),
        "brier_calibrated": brier_score_loss(y_test, p_cal),
        "baseline_lr_roc_auc": roc_auc_score(y_test, p_lr),
        "baseline_lr_pr_auc": average_precision_score(y_test, p_lr),
        "confusion": {"true_negative": int(tn), "false_positive": int(fp),
                      "false_negative": int(fn), "true_positive": int(tp)},
    }
    test = {k: (float(v) if not isinstance(v, dict) else v) for k, v in test.items()}
    print(json.dumps(test, indent=2))

    # ---- 5) bundle everything the API needs
    ref = {c: float(X_train[c].median()) for c in NUMERIC_COLS if c in INPUT_FIELDS}
    ref.update({c: X_train[c].mode().iloc[0] for c in CATEGORICAL_COLS})
    ranges = {c: (float(X_train[c].quantile(0.005)), float(X_train[c].quantile(0.995)))
              for c in NUMERIC_COLS}
    artifact = {
        "pipeline": model, "calibrator": calibrator, "threshold": threshold,
        "reference": ref, "ranges": ranges,
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_COLS},
    }
    scorer = Scorer(artifact)

    # global importance = mean |Shapley| over a sample of test applicants
    sample = X_test.sample(min(150, len(X_test)), random_state=1)
    imp = np.zeros(len(INPUT_FIELDS))
    for _, row in sample.iterrows():
        app = {f: (None if pd.isna(row[f]) else (row[f].item() if hasattr(row[f], "item") else row[f]))
               for f in INPUT_FIELDS}
        imp += np.abs(scorer.shapley(app)[0])
    imp = dict(sorted(zip(INPUT_FIELDS, (imp / len(sample) * 100).round(2).tolist()),
                      key=lambda kv: -kv[1]))

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "rows_raw": int(len(raw)), "rows_clean": int(len(df)),
        "default_rate": float(y.mean()),
        "threshold": threshold, "threshold_beta": args.beta,
        "cv_results": cv_results, "best_params": {k.split("__")[1]: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                                                   for k, v in search.best_params_.items()},
        "test": test,
        "global_importance_pp": imp,
    }
    artifact["metrics"] = metrics
    joblib.dump(artifact, MODEL_DIR / "credit_risk_model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model/credit_risk_model.joblib and model/metrics.json")

    make_charts(y_test, p_raw, p_cal, threshold, imp)


def make_charts(y_test, p_raw, p_cal, threshold, imp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
    except ImportError:
        print("matplotlib not installed - skipping charts")
        return

    prec, rec, thr = precision_recall_curve(y_test, p_cal)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec)
    i = np.argmin(np.abs(thr - threshold))
    ax.scatter(rec[i], prec[i], color="red", zorder=3, label=f"chosen threshold = {threshold:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test set)")
    ax.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "precision_recall.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for p, name in [(p_raw, "before calibration"), (p_cal, "after calibration")]:
        frac, mean = calibration_curve(y_test, p, n_bins=10)
        ax.plot(mean, frac, "o-", label=name)
    ax.set(xlabel="Predicted probability", ylabel="Actual default rate", title="Calibration (test set)")
    ax.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "calibration.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    names, vals = list(imp.keys())[::-1], list(imp.values())[::-1]
    ax.barh(names, vals)
    ax.set(xlabel="Average effect on risk (percentage points)", title="What drives predictions")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "feature_importance.png", dpi=130); plt.close(fig)
    print("Saved charts to reports/")


if __name__ == "__main__":
    main()
