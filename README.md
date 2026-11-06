# Credit Card Fraud Detection

Production-grade machine learning pipeline and FastAPI inference service for detecting fraudulent credit card transactions. The system is trained on 284,807 European card transactions containing 492 frauds, a 0.172% fraud rate.

## Executive Summary

The Random Forest classifier achieves **0.8494 PR-AUC** on a held-out test set of 56,962 transactions, including 98 frauds, that was never resampled. A cost-optimised decision threshold of **0.33** reduces expected financial loss from **$1,680** at the default 0.50 threshold to **$1,204**, a reduction of 28.3%, while catching 88 of 98 frauds. Approving every transaction would cost an expected $9,800 in fraud losses, so the tuned model avoids **$8,596** in expected loss.

| Metric | Value |
|---|---|
| PR-AUC | 0.8494 |
| Optimal threshold | 0.33 |
| Expected cost at 0.50 | $1,680 |
| Expected cost at 0.33 | $1,204 |
| Fraud recall at 0.33 | 0.8980 |
| Fraud precision at 0.33 | 0.4632 |
| False negatives at 0.33 | 10 |
| False positives at 0.33 | 102 |

## System Architecture

```
creditcard.csv
      │
      ▼
train_pipeline.py
      ├── stratified train/test split
      ├── StandardScaler fitted on X_train
      ├── SMOTE applied to X_train only
      ├── Logistic Regression and Random Forest training
      ├── cost-matrix threshold optimization
      ├── plots/ diagnostic figures
      └── artifacts/ model, scaler, metadata
                    │
                    ▼
              FastAPI service
              ├── GET  /health
              ├── POST /predict
              └── POST /predict-batch
```

## MLOps Engineering Principles

### Zero Data Leakage

The stratified 80/20 split is performed before any transformation. `StandardScaler` is fitted on `X_train[["Time", "Amount"]]` only, never on the full dataset. SMOTE is applied exclusively to the training partition. `X_test` remains raw and un-sampled, so every reported metric reflects the real 0.172% fraud prevalence.

### Cost-Sensitive Learning

The classification threshold is selected by sweeping 0.01 to 0.99 and minimising expected monetary loss under a business cost matrix of $100 per false negative and $2 per false positive. Ties are broken toward fewer alerts, which avoids unnecessary manual review at identical cost.

### Artifact Persistence and Contract

Training exports `fraud_random_forest_model.joblib`, `scaler.joblib`, and `model_metadata.json`. Metadata carries the optimal threshold, model version, feature order, PR-AUC, cost matrix, and library versions. The service validates feature count, scaler shape, and threshold bounds at startup, so an incompatible artifact fails readiness instead of returning silent mispredictions.

### Reproducibility and Versioning

All randomness is seeded with `random_state=42` for the split, SMOTE, and both estimators. Dependency versions are pinned, and every artifact records its training timestamp and library versions.

### Resilience and Observability

Model artifacts are loaded once during the FastAPI lifespan. If artifacts are missing or inconsistent, the process stays up, `/health` returns 503 with a degraded status, and inference endpoints return an actionable 503 instead of crashing. Request latency, risk level, and flag outcome are logged without recording raw transaction features. Validation failures return 422 and unexpected exceptions return an opaque 500.

### Container Deployment

The image is built from `python:3.12-slim`, runs as an unprivileged user, installs pinned dependencies without a build toolchain, and exposes a `HEALTHCHECK` that calls `/health`.

## Model Benchmark Summary

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | PR-AUC | Expected Cost |
|---|---|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.50 | 0.9742 | 0.0580 | 0.9184 | 0.1092 | 0.7704 | $3,722 |
| Random Forest (Default) | 0.50 | 0.9990 | 0.6721 | 0.8367 | 0.7455 | 0.8494 | $1,680 |
| Random Forest (Optimized) | 0.33 | 0.9980 | 0.4632 | 0.8980 | 0.6111 | 0.8494 | $1,204 |

Cost assumes $100 per false negative and $2 per false positive. The baseline's 0.058 precision at the default threshold means 94.2% of its alerts are false alarms, which is why cost, not accuracy, drives model selection.

## Repository Directory Structure

```
.
├── train_pipeline.py
├── app.py
├── schemas.py
├── requirements.txt
├── Dockerfile
├── README.md
├── LICENSE
├── artifacts/
│   ├── fraud_random_forest_model.joblib
│   ├── scaler.joblib
│   └── model_metadata.json
└── plots/
    ├── class_distribution.png
    ├── amount_distribution.png
    ├── correlation_matrix.png
    ├── confusion_matrices.png
    ├── pr_curves.png
    ├── cost_curve.png
    └── feature_importance.png
```

## Local Setup and Quickstart

Prerequisites are Python 3.12 or 3.13 and the `creditcard.csv` dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_pipeline.py --data creditcard.csv
uvicorn app:app --host 0.0.0.0 --port 8000
```

On Windows, activate the environment with `.venv\Scripts\activate`. Training completes in roughly two to three minutes, writes all diagnostic figures to `plots/`, and exports the serving artifacts to `artifacts/`. Interactive API documentation is available at http://localhost:8000/docs.

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1.0.0",
  "decision_threshold": 0.33
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.53634673796914,
    "V4": 1.37815522427443, "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507, "V9": 0.363786969611213,
    "V10": 0.0907941719789316, "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879, "V15": 1.46817697209427,
    "V16": -0.470400525259478, "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705, "V21": -0.018306777944153,
    "V22": 0.277837575558899, "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824, "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62
  }'
```

Expected response:

```json
{
  "is_fraud": false,
  "fraud_probability": 0.032086,
  "decision_threshold_used": 0.33,
  "risk_level": "LOW",
  "execution_time_ms": 36.447
}
```

The sample above is a legitimate transaction, so it scores LOW. Fraudulent patterns return `is_fraud: true` with `risk_level: HIGH`.

`risk_level` is `HIGH` at or above the threshold, `MEDIUM` at or above half the threshold, and `LOW` below that.

### Batch Prediction

`POST /predict-batch` accepts up to 100 transactions in a single vectorised call:

```json
{
  "transactions": [
    {
      "Time": 406.0,
      "V1": -1.3598071336738,
      "V2": -0.0727811733098497,
      "V3": 2.53634673796914,
      "V4": 1.37815522427443,
      "V5": -0.338320769942518,
      "V6": 0.462387777762292,
      "V7": 0.239598554061257,
      "V8": 0.0986979012610507,
      "V9": 0.363786969611213,
      "V10": 0.0907941719789316,
      "V11": -0.551599533260813,
      "V12": -0.617800855762348,
      "V13": -0.991389847235408,
      "V14": -0.311169353699879,
      "V15": 1.46817697209427,
      "V16": -0.470400525259478,
      "V17": 0.207971241929242,
      "V18": 0.0257905801985591,
      "V19": 0.403992960255733,
      "V20": 0.251412098239705,
      "V21": -0.018306777944153,
      "V22": 0.277837575558899,
      "V23": -0.110473910188767,
      "V24": 0.0669280749146731,
      "V25": 0.128539358273528,
      "V26": -0.189114843888824,
      "V27": 0.133558376740387,
      "V28": -0.0210530534538215,
      "Amount": 149.62
    }
  ]
}
```

The response contains `count`, a `predictions` array with the same fields as `/predict`, and total `execution_time_ms` for the batch.

## Docker

```bash
docker build -t fraud-api:1.0.0 .
docker run --rm -p 8000:8000 fraud-api:1.0.0
```

The image expects `artifacts/` to exist in the build context, which `train_pipeline.py` produces. The container runs as an unprivileged user and reports health status through the `/health` endpoint.

## License

Released under the MIT License. See LICENSE for details.
