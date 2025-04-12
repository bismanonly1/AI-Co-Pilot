# Model Evaluation & Visualization Agent

This agent evaluates ML model predictions and visualizes results using FastAPI.

## Features
- Uploads `y_true`, `y_pred`, `y_proba` CSVs
- Computes accuracy
- Generates confusion matrix & ROC curve
- Saves plots to `/plots/`
- FastAPI endpoint: `/evaluate`
- Swagger UI: `http://localhost:8000/docs`

## Run Locally
```bash
uvicorn evaluation_api:app --reload
