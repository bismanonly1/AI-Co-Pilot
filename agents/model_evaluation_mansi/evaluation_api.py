from fastapi import FastAPI, UploadFile, File
import pandas as pd
from agents.evaluation_agent import EvaluationAgent

app = FastAPI()
evaluator = EvaluationAgent()

@app.get("/")
def read_root():
    return {"message": "Evaluation API is working!"}

@app.post("/evaluate")
async def evaluate_model(
    y_true_file: UploadFile = File(...),
    y_pred_file: UploadFile = File(...),
    y_proba_file: UploadFile = File(...)
):
    try:
        # Read uploaded files
        y_true = pd.read_csv(y_true_file.file).squeeze()
        y_pred = pd.read_csv(y_pred_file.file).squeeze()
        y_proba = pd.read_csv(y_proba_file.file).squeeze()

        # Check lengths
        if len(y_true) != len(y_pred) or len(y_true) != len(y_proba):
            return {"error": "Files must have the same number of rows"}

        # Run evaluation
        acc = evaluator.evaluate(y_true, y_pred, y_proba)

        return {
            "accuracy": acc,
            "confusion_matrix_path": "plots/confusion_matrix.png",
            "roc_curve_path": "plots/roc_curve.png"
        }

    except Exception as e:
        return {"error": str(e)}