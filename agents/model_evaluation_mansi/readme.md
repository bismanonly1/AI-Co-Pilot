## Model Evaluation & Visualization Agent

Developed as part of the AI-Co-Pilot project, this agent evaluates ML predictions and generates performance insights with visual output.


- ✅ Accuracy score
- 📊 Confusion Matrix plot
- 📈 ROC Curve plot
- 🧠 Auto-generated performance summary

### Endpoint

**POST /evaluate**

Form-data parameters:
- `y_true_file`: CSV of true labels
- `y_pred_file`: CSV of predicted labels
- `y_proba_file`: CSV of prediction probabilities

### Response
```json
{
  "accuracy": 0.82,
  "confusion_matrix_path": "plots/confusion_matrix.png",
  "roc_curve_path": "plots/roc_curve.png",
  "summary": "The model performed well with 82.0% accuracy. ROC curve suggests good class separability."
}
