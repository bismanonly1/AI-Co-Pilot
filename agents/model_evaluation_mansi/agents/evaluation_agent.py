import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
class EvaluationAgent:
    def __init__(self, plot_dir="plots"):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
    def evaluate(self, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.2f}")
        # Confusion Matrix
        self.plot_confusion_matrix(y_true, y_pred)
        # ROC Curve
        self.plot_roc_curve(y_true, y_proba)
        return acc
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        path = os.path.join(self.plot_dir, "confusion_matrix.png")
        fig.savefig(path)
        plt.close(fig)
    def plot_roc_curve(self, y_true, y_proba):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="green")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        path = os.path.join(self.plot_dir, "roc_curve.png")
        fig.savefig(path)
        plt.close(fig)
if __name__ == "__main__":
    # Example usage
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    y_proba = [0.1, 0.9, 0.8, 0.7, 0.2, 0.4]
    evaluator = EvaluationAgent()
    evaluator.evaluate(y_true, y_pred, y_proba)
    # This is a placeholder for actual model predictions        
    # y_true = model.predict(X_test)
    # y_pred = model.predict(X_test)
    # y_proba = model.predict_proba(X_test)[:, 1]
    # evaluator.evaluate(y_true, y_pred, y_proba)
    # evaluator.evaluate(y_true, y_pred, y_proba)
    