'''
Chat Agent Class

Expected flow:
[User Input]
   |
   v
[ChatAgent.analyze()]
   |
   v
[LLM or rule-based extractor]
   |
   v
[Parse for key elements]
   ├── Task Type → "regression" / "classification"
   ├── Target Variable → e.g. "price", "churn"
   └── Feature Hints → e.g. "location", "age", "subscription_type"
   |
   v
[Return Project Goal Dict]
   {
     "task": "regression",
     "target": "price",
     "features": ["location", "size", "bedrooms"]
   }
'''

import re
import requests
import subprocess
from typing import Dict, List

class ChatAgent:
    def __init__(self, endpoint="http://localhost:11434/api/generate", model="llama2"):
        self.endpoint = endpoint
        self.model = model
        self.ensure_llama_running()

    def ensure_llama_running(self):
        """
        Ensures llama2 is running via Ollama. Starts it if not running.
        """
        try:
            result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
            if self.model not in result.stdout:
                print(f"Starting {self.model} with ollama...")
                subprocess.Popen(["ollama", "run", self.model])
            else:
                print(f"{self.model} already running.")
        except FileNotFoundError:
            print("Ollama is not installed or not in PATH.")

    def analyze(self, user_input: str) -> Dict:
        """
        Extracts project intent: task type, target variable, and feature hints using LLaMA2 via Ollama.
        """
        system_prompt = (
            "You are a helpful AI assistant that helps identify machine learning project details.\n"
            "From the user's description, extract the following:\n"
            "1. Task type (classification or regression)\n"
            "2. Target variable\n"
            "3. List of features or keywords likely to be relevant.\n"
            "Return the answer in plain text with headings."
        )

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\nUser: {user_input}\n",
            "stream": False
        }

        response = requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        output_text = response.json().get("response", "")

        # Debug print
        print("Raw LLaMA2 Output:\n", output_text)

        task = "classification" if "classification" in output_text.lower() else "regression"
        target_match = re.search(r"target.*?:\s*(\w+)", output_text, re.IGNORECASE)
        target = target_match.group(1) if target_match else "unknown"
        features = re.findall(r"['\"](.*?)['\"]", output_text)

        return {
            "task": task,
            "target": target,
            "features": features
        }

    def recommend_dataset(self, project_goal: Dict) -> List[str]:
        """
        Return list of recommended dataset names based on goal.
        """
        task = project_goal.get("task")
        target = project_goal.get("target")
        if task == "regression":
            return ["Boston Housing Dataset", "California Housing", "Kaggle Real Estate"]
        else:
            return ["Titanic Survival", "Customer Churn", "Iris Classification"]

    def summarize_results(self, eval_report: Dict, model: any) -> str:
        """
        Return a friendly summary based on the model evaluation.
        """
        summary = f"Your {model.__class__.__name__} model achieved the following:\n"
        for metric, value in eval_report.items():
            summary += f"- {metric}: {value}\n"
        summary += "\nLooks like you're well on your way! 🎯"
        return summary

