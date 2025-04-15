class ChatAgent:
    def __init__(self):
        pass

    def analyze(self, user_input: str) -> dict:
        """Extracts project goal and intent from user message."""
        pass

    def recommend_dataset(self, project_goal: dict) -> list:
        """Suggests datasets based on the project goal."""
        pass

    def summarize_results(self, eval_report: dict, model: any) -> str:
        """Generates a human-friendly summary of model results."""
        pass
