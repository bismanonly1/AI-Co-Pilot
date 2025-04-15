import streamlit as st
import pandas as pd

# Import agents from the agents folder
from agents.chat_agent import ChatAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.hyperparameter_agent import HyperparameterAgent
from agents.model_training_agent import ModelTrainingAgent
from agents.evaluation_agent import EvaluationAgent

def main():
    st.title("Multi-Agent Data Science Assistant")
    
    # Step 1: Get user input
    user_input = st.text_input("Describe your data science project:")
    
    if user_input:
        chat_agent = ChatAgent()
        project_goal = chat_agent.analyze(user_input)
        st.session_state["project_goal"] = project_goal
        st.write("Project goal extracted:", project_goal)
    
    # Step 2: Dataset upload/selection
    if "dataset" not in st.session_state:
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)")
        if uploaded_file:
            st.session_state["dataset"] = pd.read_csv(uploaded_file)
    
    # Step 3: Preprocessing
    if "dataset" in st.session_state:
        pre_agent = PreprocessingAgent()
        processed_data, prep_report = pre_agent.process(st.session_state["dataset"], st.session_state.get("project_goal", {}))
        st.session_state["processed_data"] = processed_data
        st.write("Preprocessing Report:", prep_report)
    
    # Step 4: Hyperparameter Tuning (optional)
    if st.button("Run Hyperparameter Tuning"):
        hp_agent = HyperparameterAgent()
        tuned_params = hp_agent.tune(st.session_state.get("processed_data"), st.session_state.get("project_goal", {}))
        st.session_state["tuned_params"] = tuned_params
        st.write("Tuned Parameters:", tuned_params)
    
    # Step 5: Training
    if st.button("Train Model"):
        train_agent = ModelTrainingAgent()
        model, train_report = train_agent.train(
            st.session_state.get("processed_data"),
            st.session_state.get("project_goal", {}),
            st.session_state.get("tuned_params")
        )
        st.session_state["model"] = model
        st.write("Training Report:", train_report)
    
    # Step 6: Evaluation
    if st.button("Evaluate Model"):
        eval_agent = EvaluationAgent()
        eval_report, visuals = eval_agent.evaluate(
            st.session_state.get("model"),
            st.session_state.get("processed_data"),
            st.session_state.get("project_goal", {})
        )
        st.write("Evaluation Report:", eval_report)
        st.pyplot(visuals)
    
    # Step 7: Final Summary via Chat Agent
    if "model" in st.session_state:
        summary = chat_agent.summarize_results(eval_report, st.session_state["model"])
        st.write("Summary:", summary)

if __name__ == "__main__":
    main()
