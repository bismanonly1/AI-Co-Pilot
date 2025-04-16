import streamlit as st
import pandas as pd
from agents.chat_agent import ChatAgent
from agents.preprocessing_agent import PreprocessingAgent

st.set_page_config(page_title="Data Science Assistant", layout="centered")
st.title("🤖 Multi-Agent Data Science Assistant")

chat_agent = ChatAgent()
preprocessing_agent = PreprocessingAgent()

# --- Initialize session state ---
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "project_goal" not in st.session_state:
    st.session_state["project_goal"] = {}
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None
if "preprocessed" not in st.session_state:
    st.session_state["preprocessed"] = False

# --- Upload dataset FIRST ---
st.write("### Step 1: Upload your dataset (CSV only)")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file and st.session_state["dataset"] is None:
    df = pd.read_csv(uploaded_file)
    st.session_state["dataset"] = df
    st.success("✅ Dataset uploaded successfully!")

    dataset_summary = chat_agent.inspect_dataset(df)
    st.session_state["conversation_history"].append({"role": "assistant", "content": dataset_summary})

# --- Chat input from user ---
user_input = st.chat_input("Ask me about your dataset or tell me what you'd like to build!")

if user_input:
    st.session_state["conversation_history"].append({"role": "user", "content": user_input})
    reply, goal = chat_agent.converse(st.session_state["conversation_history"])
    st.session_state["conversation_history"].append({"role": "assistant", "content": reply})
    st.session_state["project_goal"].update({k: v for k, v in goal.items() if v})

# --- Display full conversation ---
for msg in st.session_state["conversation_history"]:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Determine readiness ---
goal = st.session_state["project_goal"]
df = st.session_state["dataset"] if "dataset" in st.session_state else None
is_complete = chat_agent.is_complete(goal)

if is_complete and df is not None:
    st.success("✅ Your project goal is clear and your dataset is uploaded!")

    if not st.session_state["preprocessed"]:
        target_col = goal.get("target")
        summary = preprocessing_agent.generate_summary(df, target_col)
        st.markdown(summary)

        if st.button("Apply Preprocessing Steps"):
            X, y = preprocessing_agent.preprocess(df, target_col)
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["preprocessed"] = True
            st.success("✨ Preprocessing complete!")
            st.write("### Sample of Preprocessed Features")
            st.dataframe(X.head())
            st.write("### Sample of Target Values")
            st.write(y.head())
    else:
        st.write("✅ Preprocessing already done.")
        st.dataframe(st.session_state["X"].head())
else:
    if df is None or df.empty:
        st.info("Please upload a dataset to begin.")
    else:
        st.info("Let's keep chatting until I have everything I need to help you build your model ✨")
