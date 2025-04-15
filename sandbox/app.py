import streamlit as st
import pandas as pd
from agents.chat_agent import ChatAgent

st.set_page_config(page_title="Data Science Assistant", layout="centered")
st.title("🤖 Multi-Agent Data Science Assistant")

chat_agent = ChatAgent()

# --- Initialize session state ---
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "project_goal" not in st.session_state:
    st.session_state["project_goal"] = {}
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

# --- Handle user chat input ---
user_input = st.chat_input("Describe your project, and I'll help you step-by-step!")
if user_input:
    st.session_state["conversation_history"].append({"role": "user", "content": user_input})
    reply, goal = chat_agent.converse(st.session_state["conversation_history"])
    st.session_state["conversation_history"].append({"role": "assistant", "content": reply})
    st.session_state["project_goal"].update({k: v for k, v in goal.items() if v})
    st.chat_message("assistant").write(reply)

# --- Display chat history ---
for msg in st.session_state["conversation_history"]:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Check if the goal is complete ---
goal = st.session_state["project_goal"]
is_complete = chat_agent.is_complete(goal)

# --- Dataset logic ---
if is_complete:
    st.success("✅ Your project goal is clear!")
    if not st.session_state["dataset"]:
        st.write("### Do you already have a dataset?")
        use_recommendation = st.toggle("No, recommend one for me")
        if use_recommendation:
            st.info("Here are some datasets you might try:")
            st.write(chat_agent.recommend_dataset(goal))
        else:
            uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])
            if uploaded_file:
                st.session_state["dataset"] = pd.read_csv(uploaded_file)
                st.success("✅ Dataset uploaded successfully!")
                st.write("We'll now analyze your data and start preprocessing...")
                # 🧪 Call preprocessing agent next (to be implemented)
else:
    st.info("Please continue answering questions until your project goal is fully understood.")
