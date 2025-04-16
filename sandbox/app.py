import streamlit as st
import pandas as pd
from agents.chat_agent import ChatAgent
from agents.preprocessing_agent import PreprocessingAgent
from utils.data_helpers import detect_task_type
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
    st.success("✅ Dataset uploaded!")

    message, target, task = chat_agent.inspect_dataset(df)
    st.session_state["conversation_history"].append({"role": "assistant", "content": message})

    st.session_state["suggested_target"] = target
    st.session_state["suggested_task"] = task
    st.session_state["show_suggestion"] = True
    st.session_state["confirmed_target_task"] = False

# Suggestion phase
if st.session_state.get("show_suggestion") and not st.session_state.get("confirmed_target_task"):
    st.markdown(f"""
    ### My Suggestion:
    - **Target column:** `{st.session_state["suggested_target"]}`
    - **Task type:** `{st.session_state["suggested_task"]}`

    Would you like to continue with these?
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, let's proceed"):
            st.session_state["project_goal"] = {
                "target": st.session_state["suggested_target"],
                "task": st.session_state["suggested_task"],
                "features": []
            }
            st.session_state["confirmed_target_task"] = True
    with col2:
        if st.button("❌ No, choose another target"):
            st.session_state["manual_target_select"] = True
            st.session_state["confirmed_target_task"] = False

if st.session_state.get("manual_target_select") and not st.session_state.get("confirmed_target_task"):
    st.markdown("### Choose your own target column:")
    selected = st.selectbox("Select target column", st.session_state["dataset"].columns)

    if st.button("Confirm Target Column"):
        custom_target = selected
        custom_task = detect_task_type(st.session_state["dataset"], custom_target)

        st.session_state["project_goal"] = {
            "target": custom_target,
            "task": custom_task,
            "features": []
        }
        st.session_state["confirmed_target_task"] = True
        st.success(f"✅ You selected '{custom_target}' as target for a {custom_task} task.")


# Now ready to show preprocessing summary
df = st.session_state["dataset"]
goal = st.session_state["project_goal"]

if df is not None and st.session_state.get("confirmed_target_task") and not st.session_state.get("preprocessed"):
    target_col = goal["target"]
    summary = preprocessing_agent.generate_summary(df, target_col)
    st.markdown(summary)

    custom_mode = st.radio("Choose how to apply preprocessing:",
                       ["Full Auto", "Custom"])

    apply_missing = True
    apply_encoding = True
    apply_scaling = True

    if custom_mode == "Custom":
        steps = preprocessing_agent.suggest_preprocessing_steps(df)

        apply_missing = any("Impute" in s for s in steps)
        apply_encoding = any("One-hot" in s for s in steps)
        apply_scaling = any("Scale" in s for s in steps)

        st.write("Select which preprocessing steps to apply:")
        apply_missing = st.checkbox("Handle missing values", value=apply_missing)
        apply_encoding = st.checkbox("Encode categorical variables", value=apply_encoding)
        apply_scaling = st.checkbox("Scale numeric features", value=apply_scaling)


    if st.button("Apply Preprocessing Steps"):
        X, y = preprocessing_agent.preprocess(
            df,
            target_column=target_col,
            apply_missing=apply_missing,
            apply_encoding=apply_encoding,
            apply_scaling=apply_scaling
        )
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["preprocessed"] = True
        st.success("✅ Preprocessing complete!")
        st.write("### Sample of Preprocessed Features")
        st.dataframe(X.head())
        st.write("### Sample of Target Values")
        st.write(y.head())

