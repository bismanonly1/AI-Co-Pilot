import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = OllamaLLM(model="llama3")

# Modern chain construction
target_chain = (
    ChatPromptTemplate.from_template("""
    You're a helpful assistant. A user uploaded a dataset with these columns:
    {columns}
    
    Which column is most likely the target for ML? Reply with just the column name and brief reason.
    """)
    | llm
    | StrOutputParser()
)

def guess_target_column(df):
    columns_str = ", ".join(df.columns)
    return target_chain.invoke({"columns": columns_str})

def detect_task_type(df, target_column):
    nunique = df[target_column].nunique()
    dtype = df[target_column].dtype
    
    if nunique <= 20 and dtype in ['int64', 'object']:
        return "classification"
    elif pd.api.types.is_numeric_dtype(dtype):
        return "regression"
    else:
        return "unknown"

def suggest_preprocessing_steps(df):
    suggestions = []
    
    # Missing values
    null_cols = df.columns[df.isnull().any()]
    if len(null_cols) > 0:
        suggestions.append(
            f"Impute missing values in: {', '.join(null_cols)} "
            f"(mean/median for numeric, most frequent for categorical)"
        )
    
    # Categorical encoding
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        suggestions.append(f"One-hot encode categorical columns: {', '.join(cat_cols)}")
    
    # Numeric scaling
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) > 0:
        suggestions.append(f"Scale numeric columns: {', '.join(num_cols)}")
    
    return suggestions

def preprocess(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    num_cols = X.select_dtypes(include='number').columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if len(num_cols) > 0:
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
    if len(cat_cols) > 0:
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # One-hot encode
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(X[cat_cols])
        X = pd.concat([
            X.drop(columns=cat_cols),
            pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
        ], axis=1)

    # Scale numeric features
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X, y

def agent_chat_prompt(df, target_column):
    task = detect_task_type(df, target_column)
    steps = suggest_preprocessing_steps(df)
    
    return f"""
## Preprocessing Analysis
**Task Type:** {task.capitalize()}
**Target Column:** '{target_column}'

### Recommended Steps:
{chr(10).join(f'- {s}' for s in steps)}

Would you like to apply these preprocessing steps automatically?
"""