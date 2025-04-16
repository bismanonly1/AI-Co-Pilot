import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class PreprocessingAgent:
    def __init__(self):
        pass

    def guess_target_column(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if "target" in col.lower() or "label" in col.lower() or "quality" in col.lower():
                return col
        return df.columns[-1]  # fallback to last column

    def detect_task_type(self, df: pd.DataFrame, target_column: str) -> str:
        nunique = df[target_column].nunique()
        dtype = df[target_column].dtype

        if nunique <= 20 and dtype in ['int64', 'object']:
            return "classification"
        elif pd.api.types.is_numeric_dtype(dtype):
            return "regression"
        else:
            return "unknown"

    def suggest_preprocessing_steps(self, df: pd.DataFrame) -> list:
        suggestions = []
        null_cols = df.columns[df.isnull().any()]
        if len(null_cols) > 0:
            suggestions.append(
                f"Impute missing values in: {', '.join(null_cols)} (mean/median for numeric, most frequent for categorical)"
            )
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            suggestions.append(f"One-hot encode categorical columns: {', '.join(cat_cols)}")
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            suggestions.append(f"Scale numeric columns: {', '.join(num_cols)}")
        return suggestions

    def preprocess(self, df: pd.DataFrame, target_column: str) -> tuple:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        num_cols = X.select_dtypes(include='number').columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        if len(num_cols) > 0:
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
        if len(cat_cols) > 0:
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

        if len(cat_cols) > 0:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded = encoder.fit_transform(X[cat_cols])
            X = pd.concat([
                X.drop(columns=cat_cols),
                pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
            ], axis=1)

        if len(num_cols) > 0:
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])

        return X, y

    def generate_summary(self, df: pd.DataFrame, target_column: str) -> str:
        task = self.detect_task_type(df, target_column)
        steps = self.suggest_preprocessing_steps(df)

        return f"""
## Preprocessing Analysis
**Task Type:** {task.capitalize()}
**Target Column:** '{target_column}'

### Recommended Steps:
{chr(10).join(f'- {s}' for s in steps)}

Would you like to apply these preprocessing steps automatically?
"""
