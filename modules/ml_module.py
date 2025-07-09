import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

sns.set(style="whitegrid")


def run_ml_module(df: pd.DataFrame):
    st.subheader("🧠 Machine Learning Module")

    target = st.selectbox("🎯 Select Target Column", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Handle datetime
        X = handle_datetime_features(X)

        # Detect problem type
        problem_type = detect_problem_type(y)

        # Encode features
        X = encode_df(X)

        # Encode target if classification
        y_encoded = encode_target(y) if problem_type == "classification" else y

        # Train-test split
        test_size = st.slider("🔀 Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)

        # Model Selection
        st.markdown("🧠 **Choose Your Model**")
        model_name, model = select_model(problem_type)

        if st.button("🚀 Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success(f"✅ Model trained as **{problem_type}** using **{model_name}**")
            show_metrics(problem_type, y_test, y_pred)

            # Feature Importance
            if problem_type == "regression" and hasattr(model, 'feature_importances_'):
                plot_feature_importance(model, X)

            # Show predictions
            st.subheader("🔎 Predictions Preview")
            preview = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred
            }).reset_index(drop=True)
            st.dataframe(preview.head())

            # Allow user to download trained model
            st.subheader("💾 Download Trained Model")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                joblib.dump(model, tmp.name)
                st.download_button(
                    label="📥 Download Model (.pkl)",
                    data=open(tmp.name, 'rb').read(),
                    file_name=f"{model_name.replace(' ', '_')}_model.pkl"
                )


# === Utility Functions ===

def detect_problem_type(y: pd.Series) -> str:
    if y.nunique() <= 15 or y.dtype == 'object' or y.dtype.name == 'category':
        return "classification"
    else:
        return "regression"


def encode_df(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.select_dtypes(include=["object", "category", "bool"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X


def encode_target(y: pd.Series) -> pd.Series:
    if y.dtype == 'object' or y.dtype.name == 'category':
        return LabelEncoder().fit_transform(y.astype(str))
    return y


def handle_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_minute'] = df[col].dt.minute
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                df.drop(columns=[col], inplace=True)
            except:
                pass
    return df


def select_model(problem_type: str):
    if problem_type == "classification":
        model_name = st.selectbox("📘 Select Classification Model", [
            "Random Forest", "Logistic Regression", "Decision Tree",
            "Support Vector Classifier", "K-Nearest Neighbors", "Gradient Boosting"
        ])
        if model_name == "Random Forest":
            return model_name, RandomForestClassifier()
        elif model_name == "Logistic Regression":
            return model_name, LogisticRegression()
        elif model_name == "Decision Tree":
            return model_name, DecisionTreeClassifier()
        elif model_name == "Support Vector Classifier":
            return model_name, SVC()
        elif model_name == "K-Nearest Neighbors":
            return model_name, KNeighborsClassifier()
        else:
            return model_name, GradientBoostingClassifier()
    else:
        model_name = st.selectbox("📗 Select Regression Model", [
            "Random Forest", "Linear Regression", "Decision Tree",
            "Support Vector Regressor", "K-Nearest Neighbors", "Gradient Boosting"
        ])
        if model_name == "Random Forest":
            return model_name, RandomForestRegressor()
        elif model_name == "Linear Regression":
            return model_name, LinearRegression()
        elif model_name == "Decision Tree":
            return model_name, DecisionTreeRegressor()
        elif model_name == "Support Vector Regressor":
            return model_name, SVR()
        elif model_name == "K-Nearest Neighbors":
            return model_name, KNeighborsRegressor()
        else:
            return model_name, GradientBoostingRegressor()


def show_metrics(problem_type: str, y_true, y_pred):
    st.subheader("📊 Evaluation Metrics")
    if problem_type == "classification":
        acc = accuracy_score(y_true, y_pred)
        st.write(f"🎯 Accuracy: `{round(acc * 100, 2)}%`")
        st.text("📋 Classification Report:")
        st.text(classification_report(y_true, y_pred))
    else:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        st.write(f"📉 MAE: `{round(mae, 4)}`")
        st.write(f"📉 RMSE: `{round(rmse, 4)}`")
        st.write(f"📈 R² Score: `{round(r2, 4)}`")


def plot_feature_importance(model, X):
    st.subheader("🔍 Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)
