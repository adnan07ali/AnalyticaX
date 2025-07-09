import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------- #
def clean_data_ui(df):
    st.subheader("🔍 Step 1: View and Rename Columns")
    with st.expander("📌 View Data & Rename Columns"):
        st.dataframe(df.head())
        if st.checkbox("✏️ Rename Columns"):
            col_renames = {}
            for col in df.columns:
                new_col = st.text_input(f"Rename '{col}'", value=col)
                col_renames[col] = new_col
            df.rename(columns=col_renames, inplace=True)
            st.success("✅ Columns renamed successfully!")

    # ------------------------- #
    st.subheader("🧼 Step 2: Remove Duplicates")
    with st.expander("🧹 Remove Duplicate Rows"):
        if st.button("🚫 Remove Duplicates"):
            initial_shape = df.shape
            df.drop_duplicates(inplace=True)
            st.success(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

    # ------------------------- #
    st.subheader("❓ Step 3: Handle Missing Values")
    with st.expander("⚠️ Missing Value Treatment"):
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            for col in missing_cols:
                st.write(f"Column: `{col}` - Missing: {df[col].isnull().sum()} values")
                strategy = st.selectbox(f"Fill or drop '{col}'?", ["Drop", "Fill with Mean", "Fill with Median", "Fill with Mode"], key=col)
                if strategy == "Drop":
                    df.dropna(subset=[col], inplace=True)
                elif strategy == "Fill with Mean":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "Fill with Median":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "Fill with Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            st.success("✅ No missing values found!")

    # ------------------------- #
    st.subheader("🕒 Step 4: Datetime Conversion")
    with st.expander("📅 Convert Columns to Datetime"):
        non_dt_cols = [col for col in df.columns if not pd.api.types.is_datetime64_any_dtype(df[col])]
        datetime_cols = st.multiselect("Select columns to convert to datetime", non_dt_cols)
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col])
                st.success(f"Converted '{col}' to datetime.")
            except:
                st.error(f"Failed to convert '{col}' to datetime.")

    # ------------------------- #
    st.subheader("🧠 Step 5: Encode Categorical Variables")
    with st.expander("🔤 Label Encode Categorical Columns"):
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            cols_to_encode = st.multiselect("Select columns to label encode", cat_cols)
            for col in cols_to_encode:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            if cols_to_encode:
                st.success("✅ Label encoding applied.")
        else:
            st.info("No categorical columns found to encode.")

    # ------------------------- #
    st.subheader("🔢 Step 6: Scale/Transform Numeric Data")
    with st.expander("📈 Normalize or Standardize Numeric Columns"):
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cols_to_scale = st.multiselect("Select numeric columns to scale", num_cols, key='scale_cols')
        scaling_method = st.radio("Scaling method:", ["None", "Standard Scaling (Z-Score)", "Min-Max Normalization"])

        if scaling_method != "None" and cols_to_scale:
            if scaling_method == "Standard Scaling (Z-Score)":
                scaler = StandardScaler()
                df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                st.success("✅ Standard scaling applied.")
            elif scaling_method == "Min-Max Normalization":
                df[cols_to_scale] = (df[cols_to_scale] - df[cols_to_scale].min()) / (df[cols_to_scale].max() - df[cols_to_scale].min())
                st.success("✅ Min-max normalization applied.")

    # ------------------------- #
    st.subheader("🧨 Step 7: Outlier Removal")
    with st.expander("🚨 Detect and Remove Outliers (IQR Method)"):
        outlier_cols = st.multiselect("Select columns for outlier removal", num_cols, key='outlier_cols')
        if st.button("Remove Outliers"):
            for col in outlier_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                condition = ~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
                df = df[condition]
            st.success("✅ Outliers removed using IQR.")

    return df
