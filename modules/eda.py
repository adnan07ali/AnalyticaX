import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

def run_eda(df: pd.DataFrame):
    st.subheader("📊 Dataset Overview")

    with st.expander("📋 Dataset Shape & Data Types"):
        st.write(f"🔢 Rows: `{df.shape[0]}`, 📁 Columns: `{df.shape[1]}`")
        st.write("🧪 Data Types:")
        st.dataframe(df.dtypes.astype(str).rename("Type"))

    with st.expander("📈 Summary Statistics"):
        st.write(df.describe(include='all').T)

    with st.expander("❓ Missing Value Report"):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.write(missing.to_frame("Missing Values"))
            st.bar_chart(missing)
        else:
            st.success("✅ No missing values detected.")

    with st.expander("📉 Distribution & Skewness"):
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            for col in num_cols:
                st.write(f"📊 **{col}** - Skewness: `{round(df[col].skew(), 3)}`")
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)
        else:
            st.info("No numeric columns available for distribution analysis.")

    with st.expander("🔁 Correlation Heatmap"):
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns to compute correlations.")

    with st.expander("🧬 Auto Pairplot (Max 5 Columns)"):
        if len(num_cols) >= 2:
            selected = st.multiselect("Select numeric columns for pairplot", num_cols, default=num_cols[:4])
            if len(selected) >= 2:
                fig = sns.pairplot(df[selected].dropna())
                st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for pairplot.")

    with st.expander("📌 Auto KPI Metrics"):
        kpi_candidates = df.select_dtypes(include=['int64', 'float64']).columns
        if len(kpi_candidates) >= 1:
            st.write("💡 Potential KPI metrics (mean):")
            cols = st.columns(min(4, len(kpi_candidates)))
            for i, col in enumerate(kpi_candidates[:4]):
                metric_val = round(df[col].mean(), 2)
                cols[i].metric(label=col, value=metric_val)
        else:
            st.info("No numeric fields found to extract KPIs.")
