import pandas as pd
import streamlit as st

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("❌ Unsupported file format! Please upload a CSV or Excel file.")
            return None

        # Auto-fix column names (remove leading/trailing whitespace)
        df.columns = [col.strip() for col in df.columns]
        return df

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None
