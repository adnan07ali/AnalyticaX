import streamlit as st
import pandas as pd

from modules.data_upload import load_data
from modules.data_cleaning import clean_data_ui
from modules.eda import run_eda
from modules.visualization import run_visualization 
from modules.utils import generate_download_link, generate_excel_download_link

# ----------------- Streamlit Page Config ----------------- #
st.set_page_config(
    page_title="DataWiz Pro - Data Analysis App",
    layout="wide",
    page_icon="📊"
)

# ----------------- App Header ----------------- #
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>📊 DataWiz Pro - Advanced Data Analysis App</h1>", unsafe_allow_html=True)
st.markdown("---")

# ----------------- Sidebar Navigation ----------------- #
menu = st.sidebar.radio("📌 Navigation", ["Upload", "Clean Data", "EDA", "Visualize", "Time Series", "Download", "Machine Learning"])


# ----------------- Session State to hold DataFrame ----------------- #
if 'df' not in st.session_state:
    st.session_state.df = None

# ----------------- Module Routing ----------------- #
if menu == "Upload":
    st.header("📂 Upload CSV or Excel File")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.success("✅ File uploaded and loaded successfully!")
            st.write("🔍 Preview of your data:")
            st.dataframe(st.session_state.df.head())

elif menu == "Clean Data":
    st.header("🧹 Data Cleaning")
    if st.session_state.df is not None:
        st.session_state.df = clean_data_ui(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a file first from the 'Upload' section.")

elif menu == "EDA":
    st.header("📈 Exploratory Data Analysis")
    if st.session_state.df is not None:
        run_eda(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a file first from the 'Upload' section.")

elif menu == "Visualize":
    st.header("📊 Data Visualizations")
    if st.session_state.df is not None:
        run_visualization(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a file first from the 'Upload' section.")

elif menu == "Download":
    st.header("💾 Download Cleaned Dataset")
    if st.session_state.df is not None:
        st.markdown("✅ Click below to download your cleaned dataset:")
        st.markdown(generate_download_link(st.session_state.df), unsafe_allow_html=True)
        st.markdown(generate_excel_download_link(st.session_state.df), unsafe_allow_html=True)
        st.dataframe(st.session_state.df.head())
    else:
        st.warning("⚠️ Please upload and clean a file before downloading.")
elif menu == "Machine Learning":
    from modules.ml_module import run_ml_module
    if st.session_state.df is not None:
        run_ml_module(st.session_state.df)
    else:
        st.warning("⚠️ Please upload and clean a dataset first.")
elif menu == "Time Series":
    from modules.time_series_analysis import run_time_series_analysis
    if st.session_state.df is not None:
        run_time_series_analysis(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a dataset first.")


# ----------------- Footer ----------------- #
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit | © 2025 DataWiz Pro</p>", unsafe_allow_html=True)     