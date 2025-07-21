# 📊 AnalyticaX – Data Analysis & Forecasting Web App

**AnalyticaX** is a full-stack interactive web application built with Streamlit that empowers users to upload CSV or Excel datasets, perform detailed data analysis, visualize patterns, and generate time series forecasts using ARIMA and LSTM models.


---

## 🚀 Features

- 📂 Upload `.csv` or `.xlsx` datasets
- 🧹 Auto data cleaning & summary
- 📊 EDA with plots: histograms, pairplots, heatmaps
- 📈 Customizable visualizations (bar, pie, line, box, scatter)
- 🧠 Regression modeling with train/test splits
- 📉 Time Series Forecasting:
  - ARIMA (with auto-differencing)
  - Multivariate LSTM (Keras)
- 📥 Export cleaned data to CSV or Excel
- 🔐 Deployed securely using Docker on Google Cloud VM

---

## 🧰 Tech Stack

- **Frontend & Backend**: Python, Streamlit
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Modeling**: Scikit-learn, Statsmodels, Pmdarima, TensorFlow/Keras
- **Deployment**: Docker, GCP Compute Engine

---

## 📦 Installation

```bash
git clone https://github.com/your-username/AnalyticaX.git
cd AnalyticaX
pip install -r requirements.txt
streamlit run app.py
