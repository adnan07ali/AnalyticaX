# 📊 AnalyticaX – Streamlit Data Analysis & Forecasting Web App

**AnalyticaX** is a powerful and interactive web-based data analysis platform built using **Python** and **Streamlit**, designed for data enthusiasts, analysts, and teams. Users can upload datasets, explore and visualize data, clean it, apply machine learning models, and forecast future trends using **ARIMA** and **LSTM** models.

> 🌐 Fully Dockerized & deployed on Google Cloud Platform (GCP) Compute Engine VM.

---

## 🚀 Features

- ✅ Upload CSV or Excel datasets (`.csv`, `.xlsx`)
- ✅ Automated data cleaning with summary statistics
- ✅ Interactive visualizations (Bar, Pie, Line, Box, Histogram, Scatter)
- ✅ Exploratory Data Analysis (EDA) with pairplot & heatmap
- ✅ Regression modeling (train/test split + evaluation metrics)
- ✅ Time Series Forecasting using:
  - ARIMA (auto-ARIMA with stationarity check)
  - Multivariate LSTM with additional features
- ✅ Download cleaned dataset as CSV or Excel
- ✅ Professional UI with color pickers and layout optimization
- ✅ Docker support for containerization
- ✅ Production-ready deployment on GCP VM instance

---

## 🧰 Tech Stack

| Layer         | Technology                                         |
|---------------|----------------------------------------------------|
| Language      | Python 3.11                                        |
| Web Framework | Streamlit 1.35.0                                   |
| Data Handling | Pandas, NumPy                                      |
| Visualization | Plotly, Seaborn, Matplotlib                        |
| ML & Forecast | Scikit-learn, Statsmodels, Pmdarima, TensorFlow/Keras |
| Export Tools  | Openpyxl, base64, io                               |
| Deployment    | Docker, GCP Compute Engine (Debian VM)             |

---

## 📂 Folder Structure
## 📂 Folder Structure

```
Web-Project/
│
├── app.py                      # Main Streamlit launcher
├── Dockerfile                  # Docker container setup
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
│
└── modules/                    # Modular Streamlit sections
    ├── eda.py                  # Exploratory Data Analysis
    ├── modeling.py             # Regression modeling
    ├── visualization.py        # Interactive charts
    ├── time_series_analysis.py # ARIMA and LSTM forecasting
    └── utils.py                # Download helpers (CSV/Excel)
```
