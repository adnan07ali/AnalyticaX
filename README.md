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

 Demo Workflow
## 🖥️ Demo Workflow

1. Upload any `.csv` or `.xlsx` dataset  
2. Explore descriptive statistics  
3. Visualize data with dynamic charts  
4. Run regression models and evaluate  
5. Forecast time series with ARIMA/LSTM  
6. Download cleaned data  


## 📦 Installation & Usage

### 🔧 Local Setup

git clone https://github.com/adnan07ali/AnalyticaX.git
cd AnalyticaX
pip install -r requirements.txt
streamlit run app.py


# Docker Setup

### Build the container
docker build -t analytica-x .

### Run the app
docker run -p 8501:8501 analytica-x



🔍 Modules Overview
eda.py
Descriptive statistics (mean, median, mode, std, etc.)

Null value summary and visualizations

Histograms, boxplots, and distribution plots

Correlation heatmap and pairplot

Summary KPIs using Streamlit metrics

modeling.py
Train/Test split and feature-target selection

Linear Regression, Decision Tree, and Random Forest

RMSE, MAE, and R² evaluation metrics

Tabular and graphical results

visualization.py
Bar, Pie, Line, Box, Histogram, and Scatter plots

Auto-detection of categorical vs numerical columns

Dynamic color picker for customization

time_series_analysis.py
Stationarity test with ADF and rolling stats

ACF & PACF plotting for ARIMA model selection

Forecasting with:

Auto ARIMA (via pmdarima)

Multivariate LSTM (via Keras + TensorFlow)

Forecast visualization and confidence intervals

Evaluation: RMSE and MAE

utils.py
Export cleaned datasets to CSV and Excel

Exception handling for download errors

