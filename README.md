# 📊 AnalyticaX – Streamlit Data Analysis & Forecasting Web App

**AnalyticaX** is a powerful and interactive web-based data analysis platform built using **Python** and **Streamlit**, designed for data enthusiasts, analysts, and teams. Users can upload datasets, explore and visualize data, clean it, apply machine learning models, and forecast future trends using **ARIMA** and **LSTM** models.

> 🌐 Fully Dockerized & deployed on Google Cloud Platform (GCP) Compute Engine VM.

---

## 🚀 Features

✅ Upload CSV or Excel datasets (.csv, .xlsx)  
✅ Automated data cleaning with summary statistics  
✅ Interactive visualizations (bar, pie, line, box, histogram, scatter)  
✅ Exploratory Data Analysis (EDA) with pairplot & heatmap  
✅ Regression modeling (train/test split + evaluation metrics)  
✅ Time Series Forecasting using:
- ARIMA (auto-ARIMA with stationarity check)
- Multivariate LSTM with additional features  
✅ Download cleaned dataset as CSV or Excel  
✅ Professional UI with color pickers and layout optimization  
✅ Docker support for containerization  
✅ Production-ready deployment on GCP VM instance

---

## 🧰 Tech Stack

| Layer        | Technology                      |
|--------------|----------------------------------|
| Language     | Python 3.11                      |
| Web Framework| Streamlit 1.35.0                 |
| Data Handling| Pandas, NumPy                    |
| Visualization| Plotly, Seaborn, Matplotlib      |
| ML & Forecast| Scikit-learn, Statsmodels, Pmdarima, TensorFlow/Keras |
| Export Tools | Openpyxl, base64, io             |
| Deployment   | Docker, GCP Compute Engine (Debian VM) |

---

## 📂 Folder Structure

Web-Project/
│
├── app.py # Main Streamlit launcher
├── Dockerfile # Docker container setup
├── requirements.txt # Python dependencies
├── README.md # Project overview
│
└── modules/ # Modular Streamlit sections
├── eda.py # Exploratory Data Analysis
├── modeling.py # Regression modeling
├── visualization.py # Interactive charts
├── time_series_analysis.py # ARIMA and LSTM forecasting
└── utils.py # Download helpers (CSV/Excel)

yaml
Copy
Edit

---

## 🖥️ Demo

🧪 Upload any .csv or .xlsx dataset and explore:
- Descriptive stats
- EDA graphs
- Modeling tabs
- Time series forecasts



---

## 📦 Installation & Usage

### 🔧 Local Setup

bash
git clone https://github.com/your-username/AnalyticaX.git
cd AnalyticaX
pip install -r requirements.txt
streamlit run app.py
🐳 Docker Setup
bash
Copy
Edit
# Build the container
docker build -t analytica-x .

# Run the app
docker run -p 8501:8501 analytica-x
☁️ Deployment on Google Cloud VM
Create a Compute Engine VM (Debian 11) with port 8501 open

SSH into the instance and install Docker

Upload the project using scp or Google Cloud Shell

Build and run Docker container

bash
Copy
Edit
sudo apt update && sudo apt install docker.io
git clone https://github.com/your-username/AnalyticaX.git
cd AnalyticaX
sudo docker build -t analytica-x .
sudo docker run -d -p 8501:8501 analytica-x
Access your app at: http://[VM_EXTERNAL_IP]:8501

🔍 Modules Overview
eda.py
Descriptive statistics (mean, median, etc.)

Pairplot, heatmap, histograms

Column-wise null analysis and imputation options

modeling.py
Regression (Linear, RandomForest, DecisionTree)

Train-test split (adjustable)

RMSE, MAE, R² score evaluation

visualization.py
Bar, pie, line, box, histogram, scatter plots

Custom color picker

Categorical/numeric column detection

time_series_analysis.py
ARIMA forecasting with auto_arima and stationarity check

Multivariate LSTM using Keras with early stopping

ACF/PACF plot generation

utils.py
Export cleaned dataset to downloadable CSV or Excel

Error handling for file generation

💡 Future Enhancements
Classification models support

Auto data profiling (pandas-profiling)

Export visualizations as image files

Integrated authentication (login/signup)

Dark/light theme switch

🧠 Learning Outcomes
Modular Streamlit web app design

Dockerizing full ML pipelines

Time series forecasting using ARIMA and LSTM

Production-grade deployment using GCP VM

Data preprocessing, EDA, modeling, and visualization best practices

📜 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Adnan Ali
📫 LinkedIn | 🌐 Portfolio (Optional)

📌 Project Tags
#streamlit #docker #gcp #data-analysis #eda #lstm #arima #ml #forecasting #python
