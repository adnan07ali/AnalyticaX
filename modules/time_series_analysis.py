import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# --- Stationarity ---
def test_stationarity(timeseries, window=12):
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()
    fig, ax = plt.subplots()
    ax.plot(timeseries, color='blue', label='Original')
    ax.plot(rolmean, color='red', label='Rolling Mean')
    ax.plot(rolstd, color='black', label='Rolling Std')
    ax.legend(loc='best')
    ax.set_title('Rolling Mean & Std')
    st.pyplot(fig)

    result = sm.tsa.adfuller(timeseries.dropna(), autolag='AIC')
    st.markdown(f"""
    **ADF Statistic:** {result[0]:.4f}  
    **p-value:** {result[1]:.4f}  
    **Critical Values:**  
    {result[4]}
    """)
    return result[1] <= 0.05


def difference_until_stationary(ts, max_diff=2):
    for d in range(max_diff + 1):
        if test_stationarity(ts):
            st.success(f"✅ Stationary after {d} differencing.")
            return ts, d
        ts = ts.diff().dropna()
    st.warning(f"⚠️ Not stationary after {max_diff} differencing.")
    return ts, max_diff


def plot_acf_pacf(series, lags=40):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series, lags=lags, ax=axes[0])
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[0].set_title('ACF')
    axes[1].set_title('PACF')
    st.pyplot(fig)

# --- ARIMA Forecast ---
def forecast_arima(ts, n_periods):
    model = auto_arima(ts, seasonal=False, stepwise=True, trace=False, suppress_warnings=True)
    st.success(f"Selected ARIMA Order: {model.order}")
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    return forecast, conf_int, model


# --- Multivariate LSTM Forecast ---
def prepare_multivariate_lstm_data(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def forecast_multivariate_lstm(df, target_column, feature_columns, n_periods=30, time_steps=10):
    df_numeric = df[feature_columns + [target_column]].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_numeric)

    target_idx = df_numeric.columns.get_loc(target_column)
    X_all = scaled
    y_all = scaled[:, target_idx]

    train_size = int(len(X_all) * 0.8)
    train_X, train_y = X_all[:train_size], y_all[:train_size]

    X_seq, y_seq = prepare_multivariate_lstm_data(train_X, train_y, time_steps)
    X_seq = X_seq.reshape((X_seq.shape[0], time_steps, X_seq.shape[2]))

    val_split = int(len(X_seq) * 0.8)
    X_val, y_val = X_seq[val_split:], y_seq[val_split:]
    X_train, y_train = X_seq[:val_split], y_seq[:val_split]

    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(time_steps, X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=0, callbacks=[early_stop])

    forecast_input = scaled[-time_steps:]
    forecast_seq = []

    for _ in range(n_periods):
        input_seq = forecast_input.reshape(1, time_steps, X_seq.shape[2])
        next_scaled = model.predict(input_seq, verbose=0)[0][0]

        forecast_seq.append(next_scaled)
        next_row = forecast_input[-1].copy()
        next_row[target_idx] = next_scaled
        forecast_input = np.vstack([forecast_input[1:], next_row])

    forecast_scaled = np.tile(forecast_seq, (scaled.shape[1], 1)).T
    forecast = scaler.inverse_transform(forecast_scaled)[:, target_idx]
    return forecast


# --- Run Analysis ---
def run_time_series_analysis(df):
    st.title("📈 Time Series Forecasting App")

    date_column = st.selectbox("Select Date Column", df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_column = st.selectbox("Select Target Column", numeric_cols)

    available_features = [col for col in numeric_cols if col != target_column]
    selected_features = st.multiselect("Select Additional Features for LSTM (optional)", available_features)

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column, target_column])
    df.set_index(date_column, inplace=True)
    df.sort_index(inplace=True)

    ts = df[target_column]
    freq = pd.infer_freq(ts.index) or 'D'

    st.write("### Raw Time Series Plot")
    fig, ax = plt.subplots()
    ts.plot(ax=ax)
    ax.set_title("Original Time Series")
    st.pyplot(fig)

    model_type = st.radio("Choose Forecasting Model:", ["ARIMA", "Multivariate LSTM"])
    n_periods = st.slider("🔮 Forecast Horizon", min_value=5, max_value=60, value=30)

    if model_type == "ARIMA":
        stationary_series, d = difference_until_stationary(ts)
        st.write("### ACF & PACF Plots")
        plot_acf_pacf(stationary_series)
        st.write("### Forecasting with ARIMA")
        forecast, conf_int, model = forecast_arima(ts, n_periods)
        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(1, unit=freq[0]), periods=n_periods, freq=freq)
        forecast_df = pd.DataFrame({
            'Forecast': forecast,
            'Lower CI': conf_int[:, 0],
            'Upper CI': conf_int[:, 1]
        }, index=forecast_index)
    else:
        st.write("### Forecasting with Multivariate LSTM")
        forecast = forecast_multivariate_lstm(df, target_column, selected_features, n_periods=n_periods)
        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(1, unit=freq[0]), periods=n_periods, freq=freq)
        forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

    st.write("### 📊 Forecast Plot")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ts.plot(ax=ax2, label='Historical')
    forecast_df['Forecast'].plot(ax=ax2, label='Forecast', color='green')
    if 'Lower CI' in forecast_df.columns:
        ax2.fill_between(forecast_index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='green', alpha=0.2)
    ax2.set_title("Forecast vs Historical")
    ax2.legend()
    st.pyplot(fig2)

    st.write("### 📉 Forecast Data")
    st.dataframe(forecast_df)

    st.write("### 📏 Evaluation Metrics")
    try:
        test_actual = ts[-n_periods:]
        test_forecast = model.predict_in_sample(start=len(ts) - n_periods, end=len(ts) - 1) if model_type == "ARIMA" else forecast[:len(test_actual)]
        rmse = np.sqrt(mean_squared_error(test_actual, test_forecast))
        mae = mean_absolute_error(test_actual, test_forecast)
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MAE", f"{mae:.2f}")
    except:
        st.warning("⚠️ Not enough data for evaluation.")


# --- Entry Point ---
def main():
    st.set_page_config(page_title="Time Series Forecasting", layout="wide")
    st.sidebar.title("📂 Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Upload your time series dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        run_time_series_analysis(df)
    else:
        st.info("📤 Please upload a CSV file to get started.")


if __name__ == "__main__":
    main()
