import datetime
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.dates as mdates
import matplotlib
import os

matplotlib.use('agg')

app = Flask(__name__)
model = load_model('model.h5')


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        start_date = datetime.datetime.strptime(
            request.form['start_date'], '%Y-%m-%d')
        end_date = datetime.datetime.strptime(
            request.form['end_date'], '%Y-%m-%d')

        # Hitung perbedaan jumlah hari
        delta = end_date - start_date
        future_days = delta.days + 1  # Include the end date

        # Baca data dari file CSV
        df = pd.read_csv("data_TA.csv", delimiter=";")
        df['Tanggal'] = df['Tanggal'].str.replace(' ', '')
        df['Tanggal'] = df['Tanggal'].str.replace('/', '-')
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format="%d-%m-%Y")

        # Interpolasi dan isi nilai yang hilang
        df.interpolate(method='linear', inplace=True)
        df['Tx'].fillna(method='bfill', inplace=True)

        # Mengambil kolom tanggal dan suhu
        tanggal = df['Tanggal']
        temperature = df['Tx'].astype(float)

        # Normalisasi data suhu
        series = np.array(temperature)
        scaler = MinMaxScaler(feature_range=(0, 1))
        series = scaler.fit_transform(series.reshape(-1, 1))

        # Pembagian data untuk training dan validasi
        time = np.array(tanggal)
        ssplit = 1024
        time_train, time_valid = time[:ssplit], time[ssplit:]
        x_train, x_valid = series[:ssplit], series[ssplit:]
        x_train = (x_train - np.min(x_train)) / \
            (np.max(x_train) - np.min(x_train))
        x_valid = (x_valid - np.min(series)) / \
            (np.max(series) - np.min(series))

        # Prediksi data
        input_data = x_valid[:30][np.newaxis]
        future_forecast = []
        for _ in range(future_days):
            prediction = model.predict(input_data)
            future_forecast.append(prediction[0, 0])
            input_data = np.append(input_data[:, 1:, :], [
                                   [prediction[0, 0]]], axis=1)

        future_forecast = np.array(future_forecast)
        forecast_original = scaler.inverse_transform(future_forecast)

        # Mengatur tanggal prediksi berdasarkan input tanggal awal
        future_dates = pd.date_range(start=start_date, periods=future_days)

        # Plot hanya data prediksi
        plt.figure(figsize=(15, 10))
        plt.plot(future_dates, forecast_original,
                 label='Forecast', linestyle='dashed')
        plt.scatter(future_dates, forecast_original, color='red')

        for i, txt in enumerate(forecast_original):
            plt.annotate(f'{txt[0]:.2f}', (future_dates[i], forecast_original[i]),
                         textcoords="offset points", xytext=(0, 5), ha='center')

        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.xticks(rotation=45)  # Rotate x-axis labels
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.title('Forecast')
        plt.legend()
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        image_path = f"static/{now}.png"
        plt.savefig(image_path)

        with open("history.txt", "a") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            file.write(f"{timestamp}, \
                       {future_days}, \
                       {forecast_original[-1][0]}, {image_path}\n")

        return render_template("pages/result.html",
                               forecast=forecast_original[-1][0],
                               future_days=future_days,
                               image_path=image_path)
    else:
        return render_template("pages/index.html")


@app.route("/history")
def history():
    history_data = []
    with open("history.txt", "r") as file:
        for line in file:
            timestamp, future_days, forecast, image_path = line.strip().split(", ")
            history_data.append({
                "timestamp": timestamp,
                "future_days": future_days,
                "forecast": forecast,
                "image_path": image_path
            })

    return render_template("pages/history.html", history=history_data)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
