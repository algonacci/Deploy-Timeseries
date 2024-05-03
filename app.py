from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

beras_model = load_model('beras_model.h5')


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        future_days = int(request.form["future_days"])
        data = request.files["data"]
        df = pd.read_csv(data, delimiter=";")
        df['Tanggal'] = df['Tanggal'].str.replace(' ', '')
        df['Tanggal'] = df['Tanggal'].str.replace('/', '-')
        df.interpolate(method='linear', inplace=True)
        df['Tx'].fillna(method='bfill', inplace=True)
        tanggal = pd.to_datetime(df['Tanggal'], format="%d-%m-%Y")
        price = df['Tx'].astype(float)
        series = np.array(price)

        scaler = MinMaxScaler(feature_range=(0, 1))
        series = series.reshape(-1, 1)
        series = scaler.fit_transform(series)

        time = np.array(tanggal)
        ssplit = 1024

        time_train = time[:ssplit]
        x_train = series[:ssplit]
        time_valid = time[ssplit:]
        x_valid = series[ssplit:]

        x_train = (x_train - np.min(x_train)) / \
            (np.max(x_train) - np.min(x_train))
        x_valid = (x_valid - np.min(series)) / \
            (np.max(series) - np.min(series))

        input_data = x_valid[:30][np.newaxis]

        future_forecast = []

        for _ in range(future_days):
            prediction = beras_model.predict(input_data)

            # Append the prediction to the future forecast
            future_forecast.append(prediction[0, 0])

            # Update the input data by removing the first element and appending the predicted value
            input_data = np.append(input_data[:, 1:, :], [
                                   [prediction[0, 0]]], axis=1)

        future_forecast = np.array(future_forecast)

        x_train_original = scaler.inverse_transform(x_train)
        x_valid_original = scaler.inverse_transform(x_valid)
        forecast_original = scaler.inverse_transform(future_forecast)

        plt.figure(figsize=(10, 6))
        plt.plot(time_train, x_train_original, label='Actual')
        plt.plot(time_valid, x_valid_original, label='Test')
        # Plot forecasted values for the future_days
        future_dates = pd.date_range(
            # Generate future dates
            start=time_valid[-1], periods=future_days+1)[1:]
        plt.plot(future_dates, forecast_original,
                 label='Forecast', linestyle='dashed')

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Actual vs Forecast')
        plt.legend()
        plt.savefig("tes.png")
        return render_template("pages/index.html")
    else:
        return render_template("pages/index.html")


if __name__ == "__main__":
    app.run(debug=True)
