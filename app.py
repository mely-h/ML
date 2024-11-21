from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import io
import base64
import requests

app = Flask(__name__)


look_back = 3
model = Sequential([
    Input(shape=(look_back,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


def create_dataset(data, look_back=3):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
  
    url = "https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open("temperature_data.csv", "wb") as file:
            file.write(response.content)
    else:
        return f"Erreur lors du téléchargement du dataset : {response.status_code}", 500

  
    data = pd.read_csv("temperature_data.csv")
    if "Mean" not in data.columns:
        return "La colonne 'Mean' n'existe pas dans le dataset.", 400

    data['Mean'] = data['Mean'].fillna(method='ffill')  
    temps = data['Mean'].values

  
    scaler = MinMaxScaler()
    temps_scaled = scaler.fit_transform(temps.reshape(-1, 1))

   
    X = create_dataset(temps_scaled)
    predictions_scaled = model.predict(X)
    predictions = scaler.inverse_transform(predictions_scaled)

   
    plt.figure(figsize=(10, 6))
    plt.plot(temps[look_back:], label="Valeurs réelles")
    plt.plot(predictions, label="Prédictions", linestyle="dashed")
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Température Moyenne")
    plt.title("Prédictions des Températures")

  
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template("result.html", plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
