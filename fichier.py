import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import requests

# Télécharger les données
url = "https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv"
file_name = "temperature_data.csv"

response = requests.get(url)
if response.status_code == 200:
    with open(file_name, "wb") as file:
        file.write(response.content)
    print(f"Dataset téléchargé et enregistré sous {file_name}")
else:
    print(f"Échec du téléchargement. Code d'erreur : {response.status_code}")

# Fonction pour créer des ensembles X et y
def create_dataset(data, look_back=3):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Charger et préparer les données
data = pd.read_csv("temperature_data.csv")
print("Colonnes disponibles :", data.columns)
print(data.head())

# Remplir les valeurs manquantes et extraire les températures moyennes
data['Mean'] = data['Mean'].fillna(method='ffill')
temps = data['Mean'].values

# Créer les ensembles X et y avec un look_back de 3
look_back = 3
X, y = create_dataset(temps, look_back)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# Définir le modèle avec Input(shape=...)
model = Sequential([
    Input(shape=(look_back,)),  # Remplace input_dim
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compiler le modèle
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Prédictions
y_pred = model.predict(X_test)
y_test_inversed = scaler_y.inverse_transform(y_test)
y_pred_inversed = scaler_y.inverse_transform(y_pred)

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(y_test_inversed, label='Valeurs réelles (Mean Temp)', color='blue')
plt.plot(y_pred_inversed, label='Valeurs prédites (Mean Temp)', color='orange', linestyle='dashed')
plt.title('Prédiction de la température moyenne avec un MLP')
plt.xlabel('Index')
plt.ylabel('Température moyenne (°C)')
plt.legend()
plt.show()

# Évaluation du modèle
mse, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Mean Absolute Error (MAE) : {mae:.4f}")
