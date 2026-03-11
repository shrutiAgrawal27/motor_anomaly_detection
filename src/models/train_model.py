import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from src.signal_processing.fft_features import extract_features

df = pd.read_csv("data/signals.csv")

X = []

for _, row in df.iterrows():

    signal = row.values

    features = extract_features(signal)

    X.append(features)

X = np.array(X)

model = IsolationForest(contamination=0.05)

model.fit(X)

joblib.dump(model, "saved_models/anomaly_model.pkl")

print("Model trained and saved")
