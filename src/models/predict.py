import joblib
import numpy as np
from src.signal_processing.fft_features import extract_features

model = joblib.load("saved_models/anomaly_model.pkl")


def detect_anomaly(signal):

    features = extract_features(signal)

    features = features.reshape(1, -1)

    prediction = model.predict(features)

    return "ANOMALY" if prediction[0] == -1 else "NORMAL"
