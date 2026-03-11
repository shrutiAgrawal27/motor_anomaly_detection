import numpy as np
from scipy.fft import fft


def extract_features(signal):

    spectrum = np.abs(fft(signal))

    features = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.mean(spectrum),
        np.max(spectrum)
    ]

    return np.array(features)
