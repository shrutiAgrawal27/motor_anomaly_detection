import numpy as np
from scipy.fft import fft


def extract_features(signal):

    spectrum = np.abs(fft(signal))

    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "max_freq_energy": np.max(spectrum)
    }

    return features
