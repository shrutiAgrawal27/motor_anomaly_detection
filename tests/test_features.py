from src.signal_processing.fft_features import extract_features
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_feature_shape():

    signal = np.random.randn(100)

    features = extract_features(signal)

    assert len(features) == 5
