import numpy as np
import pandas as pd

fs = 1000
t = np.linspace(0, 1, fs)

signals = []

for i in range(200):

    signal = np.sin(2*np.pi*50*t)
    signal = signal + 0.2*np.random.randn(fs)

    signals.append(signal)

df = pd.DataFrame(signals)

df.to_csv("data/signals.csv", index=False)

print("Data saved")
