import os
import librosa
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


file_name = 'C:/Users/taitym/Downloads/jko.mp3'
res = librosa.load(file_name)
y, sr = librosa.load(file_name, mono = False)


print(y.shape[0] / sr)
print(y[0])
print(y.shape)
sns.lineplot(abs(y[0]))
#fcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=8192, n_mfcc=12)
