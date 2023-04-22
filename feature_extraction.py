import os
import librosa.display
import numpy as np
import csv
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt

genres = ['Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
for genre in genres:
    for song in os.listdir('wavfiles/' + genre):
        # loads the audio file and decodes it into a 1D array which is a time series x
        # sr is sampling rate, which is none in this case because we want the whole sample to be used
        x, sr = librosa.load('wavfiles/' + genre + '/' + song, sr=None)

        # Extracting features and putting them into the csv
        # Zero Crossings
        n0 = 9000
        n1 = 9100
        zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
        print(sum(zero_crossings))

        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)
        print(np.mean(spectral_centroids))

        # Spectral Roll-off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
        print(np.mean(spectral_rolloff))

        # Chroma Filters
        chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
        print(np.mean(chroma_stft))

        # Root Mean Squared Energy
        rms = librosa.feature.rms(y=x)
        print(np.mean(rms))

        # Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
        print(np.mean(spec_bw))

        # mel-frequency cepstral coefficients
        mfccs = librosa.feature.mfcc(y=x, sr=sr)
        print(mfccs)

        to_append = f'{genre} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spectral_centroids)} {np.mean(spec_bw)} {np.mean(spectral_rolloff)} {sum(zero_crossings)} '
        for e in mfccs:
            to_append += f' {np.mean(e)}'
        file = open('data_with_3.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())