import os
import librosa.display
import numpy as np
import csv
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt

def extract_features():
    for genre in os.listdir('wavfiles'):
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

            to_append = f'{genre} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spectral_centroids)} {np.mean(spec_bw)} {np.mean(spectral_rolloff)} {sum(zero_crossings)}'
            for e in mfccs:
                to_append += f' {np.mean(e)}'
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())


# values originall from https://arxiv.org/pdf/1804.01149.pdf? 
SR = 22050      # Sampling rate
N_FFT = 2048    # Frame/Window size
HOP_SIZE = 512  # Time advance between frames (512 results in 75% overlap)
WINDOW = 'hann' # Window Function: Hann Window
# Frequency Scale: MEL
# Number of MEL bins: 96
F_MAX = SR/2    # Highest Frequency

# WIP - 
#   probably just use Mel spectrograms from other file
def generate_spectrograms():
    filepath = './wavfiles/Pop/Woman.wav'
    sample_rate, samples = wav.read(filepath)
    left_channel = samples[:, 0]
    f, t, Zxx = signal.stft(left_channel, fs=sample_rate)
    
    plt.pcolormesh(t, f, np.abs(Zxx), cmap="jet", norm=LogNorm())
    plt.show()


if __name__ == '__main__':
    generate_spectrograms()

