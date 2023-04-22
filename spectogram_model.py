import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pylab
import os
import wave
from pydub import AudioSegment
import librosa.display
import numpy as np

genres = ['Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
for genre in genres:
    for song in os.listdir('./monowav/' + genre):
        samples, sample_rate = librosa.load('./monowav/' + genre + '/' + song, sr=None)

        sgram = librosa.stft(samples)

        # use the mel-scale instead of raw frequency
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)

        # use the decibel scale to get the final Mel Spectrogram
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig('melspecs/' + genre + '/' + song[:-4] + '.png', bbox_inches='tight', pad_inches=0)
        print("done with " + genre + '/' + song)

