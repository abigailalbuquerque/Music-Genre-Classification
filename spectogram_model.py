import matplotlib.pyplot as plt
import os
import librosa.display
import numpy as np

GENRES = ['Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
WAV_DIR = './monowav'   # TODO: change to go through new .wav files once we have them

for genre in GENRES:
    for song in os.listdir(WAV_DIR + genre):
        song_name = song[:-4]
        samples, sample_rate = librosa.load(WAV_DIR + genre + '/' + song, sr=None)

        sgram = librosa.stft(samples)

        # use the mel-scale instead of raw frequency
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)

        # use the decibel scale to get the final Mel Spectrogram
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')

        # for saving image
        plt.axis('off')
        plt.savefig('melspecs/' + genre + '/' + song_name + '.png', bbox_inches='tight', pad_inches=0)

        print("mel spectrogram saved for  " + genre + ' / ' + song_name)

