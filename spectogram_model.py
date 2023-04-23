import matplotlib.pyplot as plt
import os
import librosa.display
import numpy as np
import multiprocessing as mp
import time

GENRES = ['Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
WAV_DIR = './new_wav/'      # TODO: change based on .wavs we want

def generate_mel_spectrograms():
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


def generate_mel_spectrograms_for_one_genre(genre: str):
    start = time.time()

    song_num = 1
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

        print(str(song_num) + "\tmel spectrogram saved for  " + genre + ' /\t' + song_name, flush=True)
        song_num += 1
    
    end = time.time()
    message = "~~ thread generating mel spectrograms for  " + genre + "  took " + str((end-start)/60) + " minutes"
    print()
    print(message, flush=True)
    print()


if __name__ == '__main__':
    for genre in GENRES:
        p = mp.Process(target=generate_mel_spectrograms_for_one_genre, args=(genre,))
        p.start()

