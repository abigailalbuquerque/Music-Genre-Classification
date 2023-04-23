import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import soundfile as sf
import os
import librosa
import multiprocessing as mp

GENRES = ['Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
WAV_DIR = "./new_wav/"

def seperator (song, genre):
    # Loading test file
    y, sr = librosa.load(genre+"/"+song)

    #grab a sample of 20 seconds
    S_full, phase = librosa.magphase(librosa.stft(y))
    Audio(data=y[0 * sr:20 * sr], rate=sr)

    # Build filters based on cosine similarity, hopefully we get vocalish frequencies together
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)

    # Margins to reduce crossover
    margin_i, margin_v = 2, 10
    power = 2
    #edit above values for tweaking
    mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    #Save and export foreground audio
    y_foreground = librosa.istft(S_foreground * phase)
    Audio(data=y_foreground[0 * sr:20 * sr], rate=sr)
    sf.write(genre+'_separated/vocal_file_'+song+'.wav', y_foreground[0 * sr:20 * sr], sr, 'PCM_24')

    # Save and export background audio
    y_background = librosa.istft(S_background * phase)
    Audio(data=y_background[0 * sr:20 * sr], rate=sr)
    sf.write(genre+'_separated/instrument_file_'+song+'.wav', y_background[0 * sr:20 * sr], sr, 'PCM_24')


def vocal_separation_for_genre(genre):
    start = time.time()
    
    genre_dir = "./" + genre + "/"
    for song in os.listdir(genre_dir):
        seperator(song, genre)
    
    end = time.time()
    print("~~ thread separating vocals for  " + genre + "  took " + str((end-start)/60) + " minutes")


if __name__ == "__main__":
    for genre in GENRES:
        p = mp.Process(target=vocal_separation_for_genre, args=(genre,))
        p.start()
    sys.exit(0)

