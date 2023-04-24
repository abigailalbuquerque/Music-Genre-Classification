import os
import pickle
import sys
import math
import librosa
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf

############################################################

#Generate CSV full of proba values for every song in wavfiles/

############################################################

# def proba_distance(song1_proba,song2_proba):
#     distance_squared = 0
#     for pair in zip(song1_proba,song2_proba):
#         distance_squared += math.pow(pair[0]-pair[1],2)
#     return math.sqrt(distance_squared)

GENRES = ['Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

def featurize(songfile):
    x, sr = librosa.load(songfile, sr=None)

    # Extracting features and putting them into an array
    # Zero Crossings
    n0 = 9000
    n1 = 9100
    zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
    #print(sum(zero_crossings))

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)
    #print(np.mean(spectral_centroids))

    # Spectral Roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
    #print(np.mean(spectral_rolloff))

    # Chroma Filters
    chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
    #print(np.mean(chroma_stft))

    # Root Mean Squared Energy
    rms = librosa.feature.rms(y=x)
    #print(np.mean(rms))

    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    #print(np.mean(spec_bw))

    # mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=x, sr=sr)
    #print(mfccs)

    to_append = f'{np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spectral_centroids)} {np.mean(spec_bw)} {np.mean(spectral_rolloff)} {sum(zero_crossings)}'
    for e in mfccs:
        to_append += f' {np.mean(e)}'
    return to_append.split()

def featurize_image(songfile):
    genre = songfile.split('/')[1]
    song_name = songfile.split('/')[2][:-4]
    if(os.path.isfile('melspecs/'+genre + '/' + song_name + '.png')):
        
        path = 'melspecs/'+genre + '/' + song_name + '.png'
    else:
        samples, sample_rate = librosa.load(songfile, sr=None)

        sgram = librosa.stft(samples)

        # use the mel-scale instead of raw frequency
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)

        # use the decibel scale to get the final Mel Spectrogram
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')

        # for saving image
        plt.axis('off')
        path = 'temp/' + song_name + '.png'
        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    img = tf.keras.utils.load_img(path)   
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return img_array

if (len(sys.argv)!= 3):
    print("Command line argument must contain 2 values: <modelpath>.sav <probas>.csv, the filepath to the classifier model to use \
          and the csv to store the probas for all songs available.")
    exit()


model = pickle.load(open(sys.argv[1],'rb'))
csvfile = open(sys.argv[2],'w')
csvwriter = csv.writer(csvfile)
scaler = pickle.load(open("scaler_model.sav",'rb'))
csvwriter.writerow(["Song Name", 'Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'])
count = 0
incorrect = 0
for genre in ['Country','Blues','Classical','Electronic','Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae','Rock']:
    for song in os.listdir('new_wav/' + genre):
        count += 1
        img_array = featurize_image("new_wav/" + genre + '/' + song)
        
        

        song_proba = model.predict(img_array)

        if GENRES[np.argmax(song_proba)] is not genre:
            incorrect += 1
            print('wrong ' + str(incorrect) + ' times')
        #print(song_proba)
        #print(GENRES)
        # song_proba = model.predict_proba(song_normal)
        # print(model.predict(song_normal))
        # #print(song)
        # print(song_proba)
        output_row = []
        output_row.append(song.strip('.wav'))
        for proba in song_proba[0]:
            output_row.append(proba)
        csvwriter.writerow(output_row)
##TODO: Featurize songs##

print(count)
csvfile.close()

##TODO: Get predict proba for all songs##

##TODO: Put predict probas and names into csv for export##






