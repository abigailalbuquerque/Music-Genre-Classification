import os
import pickle
import sys
import math
import librosa
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler

############################################################

#Generate CSV full of proba values for every song in wavfiles/

############################################################

# def proba_distance(song1_proba,song2_proba):
#     distance_squared = 0
#     for pair in zip(song1_proba,song2_proba):
#         distance_squared += math.pow(pair[0]-pair[1],2)
#     return math.sqrt(distance_squared)

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

if (len(sys.argv)!= 3):
    print("Command line argument must contain 2 values: <modelpath>.sav <probas>.csv, the filepath to the classifier model to use \
          and the csv to store the probas for all songs available.")
    exit()

model = pickle.load(open(sys.argv[1],'rb'))

csvwriter = csv.writer(open(sys.argv[2], 'w'))
scaler = pickle.load(open("scaler_model.sav",'rb'))
csvwriter.writerow(["Song Name", 'Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'])
for genre in ['Country','Blues','Classical','Electronic','Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae','Rock']:
    for song in os.listdir('wavfiles/' + genre):
        song_features = featurize("wavfiles/" + genre + '/' + song)
        
        # Normalising the dataset
        song_normal = np.array(song_features, dtype=float).reshape(1,-1)
        
        song_normal = scaler.transform(np.array(song_features, dtype=float).reshape(1,-1))
        #print(song_normal)
        print(model.classes_)
        song_proba = model.predict_proba(song_normal)
        print(model.predict(song_normal))
        #print(song)
        print(song_proba)
        output_row = []
        output_row.append(song.strip('.wav'))
        for proba in song_proba[0]:
            output_row.append(proba)
        csvwriter.writerow(output_row)
##TODO: Featurize songs##


csvwriter.close()

##TODO: Get predict proba for all songs##

##TODO: Put predict probas and names into csv for export##






