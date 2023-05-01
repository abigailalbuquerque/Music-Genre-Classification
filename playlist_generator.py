import os
import pickle
import sys
import math
import csv
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

############################################################

#Generate playlist of songs with closest proba distance to a base song

############################################################
INF = 100000

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

def proba_distance(song1_proba,song2_proba):
    distance_squared = 0
    for pair in zip(song1_proba,song2_proba):
        distance_squared += math.pow(pair[0]-pair[1],2)
    return math.sqrt(distance_squared)


if (len(sys.argv)!= 3):
    print("Command line argument must contain 2 values: <songpath>.wav <size>, the filepath to the song you would like to generate a playlist from \
          and the size of the playlist you would like to generate.")
    exit()

model = pickle.load(open("image_model.sav",'rb'))


featurized_base = featurize_image(sys.argv[1])
base_proba = model.predict(featurized_base)
base_proba = base_proba[0]
print(base_proba)
print(np.argmax(base_proba))

distances = []
master_proba_list = list(csv.reader(open("master_probas_normal.csv",'r')))
for index, proba_row in enumerate(master_proba_list[1:]):
    if(len(proba_row) == 0):
        continue
    distance = proba_distance(base_proba, np.array(proba_row[1:], dtype=float))
    distances.append((distance, index+1))


#print(distances)
playlist_size = int(sys.argv[2])
playlist = sorted(distances)[:playlist_size+1]
 

print('Generated Playlist based on ' + sys.argv[1][:-4]+ ':')
for song in playlist:
    
    print(master_proba_list[song[1]][0])
##TODO: How to display?##

##TODO: What measures to show? Average distance, empirical test of vibe?##








