import os
import pickle
import sys
import math
import csv

############################################################

#Generate playlist of songs with closest proba distance to a base song

############################################################


def proba_distance(song1_proba,song2_proba):
    distance_squared = 0
    for pair in zip(song1_proba,song2_proba):
        distance_squared += math.pow(pair[0]-pair[1],2)
    return math.sqrt(distance_squared)


if (len(sys.argv)!= 2):
    print("Command line argument must contain 2 values: <songpath>.wav <size>, the filepath to the song you would like to generate a playlist from \
          and the size of the playlist you would like to generate.")
    exit()

model = pickle.load(open("genre_classifier_model.sav",'rb'))

##TODO: Featurize song##
featurized_base = None
base_proba = model.predict_proba(featurized_base)

##TODO: Get distance from base proba to every other proba in master proba csv##
distances = []
master_proba_csv = csv.reader(open("master_proba.csv",'r'),delimeter=',')
for index, proba_row in enumerate(master_proba_csv):
    distance = proba_distance(base_proba, proba_row[1:])
    distances.append(distance, index)


##TODO: Find the n minimum distance songs from the base song##
playlist_size = int(sys.argv[2])
playlist = sorted(distances)[:playlist_size]

##TODO: How to display?##

##TODO: What measures to show? Average distance, empirical test of vibe?##








