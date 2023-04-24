import os
import io
from tensorflow.python.util import compat
import tensorflow as tf

GENRES = ['Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
WAV_DIR = './new_wav/'      # TODO: change based on .wavs we want

for genre in GENRES:
    for song in os.listdir(WAV_DIR + genre):
        try:
            tf.compat.v1.gfile.IsDirectory(compat.path_to_bytes(WAV_DIR+genre + '/' + song))
        except UnicodeDecodeError as err:
            print(genre + ' ' + song)
            path = WAV_DIR + genre + '/'+ song
            os.rename(path,path.replace('ü','u').replace('ó','o').replace('é','e').replace('á','a').replace('ë','e').replace('ö','o').replace('í','i').replace('Æ','ae').replace('Ü','U'))