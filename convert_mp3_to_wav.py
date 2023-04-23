import os
from pydub import AudioSegment
import multiprocessing as mp

GENRES = ['Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
MP3_DIR = "./new_mp3/"
WAV_DIR = "./new_wav/"

def convert_mp3_of_genre_to_wav(genre: str):
	for mp3_file in os.listdir(MP3_DIR + genre):

		song_name = mp3_file[:-4]
		
		input_file = MP3_DIR + genre + "/" + mp3_file
		output_file = WAV_DIR + genre + "/" + song_name + ".wav"

		sound = AudioSegment.from_mp3(input_file)
		sound.export(output_file, format="wav")

		print("converted to .wav  " + genre + ' / ' + song_name, flush=True)


if __name__ == '__main__':
	for genre in GENRES:
		p = mp.Process(target=convert_mp3_of_genre_to_wav, args=(genre,))
		p.start()

