from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import time
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import os
import librosa.display
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pydub import AudioSegment
import tensorflow as tf
import math
import pickle
import sys
import csv
import multiprocessing
from playsound import playsound

load_dotenv()

clientID = os.getenv("SPOTIPY_CLIENT_ID")
clientSecret = os.getenv("SPOTIPY_CLIENT_SECRET")
GENRES = ['Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

def get_song_id(song, artist):
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(clientID, clientSecret))
    search_string = "track:" + song + " artist:" + artist
    results = spotify.search(q=search_string)
    
    print("First match on Spotify: " + results['tracks']['items'][0]['name'] + " by " + results['tracks']['items'][0]['artists'][0]['name'])
    return results['tracks']['items'][0]['id'], results['tracks']['items'][0]['name']

def scrape_preview_url(song_id):
    embed_string = 'https://open.spotify.com/embed/track/'



    # # Create a Firefox profile with Developer Tools enabled
    # profile = webdriver.FirefoxProfile()
    # profile.set_preference('devtools.toolbox.host', 'localhost')
    # profile.set_preference('devtools.toolbox.previous.host', 'localhost')
    # profile.set_preference('devtools.toolbox.selectedTool', 'netmonitor')
    options = Options()
    options.set_preference('devtools.toolbox.host', 'localhost')
    options.set_preference('devtools.toolbox.previous.host', 'localhost')
    options.set_preference('devtools.toolbox.selectedTool', 'netmonitor')

    # Start a Firefox browser with the created profile
    driver = webdriver.Firefox(options=options)

    # Loop through the list of websites and capture the destination of each HTTP GET request
    
    driver.get(embed_string + song_id)
    # Wait for the page to load
    driver.implicitly_wait(10)

    # actions = ActionChains(driver)
    # element = driver.find_element(By.ID,'root')
    # actions.click(on_element=element)
    # actions.key_down(Keys.CONTROL).key_down(Keys.SHIFT).send_keys('e').key_up(Keys.SHIFT).key_up(Keys.CONTROL).perform()

    # # Wait for the Network Monitor to load
    # monitor = driver.find_element(By.CSS_SELECTOR,'.devtools-tabpanel[aria-label="Analyze"]')
    # wait = WebDriverWait(driver, 10)
    # wait.until(EC.visibility_of(monitor))

    # # Enable the Network Monitor
    # enable_button = driver.find_element(By.CSS_SELECTOR,'.devtools-toolbar .js-command-button[data-id="network-enable-toggle"]')
    # enable_button.click()

    # # Wait for the GET requests to appear
    # wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.devtools-sidebar .request-list .request[method="GET"]')))
    time.sleep(2)
    # # Click Play
    # button = driver.find_element(By.CSS_SELECTOR, '[data-testid="play-pause-button"]')
    # print(button.text)
    # #button.click()
    # driver.execute_script("arguments[0].click();", button)
    driver.execute_script('document.querySelector(\'button[data-testid="play-pause-button"]\').click()')

    # Doesn't wait long enough for request to go through without this
    time.sleep(2)
    
    # # Get the list of GET requests
    # requests = driver.find_element(By.CSS_SELECTOR,'.devtools-sidebar .request-list .request[method="GET"]')

    # # Print the list of GET requests
    # for request in requests:
    #     url = driver.find_element(By.CSS_SELECTOR, '.request.url .har-log-line-item__content').text
    #     print(url)

    preview_url = None

    # Get the HTTP GET requests from the Developer Tools API
    requests = driver.execute_script('return performance.getEntriesByType("resource")')
    for request in requests:
        if request['initiatorType'] == 'other' and 'mp3-preview' in request['name']:
            print(f"Destination of HTTP GET request: {request['name']}")
            preview_url = request['name']

    # Clear the Developer Tools logs
    
    driver.execute_script('return window.performance.clearResourceTimings();')
        
    # Close the Firefox browser
    driver.quit()
    del driver
    return preview_url

def get_mp3(preview_url, song_name):
    doc = requests.get(preview_url.strip('\n'))
    mp3path = 'temp' + '/' + song_name + '.mp3'
    with open(mp3path, 'wb') as f:
        f.write(doc.content)
    return mp3path

def convert_mp3_to_wav(mp3path):
    sound = AudioSegment.from_mp3(mp3path)
    wavpath = mp3path.replace('.mp3','.wav')
    sound.export(wavpath, format="wav")
    return wavpath

def generate_spectrogram(wavpath):
    samples, sample_rate = librosa.load(wavpath, sr=None)

    sgram = librosa.stft(samples)

    # use the mel-scale instead of raw frequency
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)

    # use the decibel scale to get the final Mel Spectrogram
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')

    # for saving image
    plt.axis('off')
    spectpath = wavpath.replace('.wav', '.png')
    plt.savefig(spectpath, bbox_inches='tight', pad_inches=0)
    #p = multiprocessing.Process(target=plt.show(), args=())
    
    return spectpath

def featurize_image(spectpath):

    img = tf.keras.utils.load_img(spectpath)   
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    return img_array

def proba_distance(song1_proba,song2_proba):
    distance_squared = 0
    for pair in zip(song1_proba,song2_proba):
        distance_squared += math.pow(pair[0]-pair[1],2)
    return math.sqrt(distance_squared)

def generate_playlist(spectpath, song, master_proba_list):
    model = pickle.load(open("image_model.sav",'rb'))


    featurized_base = featurize_image(spectpath)
    base_proba = model.predict(featurized_base)
    base_proba = base_proba[0]
    #print(base_proba)
    print("Based on the preview, we think the song is " + GENRES[np.argmax(base_proba)])
    print()
    distances = []
    
    for index, proba_row in enumerate(master_proba_list[1:]):
        if(len(proba_row) == 0):
            continue
        distance = proba_distance(base_proba, np.array(proba_row[1:], dtype=float))
        distances.append((distance, index+1))


    #print(distances)
    playlist_size = int(3)
    playlist = sorted(distances)[:playlist_size]
    

    print('Generated Playlist based on ' + song + ':')
    for song in playlist:
        
        print(master_proba_list[song[1]][0])

    return playlist

def get_song_path(name):
    for genre in GENRES:
        for song in os.listdir('./new_wav/' + genre):
            if name in song:
                return "./new_wav/" + genre + '/' + song

if __name__ == "__main__":
    
    #id = get_song_id("Here Comes The Sun", "The Beatles")
    start = time.time()
    id, name = get_song_id(sys.argv[1], sys.argv[2])
    end = time.time()
    #id = "3CPFHaVEHkkH26hAEPgMMp"
    print("Retrieving Song ID took " + str((end-start)) + " seconds", flush=True)
    print("\n--------------------------------------------------------------------------\n")

    start = time.time()
    preview_url = scrape_preview_url(id)
    end = time.time()
    print("Scraping the preview url took " + str((end-start)) + " seconds", flush=True)
    print("\n--------------------------------------------------------------------------\n")

    start = time.time()
    mp3path = get_mp3(preview_url, name)
    end = time.time()
    print("Downloading the MP3 took " + str((end-start)) + " seconds", flush=True)
    print("\n--------------------------------------------------------------------------\n")

    start = time.time()
    wavpath = convert_mp3_to_wav(mp3path)
    end = time.time()
    print("Converting to a .Wav file took " + str((end-start)) + " seconds", flush=True)
    print("\n--------------------------------------------------------------------------\n")

    # input("Press Enter to play " + name)
    # p = multiprocessing.Process(target=playsound, args=(wavpath,))
    # p.start()
    # input("")
    # p.terminate()

    start = time.time()

    spectpath = generate_spectrogram(wavpath)
    end = time.time()
    print("Generating the spectrogram image took " + str((end-start)) + " seconds", flush=True)
    print("\n--------------------------------------------------------------------------\n")

    start = time.time()
    master_proba_list = list(csv.reader(open("master_probas.csv",'r')))
    playlist = generate_playlist(spectpath, name, master_proba_list)
    end = time.time()
    print("Running the model and generating the playlist took " + str((end-start)) + " seconds", flush=True)
    print("\n--------------------------------------------------------------------------\n")

    # for song in playlist:
    #     path = get_song_path(master_proba_list[song[1]][0])
    #     input("Press Enter to play " + master_proba_list[song[1]][0])
    #     p = multiprocessing.Process(target=playsound, args=(path,))
    #     p.start()
    #     input("")
    #     p.terminate()

    os.remove(mp3path)
    #os.remove(wavpath)
    os.remove(spectpath)

