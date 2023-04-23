import requests
from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import base64
from requests import post, get

# def getToken():
#     authString = clientID + ":" + clientSecret
#     authEnc = authString.encode("utf-8")
#     authToken = str(base64.b64encode(authEnc), "utf-8")

#     url = "https://accounts.spotify.com/api/token"
#     headers = {
#         "Authorization": "Basic " + authToken,
#         "Content-Type": "application/x-www-form-urlencoded"
#     }
#     data = {"grant_type": "client_credentials"}
#     result = post(url, headers = headers, data = data)
#     jsonResult = json.loads(result.content)
#     token = jsonResult["access_token"]
#     return token

load_dotenv()

clientID = os.getenv("SPOTIPY_CLIENT_ID")
clientSecret = os.getenv("SPOTIPY_CLIENT_SECRET")

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(clientID, clientSecret))

genre_tracks_map = {}
track_ids = open('SONG_IDS.TXT','r')
for line in track_ids:
    print(line)
    line = line.strip('\']\n')
    split_line = line.split(': [\'')
    print(split_line[0])
    print(split_line[1].split('\', \''))
    genre_tracks_map[split_line[0]] = split_line[1].split('\', \'')

for genre in ['Country', 'Blues','Classical','Electronic','Hip Hop', 'Jazz', 'Metal', 'Pop', 'Reggae','Rock']:
#for genre in ['Rock']:
    if not os.path.isdir("./" + genre):
        os.makedirs("./" + genre)
    index = 0
    urls = open(genre+"_preview.txt",'r')
    for url in urls:
        id = genre_tracks_map[genre][index]
        doc = requests.get(url.strip('\n'))
        info = spotify.track(id)
        filename = info["name"]+ " _by_ " + info["artists"][0]["name"] + ".mp3"
        filename = filename.replace(":","")
        filename = filename.replace("\"","")
        filename = filename.replace("/","")
        filename = filename.replace("?","")
        filename = filename.replace("*","")
        index += 1
        with open(genre + '/' + filename, 'wb') as f:
            f.write(doc.content)
