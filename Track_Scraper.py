import json
import requests
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import base64
from requests import post, get
load_dotenv()

clientID = os.getenv("CLIENT_ID")
clientSecret = os.getenv("CLIENT_SECRET")

def getToken():
    authString = clientID + ":" + clientSecret
    authEnc = authString.encode("utf-8")
    authToken = str(base64.b64encode(authEnc), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + authToken,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers = headers, data = data)
    jsonResult = json.loads(result.content)
    token = jsonResult["access_token"]
    return token

def buildHeader(token):
    return {"Authorization": "Bearer " + token}

# def artistSearch(token, artistName):
#     url = "https://api.spotify.com/v1/search"
#     headers = buildHeader(token)
#     query = "?q="+artistName+"&type=artist&limit=1"
#     queryURL = url + query
#     result = get(queryURL, headers=headers)
#     jsonResult = json.loads(result.content)["artists"]["items"]
#     if len(jsonResult) == 0:
#         print("No existing artist found")
#         return None
#     return jsonResult[0]

# def trackScraper(token):
#     url = "https://api.spotify.com/v1/me/player"
#     headers = buildHeader(token)
#     result = get(url, headers=headers)
#     jsonResult = json.loads(result.content)
#     print(jsonResult)
#     if len(jsonResult) == 0:
#         print("No artist playing")
#         return None
#     return jsonResult

def trackScraperv2(ID, Secret, ArtistID):
    lz_uri = 'spotify:artist:' + ArtistID

    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(ID, Secret))
    results = spotify.artist_top_tracks(lz_uri)
    print("Artist Name: ", spotify.artist(lz_uri)["name"])
    with open('Rock_IDS', 'a') as g:   # EDIT HERE
        for track in results['tracks'][:10]:
            print('track    : ' + track['name'])
            if (track['preview_url'] == None):
                print("No preview, continuing...")
                print()
                continue
            else:
                print('preview url: ' + track["preview_url"])

                song = requests.get(track['preview_url'])
                with open('Rock/'+track['name']+'.mp3', 'wb') as f:    # EDIT HERE
                    f.write(song.content)
                    g.write(track['name'] + " genres: ")
                    for tag in range(len(spotify.artist(lz_uri)["genres"])):
                        g.write(spotify.artist(lz_uri)["genres"][tag] + ' ')
                    g.write('\n')
                print()
    g.close()

if __name__ == '__main__':
    myToken = getToken()
    #print("Autorization success, current token: ", myToken)
    #artistInfo = artistSearch(myToken, "Ghost")
    #print(artistInfo["genres"])
    with open('rock_id_list.txt', 'r') as file:    # EDIT HERE
        myNames = file.readlines()
    for i in range(len(myNames)):
        trackScraperv2(clientID, clientSecret, myNames[i])


