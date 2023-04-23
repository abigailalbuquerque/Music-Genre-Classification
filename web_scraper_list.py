from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


import time

import json

# List of websites to visit
genre_tracks_map = {}
embed_string = 'https://open.spotify.com/embed/track/'
websites = ['https://open.spotify.com/embed/track/7wdwIaXUuzlu1grzWMFRJm', 'https://open.spotify.com/embed/track/54bm2e3tk8cliUz3VSdCPZ']
track_ids = open('SONG_IDS.TXT','r')
for line in track_ids:
    print(line)
    line = line.strip('\']\n')
    split_line = line.split(': [\'')
    print(split_line[0])
    print(split_line[1].split('\', \''))
    genre_tracks_map[split_line[0]] = split_line[1].split('\', \'')

print(genre_tracks_map['Rock'])

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
for genre in ['Country', 'Blues','Classical','Electronic','Hip Hop', 'Jazz', 'Metal', 'Pop', 'Reggae','Rock']:
#for genre in ['Metal', 'Pop', 'Reggae','Rock']:
    preview_file = open(genre+"_preview.txt",'w')
    preview_urls = []
    for track_id in genre_tracks_map[genre]:
        website = embed_string + track_id
        
        driver.get(website)
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
        time.sleep(1)
    
    # # Get the list of GET requests
    # requests = driver.find_element(By.CSS_SELECTOR,'.devtools-sidebar .request-list .request[method="GET"]')

    # # Print the list of GET requests
    # for request in requests:
    #     url = driver.find_element(By.CSS_SELECTOR, '.request.url .har-log-line-item__content').text
    #     print(url)

    # Get the HTTP GET requests from the Developer Tools API
        requests = driver.execute_script('return performance.getEntriesByType("resource")')
        for request in requests:
            if request['initiatorType'] == 'other' and 'mp3-preview' in request['name']:
                print(f"Destination of HTTP GET request: {request['name']}")
                preview_urls.append(request['name'])
                preview_file.write(request['name'] + '\n')
        
    # Clear the Developer Tools logs

        driver.execute_script('return window.performance.clearResourceTimings();')
    #for url in preview_urls:
        
    preview_file.close()
    
# Close the Firefox browser
driver.quit()
del driver