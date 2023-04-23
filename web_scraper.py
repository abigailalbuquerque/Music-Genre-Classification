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
websites = ['https://open.spotify.com/embed/track/7wdwIaXUuzlu1grzWMFRJm', 'https://open.spotify.com/embed/track/54bm2e3tk8cliUz3VSdCPZ']

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
for website in websites:
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
    time.sleep(2)
    
    # # Get the list of GET requests
    # requests = driver.find_element(By.CSS_SELECTOR,'.devtools-sidebar .request-list .request[method="GET"]')

    # # Print the list of GET requests
    # for request in requests:
    #     url = driver.find_element(By.CSS_SELECTOR, '.request.url .har-log-line-item__content').text
    #     print(url)

    # Get the HTTP GET requests from the Developer Tools API
    requests = driver.execute_script('return performance.getEntriesByType("resource")')
    for request in requests:
        if request['initiatorType'] == 'other':
            print(f"Destination of HTTP GET request: {request['name']}")

    # Clear the Developer Tools logs
    
    driver.execute_script('return window.performance.clearResourceTimings();')
    
# Close the Firefox browser
driver.quit()
del driver