import sys
sys.path.append('../')

from selenium import webdriver
import os
from urllib.parse import urlparse

DRIVER = 'chromedriver'

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(DRIVER, chrome_options=chrome_options)
driver.set_window_size(1920,1080)
for i in range(1, len(sys.argv)):
    driver.get(sys.argv[i])
    pre = '../datasets/test_sites/'
    os.mkdir(pre + urlparse(sys.argv[i]).netloc)
    pref = pre + urlparse(sys.argv[i]).netloc
    screenshot = driver.save_screenshot(pref + '/shot.png')
    info = open(pref + "/info.txt", "w")
    info.write("https://www.phish.com")
    #info.write(sys.argv[i])
    info.close()
    info = open(pref + "/lure.txt", "w")
    info.write(sys.argv[i])
    info.close()
    html = open(pref + "/html.txt", "w")
    html.write(driver.page_source)
    html.close()
driver.quit()
    