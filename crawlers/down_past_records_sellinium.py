import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from webdriver_manager.firefox import GeckoDriverManager

total_pages = 55

# Set up Selenium WebDriver
service = Service(GeckoDriverManager().install())
driver = webdriver.Firefox(service=service)  # Launch Firefox browser
for p in tqdm(range(total_pages)):

    url = f"https://www.supremecourt.uk/cases?cs=Judgment+given&jded=2019-12-31&jdsd=2009-01-01&sort=judgmentDateOldest&p={p}"

    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    judgment_links = soup.findAll("a", attrs={
        "class": "whitespace-pre-line text-secondary-400 dark:text-secondary-100 font-bold"})

    with open("data/past_data/links.txt", "a") as f:
        for link in judgment_links:
            f.write(link["href"] + "\n")

    driver.quit()
