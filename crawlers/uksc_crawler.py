import os
import time
import urllib.request

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def download_press_summaries(base_url, column, folder):
    case_links = []
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, "html.parser")

    test_links = soup.find_all(class_=column)

    for link in test_links:
        case = link.find("a", class_="more")  # get relative link
        if case is not None:
            case_links.append(case["href"])

    print(
        f'total cases for {folder} : {len(case_links)}')  # the number downloads do not match number of links as
        # sometimes multiple links refer to same press summary

    for link in tqdm(case_links, total=len(case_links)):
        page = requests.get(f'https://www.supremecourt.uk/{link}')
        soup = BeautifulSoup(page.content, "html.parser")

        # Search for either "Press Summary" or "Press Summary (PDF)"
        text = soup.find("a",
                         title=lambda title: title in ["Press Summary", "Press Summary (PDF)", "Press summary (PDF)",
                                                       "Press summary (HTML version)", "Press summary HTML version",
                                                       "Press Summary HTML version", "Press summary (HTML Version)"])

        if text is not None:
            filename = text["href"].split("/")[-1]
            if filename.endswith('.pdf'):
                url = f'https://www.supremecourt.uk/cases/docs/{filename}'  # need to correct here, there are multiple types
            elif filename.endswith('.html'):
                url = f'https://www.supremecourt.uk{text["href"]}'
            filepath = f'{folder}/{filename}'
            try:
                urllib.request.urlretrieve(url, filepath)
                time.sleep(0.5)
            except:
                print(f"Exception : The Press Summary (PDF) is not available in https://www.supremecourt.uk{link}")
        else:
            print(f"The Press Summary (PDF) is not available in https://www.supremecourt.uk{link}")


def run():
    base_urls = ["https://www.supremecourt.uk/decided-cases/2022.html",
                 "https://www.supremecourt.uk/decided-cases/2021.html",
                 "https://www.supremecourt.uk/decided-cases/2020.html"]
    column_names = ['fourthColumn', 'fourthColumn', 'fourthColumn']
    folders = ['downloads/2022', 'downloads/2021', 'downloads/2020']

    for url, column, folder in zip(base_urls, column_names, folders):
        download_press_summaries(url, column, folder)


if __name__ == '__main__':
    run()
