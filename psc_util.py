"""
Util.py
Useful functions for data scraping / processing

JSON READING / WRITING

* read / write :
    Read and write from a json file. The write function ensures that the data is pretty printed, and that non ascii
    characters are not altered

DATA SCRAPING

* fetch_urls :
    This function takes a list of tuples (url_key, url) in order to keep track of which url is which
    get the html content of multiple pages in an asynchronous way (much fasterthan synchronous)
    this function must be awaited, ie: await fetch_urls(urls).
    When running this function with conda, ignore "coroutine 'ClientResponse.text' was never awaited"

DATA PROCESSING

* parse_lat_lon :
    get (latitude, longitude) from a wikipedia page (html text)
    Useful to gather the positions of metro stations around the world

* best_match :
    find the best match for a string in a list of strings. Useful for getting the exact name of a station as stored in
    the data.

* station_indexes :
    find all the indexes of the stations with the given name. Useful to identify indexes for edges.

"""
import asyncio
import json

import aiohttp
from typing import List, Tuple, Dict, Union
from bs4 import BeautifulSoup

import numpy as np


# JSON READING / WRITING FUNCTIONS

def read(filepath: str) -> Union[List, Dict]:
    """Read json data from a file"""
    with open(filepath, "r") as file:
        data = json.loads(file.read())
    return data


def write(filepath: str, data) -> None:
    """Write json data to a file"""
    with open(filepath, "w") as file:
        data = json.dumps(data, indent=4, ensure_ascii=False)
        file.write(data)


# DATA SCRAPING FUNCTIONS

async def get_html_text(url: Tuple[str, str], session: aiohttp.ClientSession) -> Tuple[str, str]:
    """Basic asynchronous task, performed in batches"""
    response = await session.get(url[1])
    return url[0], await response.text()


async def fetch_batch(urls: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Like fetch_urls, but not safe if too many requests (~100) are sent"""
    async with aiohttp.ClientSession() as session:
        requests = (get_html_text(url, session) for url in urls)  # Create all http text requests
        results = await asyncio.gather(*requests)  # Send them at the same time
    # noinspection PyTypeChecker
    return results


async def fetch_urls(urls: List[Tuple[str, str]], batch=50) -> List[Tuple[str, str]]:
    """Fetch multiple urls at the same time.

    The batch size avoid connection shutdown when too many
    requests are sent to a website at the same time.

    This function must be awaited, ie : await fetch_urls(urls)
    """
    responses = []
    n = int(len(urls) / batch) + 1  # Total number of batches

    for i in range(n):
        print(f"Batch {i + 1} of {n}...")
        responses.extend(await fetch_batch(urls[i * batch: (i + 1) * batch]))

    return responses


# DATA PROCESSING FUNCTIONS

def parse_lat_lon(wikipedia_html_page: str) -> Tuple[float, float]:
    """Parse latitude and longitude from a wikipedia page"""

    # Parse html data
    soup = BeautifulSoup(wikipedia_html_page, 'html.parser')

    gps_element = soup.find('a', {"class": "mw-kartographer-maplink"})

    return float(gps_element.attrs["data-lat"]), float(gps_element.attrs["data-lon"])


def match_ratio(str1: str, str2: str) -> float:
    """Returns the matching ratio of the str's alphanumeric characters"""
    n = min(len(str1), len(str2))

    max_chars = max(np.sum(np.char.isalnum(list(str1))), np.sum(np.char.isalnum(list(str2))))

    counter = 0
    for i in range(n):

        if str1[i].isalpha():

            if str1[i] == str2[i]:
                counter += 1

    return counter / max_chars


def best_match(name: str, name_list: List[str]) -> str:
    """Returns the element in the list that matches best the "name". O(n)
    Useful to get the name of a station stored in the data while not
    knowing its exact characters"""

    ratio = 0
    best = ""

    for other_name in name_list:
        new_ratio = match_ratio(name, other_name)
        if new_ratio > ratio:
            ratio = new_ratio
            best = other_name

    return best


def station_indexes(name: str, stations: List[str]) -> List[int]:
    """Get the indexes of the station with name "name", in the "stations" list.
    This method returns all the indexes found"""

    matched_name = best_match(name, stations)
    print("Match found :", matched_name, " for ", name)
    numpy_stations = np.array(stations)

    return list(np.where(numpy_stations == matched_name))[0]
