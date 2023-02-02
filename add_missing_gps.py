"""
Fetch missing gps entries from wikipedia
"""
import json
import numpy as np
from util.psc_util import *
from bs4 import BeautifulSoup, Tag
from typing import Tuple, List, Callable
from api.google_search_api import searchGoogleForEntries


index = 5 # pour le fichier

def prepare_search_entries_for_wikipedia(search_entries: List[str]) -> List[str]:
    """Add keywords to station names in order for the google search to return relevant urls"""
    return [f"wikipedia gare métro {search_entry}" for search_entry in search_entries]

async def get_geoloc_for_entries(search_list: List[str]) -> List[Tuple[str, Tuple[float, float]]]:
    """Get geolocalization data for each search entry"""
    # Fetch wikipedia page urls for each search entry
    urls = await searchGoogleForEntries(prepare_search_entries_for_wikipedia(search_list))

    # Fetch all wikipedia pages
    print("Fetching wikipedia pages...")
    resp = await fetch_urls(list(zip(search_list, urls)))

    # Analyze geolocalization data and return it
    return list(zip(search_list, [parse_lat_lon(html) for _, html in resp]))


def find_missing_entries(graph: str, gps: str):
    """Find missing gps entries in comparison to the graph"""
    # todo: file names, etc
    f = open("paris_network.json", encoding="utf-8")
    data = json.loads(f.read())
    f.close()

    entries = data["stations"]

    f = open(f"paris_gps{index}.json")
    gps = json.loads(f.read())
    f.close()

    missing = []

    for i, entry in enumerate(entries):

        try:
            gps[entry]
        except:
            missing.append(entry)
            print("Missing entry no", i, entry)

    missing = np.unique(missing)
    print(len(missing), "missing entries")

    # Recherche des trucs qui restent
    search_entries = [f"métro {entry}" for entry in missing]

# CODE PROPRE ICI ##################################################################################################


def createWikipediaSearchUrl(search: str) -> str:
    """Wikipedia search url"""
    return f"https://fr.wikipedia.org/w/index.php?search={search.replace(' ', '+')}&title=Spécial:Recherche&ns0=1"

def createGoogleSearchUrl(search: str, client : str = 'firefox') -> str:
    """Search google for result"""
    return f"https://www.google.com/search?client={client}-b-d&q={search.replace(' ', '+')}"


def parseSearchPageEntries(html: str, origin_search: str, origin_url: str) -> List[Tuple[str, str]]:
    """Parse title and full url for each result entry on the Wikipedia search page"""
    soup = BeautifulSoup(html, 'html.parser')

    # Get search result ul container
    search_results = soup.find("ul", {"class": "mw-search-results"})

    if search_results is None:
        # Wikipedia already redirected the url to the correct page : no need to process search results
        return [[origin_search, origin_url]]

        # Find all individual li elements
    result_list = search_results.find_all(
        "li", {"class": "mw-search-result mw-search-result-ns-0"})

    def parse_title_and_url(entry: Tag) -> Tuple[str, str]:
        heading = entry.find("div", {"class": "mw-search-result-heading"})
        a = heading.find("a")
        return (a["title"], base_wikipedia_url + a["href"])

    return [parse_title_and_url(result) for result in result_list]


async def searchWikipediaForEntries(search_list: List[str], best_match_function: Callable[[List[str]], int] = lambda x: 0):
    """
    Search for a list of entries in wikipedia.
    Batches of entries are requested asynchronously in order to improve performance

    * search_list : list of independent searches to be requested from wikipedia
    * best_match_function : callback that returns the index of the best matching title in the entry titles it receives as argument
    default : returns the first matching entry.

    TODO: on fait quoi une fois qu'on a tout fetch? Il faut return les urls, en gros? Ou les couples titre + url?
    TODO: if massive amounts of urls have to be fetched, use multiprocessing in order to fetch asynchronously + use parallel
    computing to parse the results
    """
    # Create urls and index them by their search input
    urls = [(search, createWikipediaSearchUrl(search)) for search in search_list]

    # Fetch search entries batch by batch, asynchronously
    fetched = await fetch_urls(urls)

    def extract_best_match(search_results: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Extracts the title + url of the best match in a list of search result (title, url) tuples,
        according to the best_match_function"""
        index = best_match_function([title for title, _ in search_results]
                                    )  # TODO: this could be improved if fetch_urls did not return a str index tupled with each url
        return search_results[index]

    # Parse the result entries and select the best matching one for each request
    best_entries = [extract_best_match(
        parseSearchPageEntries(html, origin_url, origin_url)) for (search, html), (origin_search, origin_url) in zip(fetched, urls)]

    # pour le debug : print les noms et urls
    for title, url in best_entries:
        print(title, url)

    # TODO: continue

# CODE SALE ICI ##############################################################################


async def main():
    f = open("paris_network.json", encoding="utf-8")
    data = json.loads(f.read())
    f.close()

    entries = data["stations"]

    f = open(f"paris_gps{index}.json")
    gps = json.loads(f.read())
    f.close()

    missing = []

    for i, entry in enumerate(entries):

        try:
            gps[entry]
        except:
            missing.append(entry)
            print("Missing entry no", i, entry)

    missing = np.unique(missing)
    print(len(missing), "missing entries")

    # Recherche des trucs qui restent
    search_entries = [f"métro {entry}" for entry in missing]

    await searchWikipediaForEntries(search_entries)

    # # explore urls
    # fetched = await fetch_urls(urls)

    # for name, result in fetched:

    #     try:

    #         lat, lon = parse_lat_lon(result)

    #         gps[name] = [lat, lon]
    #     except Exception as e:
    #         print("Error fetching url", name)

    # data = json.dumps(gps, indent=4, ensure_ascii=False)
    # f = open(f"paris_gps{index + 1}.json", "w")
    # f.write(data)
    # f.close()


if __name__ == "__main__":

    asyncio.run(main())
