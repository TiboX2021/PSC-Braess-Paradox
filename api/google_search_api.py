"""Functions for google search

TODO: allow to do one first fetch to seek target urls, and then one final fetch to request all urls that were found this way.
Use examples from add_missing_gps.py

Le mot clé "gare" devrait suffire

TODO : create a function to do google searches asynchronously, and analyse incoming urls (use generator function until url is deemed correct)

Then, use the traditionnal fetching from wikipedia page.

"""
from typing import List, Tuple
from util.psc_util import fetch_urls, parse_lat_lon
import asyncio
import json

def searchGoogle(search_entry: str) -> str:
    """Synchronous function for searching a url from google
    TODO: filter url generator to get the correct wikipedia url"""
    from googlesearch import search
    url_generator = search(search_entry)

    # Loop through urls until a correct one is found
    for url in url_generator:
        if True:
            return url

async def searchGoogleForEntries(search_list: List[str]) -> List[str]:
    """
    Asynchronously execute google searches
    """
    async_requests = []
    loop = asyncio.get_event_loop()

    for i, entry in enumerate(search_list):
        print(f"\rCreating coroutine {i + 1}", "in", len(search_list), end="")
        async_requests.append(loop.run_in_executor(None, searchGoogle, entry))

    print("\nLaunching coroutines...")

    result = await asyncio.gather(*async_requests)
    print("Coroutines returned.")

    return result

def prepare_search_entries_for_wikipedia(search_entries: List[str]) -> List[str]:
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
    
if __name__ == "__main__":

    # test async:
    searches = ["chien", "chat", "cochon"]

    gares = ["wikipedia gare métro la défense", "wikipedia gare metro Joinville-le-Pont", "wikipedia gare metro bibliothèque francois mitterrand"]

    asyncio.run(get_geoloc_for_entries(gares))