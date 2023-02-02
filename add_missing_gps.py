"""
Fetch missing gps entries from wikipedia
"""
import json
import numpy as np
from util.psc_util import *
from bs4 import BeautifulSoup, Tag
from typing import Tuple, List, Callable
from api.google_search_api import searchGoogleForEntries


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


def find_missing_entries(graph: str = "paris_network.json", gps: str = "paris_gps2.json") -> List[str]:
    """Find missing gps entries in comparison to the graph and return the names"""
    f = open(graph, encoding="utf-8")
    data = json.loads(f.read())
    f.close()

    entries = data["stations"]

    f = open(gps)
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

    return missing


if __name__ == "__main__":

    graph = "paris_network.json"
    gps = "paris_gps2.json"

    missing_entries = find_missing_entries(graph, gps)

    geoloc = asyncio.run(get_geoloc_for_entries(missing_entries))

    # Stockage des géoloc en créant l'objet dans le bon format
    out = {}

    for name, coordinates in geoloc:
        out[name] = coordinates

    data = json.dumps(out, indent=4, ensure_ascii=False)

    f = open("out.json", "w", encoding="utf-8")
    f.write(data)
    f.close()
