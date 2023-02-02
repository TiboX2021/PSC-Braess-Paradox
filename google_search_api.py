"""Functions for google search

TODO: allow to do one first fetch to seek target urls, and then one final fetch to request all urls that were found this way.
Use examples from add_missing_gps.py

Le mot clé "gare" devrait suffire

TODO : create a function to do google searches asynchronously, and analyse incoming urls (use generator function until url is deemed correct)

Then, use the traditionnal fetching from wikipedia page.

"""
from typing import List, Tuple
from psc_util import fetch_urls
from bs4 import BeautifulSoup, Tag
import asyncio

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
    TODO: return results + print
    """
    async_requests = []
    loop = asyncio.get_event_loop()

    print("Creating coroutines...")

    for i, entry in enumerate(search_list):
        print(f"\rCreating coroutine {i + 1}", "in", len(search_list), end="")
        async_requests.append(loop.run_in_executor(None, searchGoogle, entry))

    print("\nLaunching coroutines...")

    return await asyncio.gather(*async_requests)
    
if __name__ == "__main__":

    # test async:
    searches = ["chien", "chat", "cochon"]

    gares = ["wikipedia gare métro la défense", "wikipedia gare metro Joinville-le-Pont", "wikipedia gare metro bibliothèque francois mitterrand"]

    asyncio.run(searchGoogleForEntries(gares))