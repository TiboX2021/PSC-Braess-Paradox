"""Functions for google search

TODO: allow to do one first fetch to seek target urls, and then one final fetch to request all urls that were found this way.
Use examples from add_missing_gps.py

Le mot clé "gare" devrait suffire

TODO : faire un repo gitlab et leur partager le code. Je leur donnerai les accès

"""
from typing import List
from psc_util import fetch_urls


def createGoogleSearchUrl(search: str, client : str = 'firefox') -> str:
    """Search google for result"""
    return f"https://www.google.com/search?client={client}-b-d&q={search.replace(' ', '+')}"

async def searchGoogleForEntries(searchList: List[str]) -> List[str]:
    """Search google for multiple search entries and return the first url for each entry"""

    # Create urls and index them by their search content
    indexed_urls = [(search, createGoogleSearchUrl(search)) for search in searchList]

    indexed_responses = await fetch_urls(indexed_urls)

    # TODO : analyze response and parse first url