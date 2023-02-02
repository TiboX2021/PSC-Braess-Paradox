"""Functions for asynchronous google search
"""
from typing import List
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

    
if __name__ == "__main__":

    # test :
    searches = ["chien", "chat", "cochon"]
    asyncio.run(searchGoogleForEntries(searches))