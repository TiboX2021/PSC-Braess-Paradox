"""Functions for google search

TODO: allow to do one first fetch to seek target urls, and then one final fetch to request all urls that were found this way.
Use examples from add_missing_gps.py

Le mot clé "gare" devrait suffire

TODO : faire un repo gitlab et leur partager le code. Je leur donnerai les accès

"""
from typing import List, Tuple
from psc_util import fetch_urls
from bs4 import BeautifulSoup, Tag
import asyncio


def createGoogleSearchUrl(search: str, client : str = 'firefox') -> str:
    """Search google for result"""
    return f"https://www.google.com/search?q={search.replace(' ', '+')}"

def parseGoogleSearchResults(html: str) -> Tuple[str, str]:
    """Parse the Google search result page for the first url"""

    # Create html parser
    soup = BeautifulSoup(html, 'html.parser')
    print(soup)

    # Get the search results from the page
    search = soup.find("div", {"id": "search"})
    print(search)
    truc = search.find("div", {"id": "rso"})
    print(truc)
    results_list = truc.findChildren("div", recursive=False)
    print(results_list)
    # Get the first search result
    first_result: Tag = results_list[0]

    # Get url and title for the selected result
    link_element = first_result.find("a")

    link_url = link_element["href"]
    title = link_element.find("h3").getText()

    return title, link_url


async def searchGoogleForEntries(search_list: List[str]) -> List[str]:
    """Search google for multiple search entries and return the first url for each entry"""

    # Create urls and index them by their search content
    indexed_urls = [(search, createGoogleSearchUrl(search)) for search in search_list]

    indexed_responses = await fetch_urls(indexed_urls)

    indexed_parsed_results = [parseGoogleSearchResults(html_response) for _, html_response in indexed_responses]

    # DEBUG : print out the results
    for title, url in indexed_parsed_results:
        print(title, url)

if __name__ == "__main__":

    # search_list = ["wikipédia métro gare Joinville-le-Pont"]

    # asyncio.run(searchGoogleForEntries(search_list))

    # Using google's api instead
    from googlesearch import search
    import itertools

    # result = search("wikipédia métro gare Bibliothèque François-Mitterrand") # Attention: ça ne renvoie pas forcément le bon url en premier!
    # # Il y a parfois ['https://upload.wikimedia.org en premier, à éliminer de la recherche
    # print(list(itertools.islice(result, 3)))

    # Utiliser aiogoogle pour le faire de manière asynchrone?
    # Sinon essayer de wrap la recherche google dans des trucs async

    def n_first_results(generator, n=3):
        return list(itertools.islice(generator, n))

    # test async:
    searches = ["chien", "chat", "cochon"]

    def one_google_search(search_content):
        print("searching for", search_content, "!")
        result = search(search_content)
        print(n_first_results(result))


    async def execute_one_google_search_asynchronously(search_content):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, one_google_search, search_content)  # utiliser ça pour faire des requêtes asynchrones!


    async def do_all_searches():
        await asyncio.gather(*[execute_one_google_search_asynchronously(a) for a in searches])

    # Lancement des requêtes : regarder si c'est bien asynchrone
    asyncio.run(do_all_searches())
