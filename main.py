"""
Test des fonctions
"""
from util import Network
from nettoyage import *
import json


def load_dataset(filename: str) -> Network:

    f = open(filename, 'r')
    data = f.read()
    f.close()

    return json.loads(data)


if __name__ == "__main__":

    filename = "paris_network.json"

    data = load_dataset(filename)

    deleted = compute_useless_vertices(data)

    print("deleted stations : \n")
    print(deleted)
