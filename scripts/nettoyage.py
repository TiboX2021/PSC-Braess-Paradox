"""
Nettoyage du graphe en listant les sommets 'extrémités' et en les retirant, pour écourter les
parties inutiles

format des données : Network, de util.py

TODO: parcours du graphe pour supprimer les stations
Remarque: à cause des indices, on peut pas vraiment supprimer les stations de la liste de stations
autant les laisser, les remplacer ou les marquer pour indiquer qu'elles ne sont plus réellement là

Parcours du graphe: il faut charger les données de manière rapidement accessible => matrice d'adjacence?
Non, liste de stations adjacentes pour chaque station. Autant faire ça dès l'identification de network ends

TODO: il faut gérer les stations ligne 10 qui se font avoir parce que le graphe est orienté et donc elles
comptent comme 'end' alors qu'elles sont en plein milieu du réseau.
"""
from util.util import Network
from typing import Tuple, List
import numpy as np
from scripts.nettoyage import *
import json


def load_edges(network: Network) -> List[np.ndarray]:
    """Load edges in a format that will allow fast computing"""

    edges = network['edges']
    stations = len(network['stations'])
    station_connections = [[] for _ in range(stations)]

    for start, end in edges:
        station_connections[start].append(end)

    print(station_connections[0: 3])

    return [np.array(connections) for connections in station_connections]


def network_ends(network: Network) -> np.ndarray:
    """Identify the end stations of the network"""

    edges = np.array(network['edges'])  # load edges

    # for (i, j) and (j, i) edges, only count the first station.
    # This way, (i, j) and (j, i) will count as 1 i and 1 j
    first_station = edges[:, 0]

    stations, counts = np.unique(first_station, return_counts=True)

    return stations[counts < 2]  # Return stations that appear only one time


def connected_stations(network: Network) -> np.ndarray:
    """Identify stations that have connections to other lines. Even though they may be
    stations idintified as 'to be removed', the connections make them untargetable"""

    connections = np.array(network["metro_connections"] +
                           network["rer_connections"] + network["trans_connections"])

    return np.unique(connections[:, 0])


def compute_useless_vertices(network: Network) -> List[int]:
    """
    Parcours des extrémités du graphe

    quand on élimine des stations qui étaient liées à une station qu'on ne supprimera pas,
    on l'ajoute à la blacklist: elle ne comptera plus
    """

    ends = network_ends(network)
    edges = load_edges(network)

    whitelist = connected_stations(network)
    print("whitelist")
    print(whitelist)
    blacklist = []  # blacklisted stations
    deleted = []  # TODO: deleted stations

    for end in ends:

        print("################################################################")
        print("New network end")
        print("################################################################")

        # station to be blacklisted if the end of the 'lone branch' is encountered
        last_station = None
        next_stations = edges[end]
        current = end
        print("starting at :", current, network['stations'][current])

        # If there are not more than 1 non blacklisted station
        while current not in whitelist and len(non_blacklisted_next_stations := next_stations[np.isin(next_stations, blacklist + [last_station], invert=True)]) < 2:

            # TODO: delete 'current' station
            deleted.append(current)
            print("deleted", network["stations"][current])

            if len(non_blacklisted_next_stations) > 0:
                print("non blacklisted next", non_blacklisted_next_stations,
                      network["stations"][non_blacklisted_next_stations[0]])
            else:
                print("no next non blacklisted station")

            # Go to next station
            last_station = current
            current = non_blacklisted_next_stations[0]
            next_stations = edges[current]

        # End: the current station is part of a larger network, do not remove

        # Blacklist the previous station
        if last_station is not None:
            blacklist.append(last_station)

    return deleted



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
