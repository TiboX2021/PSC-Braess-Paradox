"""
Test de l'affichage du flux sur une carte avec les données gps et les données de flux sur le premier chemin

L'idée pour l'instant est de montrer ça avec des lignes de couleur différente selon la fréquentation.
On pourrait différencier le rer du métro avec des styles de traits différents

TODO: mettre ça dans un fichier util pour l'affichage des données

en gros : le flux est pour les stations dans la liste. Il n'y a quá les fetch

Faire une classe qui load les données pour éviter d'avoir à les refetch à chaque fois (pas besoin d'un système de cache)
On n'oublie pas les type hints
"""
import json
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from typing import Dict, Tuple
from util.util import Network

class Display:

    def __init__(self, gps_datafile: str, network_datafile: str) -> None:
        
        f = open(gps_datafile, encoding="utf-8")
        self.gps_data: Dict[str, Tuple[float, float]] = json.loads(f.read())
        f.close()

        f = open(network_datafile, encoding="utf-8")
        self.network_data: Network = json.loads(f.read())
        f.close()

    def show_flow(self, flow: np.ndarray, cmap = "Wistia"):
        """Display a flow with geographical data"""

        # Show only edges, not the multiple connections inside stations
        end_index = len(self.network_data["edges"])

        # Prepare heatmap: get maximum flow in order to normalize the maximum to 1
        max_flow = np.max(flow[:end_index])
        cmap = get_cmap(cmap)

        for index, flow_value in enumerate(flow[:end_index]):
            
            color = cmap(flow_value / max_flow * 2) # Get color from flow value

            # TODO : preprocess data for faster geoloc fetching
            start, end = self.network_data["edges"][index]
            start_name, end_name = self.network_data["stations"][start], self.network_data["stations"][end]
            start_lat, start_lon = self.gps_data[start_name]
            end_lat, end_lon = self.gps_data[end_name]

            plt.plot((start_lon, end_lon), (start_lat, end_lat), color=color)
            # print('\rplot', index + 1, "of", end_index, "with color", color, end="")
            # print(flow_value)
    
        plt.show()


if __name__ == "__main__":

    # Load flows
    flows = np.genfromtxt("out1.csv", delimiter=',')

    last_flow = flows[-1]

    display = Display(gps_datafile="paris_gps.json", network_datafile="paris_network.json")

    display.show_flow(last_flow)



