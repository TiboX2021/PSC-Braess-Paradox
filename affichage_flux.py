"""
Affichage du flux sur une carte avec les données gps et les données de flux sur un chemin donné

POUR L'UTILISER, IL SUFFIT DE CHANGER LA VARIABLE "FICHIER_DU_FLUX_A_AFFICHER"

La taille des marqueurs peut être changée avec "TAILLE_DES_MARQUEURS". Ils ne s'agrandissent pas
quand on zoom, c'est pour ça qu'ils ont l'air trop grands au début ! Il faut zoomer sur la partie intéressante du graphe
"""
import json
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from util.util import Network, read_json
from util.util import read_argv

################################################################################
#                          CHANGER CETTE VARIABLE                              #
################################################################################
TAILLE_DES_MARQUEURS = 1.  # 0.5 ou 2, par exemple.


class Display:

    def __init__(self, gps_datafile: str, network_datafile: str) -> None:

        f = open(gps_datafile, encoding="utf-8")
        self.gps_data: Dict[str, Tuple[float, float]] = json.loads(f.read())
        f.close()

        f = open(network_datafile, encoding="utf-8")
        self.network_data: Network = json.loads(f.read())
        f.close()

        n_stations = len(self.network_data["stations"])

        # station to station matrix in order to combine, for each pair, the incoming and departing flows
        self.flow_matrix = np.zeros((n_stations, n_stations))

        # Access to line color for each station in constant time
        self.colors = []
        self.colors.extend(["#FECE00"] * 25)
        self.colors.extend(["#0065AE"] * (49 - 24))
        self.colors.extend(["#9F971A"] * (74 - 49))
        self.colors.extend(["#99D4DE"] * (78 - 74))
        self.colors.extend(["#BE418D"] * (105 - 78))
        self.colors.extend(["#F19043"] * (127 - 105))  # Ligne 5flows
        self.colors.extend(["#84C28E"] * (155 - 127))  # Ligne 6
        self.colors.extend(["#F2A4B7"] * (193 - 155))  # ) Ligne 7
        self.colors.extend(["#84C28E"] * (201 - 193))  # Ligne 7 bis
        self.colors.extend(["#CDACCF"] * (238 - 201))  # Ligne 8
        self.colors.extend(["#D5C900"] * (275 - 238))  # Ligne 9
        self.colors.extend(["#8C5E24"] * (288 - 275))  # Ligne 11
        self.colors.extend(["#007E49"] * (316 - 288))  # Ligne 12
        self.colors.extend(["#622280"] * (329 - 316))  # Ligne 14
        self.colors.extend(["#E4B327"] * (352 - 329))  # Ligne 10
        self.colors.extend(["#99D4DE"] * (384 - 352))  # Ligne 13
        self.colors.extend(["#FF1400"] * (430 - 384))  # RER A
        self.colors.extend(["#3C91DC"] * (477 - 430))  # RER B
        self.colors.extend(["#FFBE00"] * (561 - 477))  # RER C
        self.colors.extend(["#00643C"] * (624 - 561))  # RER D
        self.colors.extend(["#A0006E"] * (646 - 624))  # RER E

        # print(n_stations)
        # print(len(self.colors))

    def compute_flow_matrix(self, flow: np.ndarray):
        """Combine the flow of different edges in the flow matrix"""

        for value, (start, end) in zip(flow, self.network_data["edges"]):
            self.flow_matrix[start, end] = value

        self.flow_matrix += np.transpose(self.flow_matrix)

    def show_flow(self, flow: np.ndarray, endpoints: List[Tuple[int, int]] = ((0, 100),)):
        """Display a flow with geographical data"""

        self.compute_flow_matrix(flow)

        max_flow = np.max(self.flow_matrix)
        for (start, end) in self.network_data["edges"]:

            if self.flow_matrix[start, end] != -1:

                relative_flow = self.flow_matrix[start, end] / max_flow
                width = TAILLE_DES_MARQUEURS * relative_flow * 15 + 1

                # Edge width en fonction du flot relatif

                start_name, end_name = self.network_data["stations"][start], self.network_data["stations"][end]
                start_lat, start_lon = self.gps_data[start_name]
                end_lat, end_lon = self.gps_data[end_name]

                color = self.colors[max(start, end)]

                # Pointillé si RER:
                linestyle = '-'
                if max(start, end) > 384:
                    linestyle = '-.'

                plt.plot((start_lon, end_lon), (start_lat, end_lat), linestyle, color=color, linewidth=str(width),
                         solid_capstyle='round')

                # Do not plot this edge again
                self.flow_matrix[start, end] = -1
                self.flow_matrix[end, start] = -1

        # DEBUG : affichage du point de départ et du point d'arrivée pour bien visualiser
        # La station 0 et la 100 par défaut
        for start, end in endpoints:
            start = self.network_data["stations"][start]
            y1, x1 = self.gps_data[start]
            end = self.network_data["stations"][end]
            y2, x2 = self.gps_data[end]
            marker_size = 20 * TAILLE_DES_MARQUEURS
            plt.plot(x1, y1, 'bo', markersize=marker_size)
            plt.plot(x2, y2, 'bo', markersize=marker_size)

        plt.show()

    def show_first_paths(self, filename: str):
        """Show the first paths in different colors (for each path)"""

        flows = read_json(filename)

        colors = ['r', 'g', 'b', 'k', 'm', 'y']

        for index, flow in enumerate(flows):

            for value, (start, end) in zip(flow, self.network_data["edges"]):

                # Enumerate all edge values. Only display those that have non zero value

                start_name, end_name = self.network_data["stations"][start], self.network_data["stations"][end]
                start_lat, start_lon = self.gps_data[start_name]
                end_lat, end_lon = self.gps_data[end_name]

                # Pointillé si RER:
                linestyle = '-'
                if max(start, end) > 384:
                    linestyle = '-.'

                if value != 0:
                    plt.plot((start_lon, end_lon), (start_lat, end_lat), linestyle + colors[index], linewidth=str(10),
                             solid_capstyle='round')
                else:
                    plt.plot((start_lon, end_lon), (start_lat, end_lat), linestyle + 'k', linewidth=str(1),
                             solid_capstyle='round')

        plt.show()


if __name__ == "__main__":

    # Load last flow (extracted via script because opening the total flow file is way too long)
    display = Display(gps_datafile="paris_gps.json", network_datafile="paris_network.json")

    mode, filename = read_argv(2)

    last_flow = np.array(read_json(filename))
    mode = int(mode)

    if mode == 1:
        # display.show_flow(last_flow, endpoints=[(0, 100), (68, 210)])
        display.show_flow(last_flow, endpoints=[(331, 280), (82, 266), (312, 294), (109, 249)])
    elif mode == 2:
        display.show_first_paths("first-5-paths.json")
