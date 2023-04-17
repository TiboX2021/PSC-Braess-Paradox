"""
En réalité il y a des oscillations, donc cumsum des erreurs c'est pas pertinent
=> il faudrait les erreurs par rapport à la dernière valeur

consecutive costs & consecutive errors are the same (no advantage using one or another)

But: does cost show a quicker convergence than absolute error?
De la même manière, si on les map ensemble, il y a une corrélation directe entre erreur totale et coût total.

Pour donner des nombres : passage sous 

c'est pas 5%, mais 500%, en réalité?? c'est beaucoup trop
500% d'erreur : 5% - 60ème itération
100% d'erreur : 1% - 280ème itération
20 % d'erreur : 0.2% - 1130ème itération
10 % d'erreur : 0.1% - 1860ème itération
1  % d'erreur : 0.001 - 4550 itération (1% d'erreur)

200 itérations par minute

c'est pour un chemin pris entre la station 0 et 100, soit entre
- la défense (ligne 1)
- raspail

c"est un chemin pas trop absurde, mais il me manque encore l'affichage des flux via matplotlib, fonctionà faire...
"""

import json
from typing import Dict, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from compute_flow_cost import compute_flow_cost
from util.util import Network


def error_percentage(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b)) / np.sum(np.abs(b)) * 100


def debug():
    flows = np.genfromtxt("out1.csv", delimiter=',')

    # Affichage des erreurs par rapport au dernier en fonction du temps
    last_flow = flows[-1]

    errors = [error_percentage(last_flow, flow) for flow in flows]
    consecutive_errors = [error_percentage(
        flows[i + 1], flows[i]) for i in range(len(flows) - 1)]
    consecutive_errors.append(0)

    costs = np.sum(flows, axis=1)

    x = range(len(errors))

    costs = costs / np.max(costs) * np.max(errors)

    costs -= np.min(costs)

    consecutive_costs = [error_percentage(
        costs[i + 1], costs[i]) for i in range(len(costs) - 1)]
    consecutive_costs.append(consecutive_costs[-1])

    # plt.plot(x, costs, label="costs")
    plt.plot(x, consecutive_errors, label="consecutive errors")
    plt.plot(x, errors, label="last error")
    # plt.plot(x, consecutive_costs, label="consecutive costs")

    plt.grid()

    plt.legend()

    plt.show()


def convergence_couts():
    """Etude de la convergence avec les fonctions de coût"""
    f = open("paris_network.json")
    data = json.loads(f.read())

    f.close()
    print("Voyage entre les stations suivantes :")
    print(data["stations"][0])
    print(data["stations"][100])

    # Quelle fonction de coût pour les arêtes? Il faut que ça soit croissant: en x. Les coûts sont croissants
    # Les coûts qui ont été utilisés pour faire tourner l'algo sont les x


def metadata():
    f = open("paris_network.json")
    data: Network = json.loads(f.read())
    f.close()

    print("sommets :", len(data["stations"]))
    print("arêtes :", len(data["edges"]) + len(data["metro_connections"]
                                               ) + len(data["rer_connections"]) + len(data["trans_connections"]))


def print_relative_flow_costs():
    """Print successive flow cost errors relative to the last one"""
    flows = np.genfromtxt("out.csv", delimiter=',')

    # Affichage des erreurs par rapport au dernier en fonction du temps
    last_flow = flows[-1]
    last_flow_cost = compute_flow_cost(last_flow)

    errors = np.array([error_percentage(last_flow_cost, compute_flow_cost(flow)) for flow in flows])

    # affichage
    plt.plot(range(len(errors)), errors)
    plt.title("Erreurs relatives à la dernière itération")
    plt.xlabel("Itération i")
    plt.ylabel("Erreur relative (%)")
    plt.ylim(-10, 110)
    plt.grid()
    plt.legend()
    plt.show()


def affichage_dernier_flot():
    flows = np.genfromtxt("out1.csv", delimiter=',')
    last_flow = flows[-1]  # Dernier flot, à afficher

    # Métadonnées :
    f = open("paris_network.json", encoding="utf-8")
    data: Network = json.loads(f.read())
    f.close()

    # Données gps
    f = open("paris_gps.json")
    gps: Dict[str, Tuple[float, float]] = json.loads(f.read())
    f.close()

    # colormap choisie
    cmap = cm.get_cmap("gist_rainbow")

    # pour normaliser les flots
    max_flow = np.max(last_flow[:384])

    # on n'affiche que les trajets, pas les correspondances.
    for i in range(len(data["edges"])):

        start, end = data["edges"][i]

        flow_value = last_flow[i]

        try:

            gps_start, gps_end = gps[data["stations"]
            [start]], gps[data["stations"][end]]

            # affichage de la ligne avec un colormap
            plt.plot((gps_start[1], gps_end[1]), (gps_start[0],
                                                  gps_end[0]), c=cmap(flow_value / max_flow))
        except:
            print("not found")

    plt.show()


if __name__ == "__main__":
    # debug()
    # convergence_couts()
    # metadata()
    print_relative_flow_costs()
    # affichage_dernier_flot()
