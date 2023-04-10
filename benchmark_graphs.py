"""
Benchmark graphs for the algorithm
Todolist:
- see carnets.binets.fr pour les différents trucs mentionnés pendant la réunion

A faire pour la première courbe :
# 1 ######################################################
Pour 5 destinations différentes (en même temps)
(au total : 646 stations, on peut faire un espacement de 100)

Etudier 2 chemins, 3 chemins, 4 chemins, 5 chemins, 6 chemins, 7 chemins
donner le coût total obtenu à la fin.

Puis pour 10 couples

=> faire 2 courbes sur le même graphe. Ca montrera que 5 chemins est un choix optimal

# 2 ######################################################
TODO: improve performance with multiprocessing
"""
import multiprocessing

import matplotlib.pyplot as plt

from compute_flow_cost import compute_flow_cost
from franck_wolfe import Paris
from util.util import read_json


def benchmark_correctness_for_n_paths():
    """Test first benchmark from the benchmark list
    The goal is to show correctness for least than 5 different paths"""
    print('Running correctness benchmark...')
    passengers = 1000
    n_values = [2, 3, 4, 5, 6, 7]

    # Generate 5 destinations
    destinations1 = [(i * 100, (i + 1) * 100, passengers) for i in range(5)]

    # Generate 10 destinations
    destinations2 = [(i * 50, (i + 1) * 50, passengers) for i in range(10)]

    # Create multiprocessing params
    args = [(n, destinations1 if i < 6 else destinations2, False) for i, n in enumerate(n_values * 2)]

    pool = multiprocessing.Pool(processes=12)

    results = pool.starmap(p.solve_paths, args)

    # Compute flow costs and normalize
    results = [compute_flow_cost(flow) for flow in results]

    # Normalize the first run
    results[:6] = [results[i] / max(results[:6]) for i in range(6)]

    # Normalize the second run
    results[6:] = [results[i] / max(results[6:]) for i in range(6, 12)]

    # Print flows
    plt.title("Correctness benchmark - paths per destination")
    plt.plot(n_values, results[:6], label='5 couples')
    plt.plot(n_values, results[6:], label='10 couples')
    plt.xlabel("Paths per couple")
    plt.ylabel('Total flow cost (UA)')
    plt.legend()
    plt.grid()
    plt.show()
    print("Correctness benchmark completed")


# TODO : comparer les aglos lents et rapides sur quelques chemins
# Ne tester qu'un couple (parce que l'algo lent n'en supporte qu'un),
# Mais sachant que le temps de calcul est proportionnel c'est pas très grave
# Objectif: trouver au moins un facteur 200 pour 1 couple. Comparer les performances en termes de coût
def benchmark_compare_fast_slow():
    pass


# TODO : algo rapide sur 100 sources et destinations isolées, à 5 chemins.
# En vrai je pense pas que ça soit nécessaire de les isoler, sur autant de chemins ça reviendra au même
def benchmark_heavy_fast():
    pass


if __name__ == "__main__":
    # Initialize data
    print("Loading data...")
    graph_data = read_json("paris_network.json")
    edge_distances = read_json("edge_distances.json")

    p = Paris(graph_data, edge_distances)
    print("Finished loading")

    benchmark_correctness_for_n_paths()
