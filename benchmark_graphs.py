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
"""
import matplotlib.pyplot as plt

from compute_flow_cost import compute_flow_cost
from franck_wolfe import Paris
from util.util import read_json


def benchmark_correctness_for_n_paths():
    """Test first benchmark from the benchmark list
    The goal is to show correctness for least than 5 different paths"""
    passengers = 1000

    # Generate 5 destinations
    destinations = [(i * 100, (i + 1) * 100, passengers) for i in range(5)]

    n_values = [2, 3, 4, 5, 6, 7]
    flow_costs1 = []

    for i, n in enumerate(n_values):
        print("Test", i + 1, "of", len(n_values))
        flow = p.solve_paths(n, destinations, False)
        flow_costs1.append(compute_flow_cost(flow))

    # Generate 10 destinations
    destinations = [(i * 50, (i + 1) * 50, passengers) for i in range(10)]
    flow_costs2 = []

    for i, n in enumerate(n_values):
        print("Test", i + 1 + len(n_values), "of", len(n_values) * 2)
        flow = p.solve_paths(n, destinations, False)
        flow_costs2.append(compute_flow_cost(flow))

    # Normalize costs
    flow_costs1 = [value / max(flow_costs1) for value in flow_costs1]
    flow_costs2 = [value / max(flow_costs2) for value in flow_costs2]

    # Print flows
    plt.title("Correctness benchmark - paths per destination")
    plt.plot(n_values, flow_costs1, label='5 couples')
    plt.plot(n_values, flow_costs2, label='10 couples')
    plt.xlabel("Paths per couple")
    plt.ylabel('Total flow cost (UA)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Initialize data
    graph_data = read_json("paris_network.json")
    edge_distances = read_json("edge_distances.json")

    p = Paris(graph_data, edge_distances)

    benchmark_correctness_for_n_paths()
