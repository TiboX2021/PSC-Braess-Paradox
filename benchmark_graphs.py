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


def test_benchmark():
    """Test first benchmark from the benchmark list
    The goal is to show correctness for least than 5 different paths"""
    passengers = 1000

    # Generate 5 destinations
    destinations = [(i * 100, (i + 1) * 100, passengers) for i in range(5)]

    n_values = [2, 3, 4, 5, 6, 7]
    flow_costs = []

    for i, n in enumerate(n_values):
        print("Test", i + 1, "of", len(n_values))
        flow = p.solve_paths(n, destinations, False)
        flow_costs.append(compute_flow_cost(flow))

    # Normalize cost
    max_cost = max(flow_costs)
    print(max_cost)
    # flow_costs = [value / max_cost for value in flow_costs]

    # Print flows
    plt.plot(n_values, flow_costs)
    plt.show()


if __name__ == "__main__":
    # Initialize data
    graph_data = read_json("paris_network.json")
    edge_distances = read_json("edge_distances.json")

    p = Paris(graph_data, edge_distances)

    test_benchmark()
