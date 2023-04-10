"""
Benchmark graphs for the algorithm

# SUMMARY
### FIRST : set the runtime parameters for both algorithms
1. Correctness benchmark for n paths (how many paths for correctness)
The optimal number of paths to chose  for the fast algorithm.
Make the algorithm run for 5 destinations, then 10 destinations using a selection of 2, 3, 4, 5, 6, 7 paths.
Compare the performance for each path count, and determine the best pathc count.

### SECOND : study the convergence for both algorithms : when to stop ?
POINT: the difference between consecutive flows cannot be used.
2. Convergence of the long algorithm : monitor cost for each iteration (normalize + mean for different paths)
For different destinations (in order to take the mean), keep track of the cost at each iteration until a very low
convergence threshold is reached.
Then, evaluate the number of iterations needed to reach 10%, 5%, 1%, 0.1% of the last flow cost.
Plot the number of iterations as a function of the percentage reached. TODO : log percentage?
Display the standard deviation for each result (matplotlib error bars)
=> From the resulting graph, chose a number of iterations for the long algorithm (1 path)

3. Convergence of the fast algorithm : monitor cost for each iteration (normalize + mean for different paths)
Same for the fast algorithm.
TODO : do the same and monitor the variation in iterations in function of the number of people and the number of paths
TODO : there might be an important difference


### THIRD : compare the two algorithms
TODO : do the rest of the benchmarks

4. Convergence threshold benchmark (what convergence threshold for correctness)
5. Compare fast and slow algorithms (how much faster is the fast algorithm)

### FOURTH : study the performance of the algorithm under heavy load
6. Heavy load benchmark (performance on heavy load)
"""
import multiprocessing
from time import time
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from compute_flow_cost import compute_flow_cost
from franck_wolfe import Paris
from util.util import read_json


def generate_pseudo_random_sources(n_sources: int, n_stations: int, min_diff: int = 50) -> Tuple[
    np.ndarray, np.ndarray]:
    """Generate pseudo random sources and destinations."""
    sources = np.random.randint(0, n_stations, n_sources)
    destinations = np.random.randint(0, n_stations, n_sources)
    for i, (source, destination) in enumerate(zip(sources, destinations)):
        while abs(source - destinations[i]) < min_diff:
            destinations[i] = np.random.randint(0, len(p.data["stations"]))

    return sources, destinations


def sliding_mean(l: List, index: int, sliding_range: int = 3):
    """Compute the mean of the last sliding_range values"""
    if index < 1:
        return l[0]
    elif index < sliding_range:
        return np.mean(l[:index + 1])
    else:
        return np.mean(l[index - sliding_range:index])


def get_first_iterations_reaching_percentage(costs: List[np.ndarray], percentages: List[float]) -> List[int]:
    """Get the first iteration for which the cost is below a given percentage of the last cost.
    Use a sliding mean to avoid noise."""
    last_cost = costs[-1]
    iterations = []

    index = 0

    for percentage_index in range(len(percentages)):
        percentage = percentages[percentage_index]
        threshold = (1 + percentage) * last_cost

        while index < len(costs) and sliding_mean(costs, index, 3) > threshold:
            index += 1
        iterations.append(index)

    return iterations


#############################################################################
#                      FIRST : ALGORITHM PARAMETERS                         #
#############################################################################
def benchmark_correctness_for_n_paths():
    """Test first benchmark from the benchmark list
    The goal is to show correctness for least than 5 different paths
    """
    print('Running correctness benchmark...')
    passengers = 1000
    n_values = [2, 3, 4, 5, 6, 7]

    # Generate 5 destinations
    destinations1 = generate_pseudo_random_sources(5, len(p.data["stations"]), min_diff=100)

    # Generate 10 destinations
    destinations2 = generate_pseudo_random_sources(10, len(p.data["stations"]), min_diff=100)

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


#############################################################################
#                    SECOND : ALGORITHMS CONVERGENCE                        #
#############################################################################

def benchmark_convergence_long():
    """Monitor the evolution of cost at each iteration for the long algorithm

    It appears that only 10 iterations are needed in order to reach the final cost
    How to properly write this benchmark?
    => Make the algorithm run for long (very low threshold)
    => Analyse the costs : take the iteration for 10%, 5%, 1%, 0.1%, 0.01% of the final cost,
    and see what is a good choice
    => For this result, take the mean
    """
    percentages = [0.1, 0.05, 0.01, 0.001, 0.0001]
    percentage_tags = ['10%', '5%', '1%', '0.1%', '0.01%']

    # TODO : take the mean for n different paths (better for representative results)
    # TODO : use multiprocessing to speedup the process

    # Pour l'instant : test avec les stations 0  à 100
    # Voluntarily chose a very low convergence threshold to gather more data
    print("Running convergence benchmark for the long algorithm...")
    flows = p.solve(convergence_threshold=5, destination=(0, 100, 1000), log=False, output_file="out.csv")

    costs = [compute_flow_cost(flow) for flow in flows]

    iterations = get_first_iterations_reaching_percentage(costs, percentages)

    plt.title("Convergence benchmark for the long algorithm")
    y_pos = np.arange(len(percentage_tags))
    plt.bar(y_pos, iterations, align='center')
    plt.xticks(y_pos, percentage_tags)
    plt.xlabel("Percentage of final cost")
    plt.ylabel('Number of iterations')
    # TODO : plot error bars
    plt.show()
    print("Convergence benchmark completed")


def benchmark_convergence_threshold():
    """Test optimal convergence threshold between precision and execution time.
    Because the comparison is done with a linear difference between consecutive flows, the threshold should be
    proportionnal to the total number of passengers. The key is to find the correct factor.
    Note: this test was done with the fast algorithm, as it has aproximately the same progression as the fast one

    TODO : run multiple instances with different sources and 5 paths in parallel with multiprocessing in order to go faster
    """
    # First test
    print("Running convergence threshold benchmark...")

    # Simulation parameters
    n_paths = 5
    passengers = 1000
    n_sources = 1
    # sources, destinations = generate_pseudo_random_sources(n_sources, len(p.data["stations"]), min_diff=50)
    # print("sources and destinations generated", sources, destinations)
    # TODO debug : la défense raspail
    sources = (0,)
    destinations = (100,)

    # Run benchmark
    total_passengers = n_sources * passengers
    dividers = [5000, 1000, 800, 500, 100, 50, 10, 5, 1, 0.1]
    convergence_thresholds = [total_passengers / divider for divider in dividers]
    threshold_costs = p.benchmark_convergence_thresholds(n_paths,
                                                         list(zip(sources, destinations, [passengers] * n_sources)),
                                                         tuple(convergence_thresholds), log=False)
    # Normalize costs
    threshold_costs = [threshold_cost / max(threshold_costs) for threshold_cost in threshold_costs]
    print(threshold_costs)
    print("Convergence threshold benchmark completed")

    # Plot results
    plt.title("Convergence threshold benchmark")
    plt.plot(convergence_thresholds, threshold_costs, label='5 couples')
    # plt.xscale('log')  # Log scale for the threshold dividers
    plt.xlabel('Passagers total divisé par...')
    plt.ylabel('Coût total du flux (UA)')
    plt.ylim((0, 1.1))
    plt.show()


# TODO : comparer les aglos lents et rapides sur quelques chemins
# Ne tester qu'un couple (parce que l'algo lent n'en supporte qu'un),
# Mais sachant que le temps de calcul est proportionnel c'est pas très grave
# Objectif: trouver au moins un facteur 200 pour 1 couple. Comparer les performances en termes de coût
def benchmark_compare_fast_slow():
    pass


# TODO : algo rapide sur 100 sources et destinations isolées, à 5 chemins.
# En vrai je pense pas que ça soit nécessaire de les isoler, sur autant de chemins ça reviendra au même
def benchmark_heavy_fast():
    n_sources = 100  # Number of sources and destinations
    n_paths = 5  # Number of paths per source
    passengers = 1000  # Number of passengers per source

    # Chose sources and destinations at random:
    sources, destinations = generate_pseudo_random_sources(n_sources, len(p.data["stations"]), min_diff=50)

    start = time()
    print("Starting heavy benchmark")

    # TODO: pour l'instant, je prends un seuil de convergence de PASSAGERS_TOT / 1000, il faut benchmark ça
    p.solve_paths(n_paths, list(zip(sources, destinations, [passengers] * n_sources)),
                  convergence_threshold=passengers * n_sources / 1000, log=False)

    exec_time = time() - start

    print(f"Fast algorithm took {exec_time} seconds")
    print("Heavy benchmark completed")
    # 20s when plugged in


if __name__ == "__main__":
    # Initialize data
    print("Loading data...")
    graph_data = read_json("paris_network.json")
    edge_distances = read_json("edge_distances.json")

    p = Paris(graph_data, edge_distances)
    print("Finished loading")

    # benchmark_correctness_for_n_paths()
    benchmark_convergence_long()

    # benchmark_heavy_fast()
    # benchmark_convergence_threshold()
