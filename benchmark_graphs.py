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
# TODO: see if this is related to the total number of passengers. Test 10 destinations with 500 people instead to match
5 destinations with 1000 people.


### THIRD : compare the two algorithms
TODO : do the rest of the benchmarks
4. Compare fast and slow algorithms (how much faster is the fast algorithm)

### FOURTH : study the performance of the algorithm under heavy load
5. Heavy load benchmark (performance on heavy load for the fast algorithm)

TODO: voir les notes évoquées en réunion. Il faut qu'en présentant les graphes dans l'ordre on comprenne
directement de quoi on parle
"""
import multiprocessing
from time import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from compute_flow_cost import compute_flow_cost
from franck_wolfe import Paris
from util.util import read_json, write_json


def solve_paths_time(a, b, c, d, e):
    start = time()
    p.solve_paths(a, b, c, d, e)
    end = time() - start
    return end


def generate_pseudo_random_sources(
    n_sources: int, n_stations: int, min_diff: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
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
        return np.mean(l[: index + 1])
    else:
        return np.mean(l[index - sliding_range : index])


def get_first_iterations_reaching_percentage(
    costs: List[np.ndarray], percentages: List[float]
) -> List[int]:
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
    print("Running correctness benchmark...")
    passengers = 1000
    n_values = [2, 3, 4, 5, 6, 7]

    # Generate 5 destinations
    destinations1 = generate_pseudo_random_sources(
        5, len(p.data["stations"]), min_diff=100
    )

    # Generate 10 destinations
    destinations2 = generate_pseudo_random_sources(
        10, len(p.data["stations"]), min_diff=100
    )

    # Create multiprocessing params
    args = [
        (n, destinations1 if i < 6 else destinations2, False)
        for i, n in enumerate(n_values * 2)
    ]

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
    plt.plot(n_values, results[:6], label="5 couples")
    plt.plot(n_values, results[6:], label="10 couples")
    plt.xlabel("Paths per couple")
    plt.ylabel("Total flow cost (UA)")
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
    percentage_tags = ["10%", "5%", "1%", "0.1%", "0.01%"]

    # Generate args
    n_destinations = 100  # Nombre de simulations
    n_passengers = 1000  # For 1 path, 1000 total passengers.
    starts, ends = generate_pseudo_random_sources(
        n_destinations, len(p.data["stations"]), min_diff=100
    )
    # Create function arguments
    # TODO : adjust convergence_threshold=5 (the left iterations are not needed)
    args = [
        (
            100,
            (start, end, n_passengers),
            False,
            "out.csv",
        )  # Le premier est le nombre d'itérations à faire
        for start, end in zip(starts, ends)
    ]

    # Run parallelized benchmarks
    print("Running convergence benchmark for the long algorithm...")
    pool = multiprocessing.Pool(processes=12)
    flows_list = pool.starmap(p.solve, args)  # Generates flows for each path

    # flows = p.solve(convergence_threshold=5, destination=(0, 100, 1000), log=False, output_file="out.csv")
    # Compute all costs
    costs_list = [[compute_flow_cost(flow) for flow in flows] for flows in flows_list]
    iterations_list = [
        get_first_iterations_reaching_percentage(costs, percentages)
        for costs in costs_list
    ]

    # Numpify arrays for easier computation
    np_iterations_list = np.array(iterations_list)

    # plot only the mean result
    means = np.mean(np_iterations_list, axis=0)

    plt.title("Convergence benchmark for the long algorithm")
    y_pos = np.arange(len(percentage_tags))
    plt.plot(range(len(means)), means, "+b")
    plt.xticks(y_pos, percentage_tags)
    # plt.gca().invert_yaxis()  # Reverse y axis
    plt.gca().yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # Set x axis to be integers
    plt.xlabel("Percentage of final cost")
    plt.ylabel("Number of iterations")
    plt.grid()
    plt.show()
    print("Convergence benchmark completed")


def benchmark_convergence_fast():
    """Test optimal convergence threshold between precision and execution time.
    Because the comparison is done with a linear difference between consecutive flows, the threshold should be
    proportionnal to the total number of passengers. The key is to find the correct factor.
    Note: this test was done with the fast algorithm, as it has aproximately the same progression as the fast one
    """

    percentages = [0.1, 0.05, 0.01, 0.001, 0.0001]
    percentage_tags = ["10%", "5%", "1%", "0.1%", "0.01%"]

    # Simulation parameters
    n_paths = 5
    passengers = 1000  # in {500, 1000}
    n_sources = 10  # in {5, 10} Nombre de destinations
    n_simulations = 100  # Simulations en parallèle (20)
    # Avec la nouvelle version, 100 itérations suffiront (au vu de l'output pour l'algo lent)
    convergence_threshold = 100

    # Generate random sources
    def convert_random_sources_to_args(couples: Tuple[np.ndarray, np.ndarray]):
        return list(zip(couples[0], couples[1], [passengers] * len(couples[0])))

    # Creating arguments
    args = (
        (
            n_paths,
            convert_random_sources_to_args(
                generate_pseudo_random_sources(
                    n_sources, len(p.data["stations"]), min_diff=100
                )
            ),
            convergence_threshold,
            False,
            True,
        )
        for _ in range(n_simulations)
    )

    print("Running convergence benchmark for the fast algorithm...")
    pool = multiprocessing.Pool(processes=15)
    # Generates flows for each path
    flows_list = pool.starmap(p.solve_paths, args)

    # Flows_list : all flows for the 20 simulations.
    # flows = p.solve(convergence_threshold=5, destination=(0, 100, 1000), log=False, output_file="out.csv")
    # Compute all costs
    costs_list = [[compute_flow_cost(flow) for flow in flows] for flows in flows_list]
    iterations_list = [
        get_first_iterations_reaching_percentage(costs, percentages)
        for costs in costs_list
    ]

    # Numpify arrays for easier computation
    np_iterations_list = np.array(iterations_list)

    print("Convergence threshold benchmark completed")

    # plt.figure(dpi=300)  # Increased figure scale
    plt.title("Convergence benchmark for the fast algorithm")
    # Vert=false for horizontal boxplot
    plt.boxplot(np_iterations_list)
    plt.xticks(np.arange(1, len(percentage_tags) + 1), percentage_tags)
    # plt.gca().invert_yaxis()  # Reverse y axis
    plt.gca().yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # Set x axis to be integers
    plt.xlabel("Percentage of final cost")
    plt.ylabel("Number of iterations")
    plt.show()


# TODO : comparer les aglos lents et rapides sur quelques chemins
# Ne tester qu'un couple (parce que l'algo lent n'en supporte qu'un),
# Mais sachant que le temps de calcul est proportionnel c'est pas très grave
# Objectif: trouver au moins un facteur 200 pour 1 couple. Comparer les performances en termes de coût
# TODO : utiliser le nombre d'itérations trouvé dans les benchmarks précédents
def benchmark_compare_fast_slow():
    """On fait tourner les 2 algorithmes sur une même sélection de chemins, puis on compare le temps de fin de cv.
    D'après les benchmarks précédents, on les fait tourner sur 50 itérations chacun.
    On comparera les coûts et les temps de convergence"""

    # generate 100 destinations
    n_destinations = 100  # TODO : 100
    n_passagers = 1000
    n_iterations = 100
    starts, ends = generate_pseudo_random_sources(
        n_destinations, len(p.data["stations"]), min_diff=100
    )

    # prepare args. Type de retour : renvoyer temps, coût
    fast_args = [
        (5, ((start, end, n_passagers),), n_iterations, False, False)
        for start, end in zip(starts, ends)
    ]
    slow_args = [
        (n_iterations, (start, end, n_passagers), False, None)
        for start, end in zip(starts, ends)
    ]

    # Run both algos
    pool = multiprocessing.Pool(processes=16)

    results_fast = pool.starmap(p.solve_paths, fast_args)
    results_slow = pool.starmap(p.solve, slow_args)

    # list of (time, cost)

    # Compute and plot results ? As boxplot ?
    time_comparison = [
        fast[0] / slow[0] for fast, slow in zip(results_fast, results_slow)
    ]
    cost_comparison = [
        fast[1] / slow[1] for fast, slow in zip(results_fast, results_slow)
    ]

    # save the results in a file for later use
    write_json("time_comparison.json", time_comparison)
    write_json("cost_comparison.json", cost_comparison)


# TODO : algo rapide sur 100 sources et destinations isolées, à 5 chemins.
# En vrai je pense pas que ça soit nécessaire de les isoler, sur autant de chemins ça reviendra au même
def benchmark_heavy_fast():
    """Benchmark du temps que ça prend en fonction du nombre de destinations
    TODO : faire aussi un boxplot vertical, comme pour les autres
    """

    destinations = [5, 10, 50, 100, 300, 500]
    max_dest = max(destinations)

    n_paths = 5  # Number of paths per source
    passengers = 1000  # Number of passengers per source
    iterations = 100
    moyennes = 20

    departs, arrivees = generate_pseudo_random_sources(
        max_dest, len(p.data["stations"]), min_diff=50
    )

    args_destinations = list(zip(departs, arrivees, [passengers] * max_dest))

    exec_times = []

    print("Starting heavy benchmark")
    pool = multiprocessing.Pool(processes=moyennes)

    for destination in destinations:
        print(f"Starting benchmark for {destination} destinations")

        # Construire les args : moyennes * le même truc
        args = []
        for _ in range(moyennes):
            args.append(
                (n_paths, args_destinations[:destination], iterations, False, False)
            )

        exec_times.append(pool.starmap(solve_paths_time, args))

    print("Heavy benchmark completed")
    write_json("heavy.json", exec_times)

    # Affichage des résultats sur un boxplot avec le temps
    plt.title("Convergence benchmark for the fast algorithm")
    plt.boxplot(exec_times)
    plt.xticks(np.arange(1, len(exec_times) + 1), destinations)
    plt.xlabel("Number of destinations")
    plt.ylabel("Execution time (seconds)")
    plt.grid()
    plt.show()


def benchmark_fast_on_irl_paths():
    """Benchmark the fas algorithm on real heavy duty data
    (1M passengers on peak hours)"""

    start = time()

    # 100 iterations ?

    result = p.solve_paths(
        5, [(0, 100, 1000000)], convergence_threshold=100, log=False, log_all=False
    )

    end = time() - start


def display_fast_slow_benchmark():
    time_comparison = read_json("time_comparison.json")
    cost_comparison = read_json("cost_comparison.json")

    # Debug : affichage des résultats pour le temps
    times = [1 / time for time in time_comparison]

    # DEBUG : affichage des résultats pour le coût
    costs = [1 / cost for cost in cost_comparison]

    plt.title("Rapport des temps d'exécution de l'algo rapide / complet")
    plt.boxplot([times])
    plt.xticks(range(1, 2), ["Time"])
    plt.ylabel("Rapport complet / rapide")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Initialize data
    print("Loading data...")
    graph_data = read_json("paris_network.json")
    edge_distances = read_json("edge_distances.json")

    p = Paris(graph_data, edge_distances)
    print("Finished loading")

    # benchmark_correctness_for_n_paths()
    # benchmark_convergence_long()
    # benchmark_convergence_fast()
    # benchmark_compare_fast_slow()
    # benchmark_convergence_threshold()
    # display_fast_slow_benchmark()
    benchmark_heavy_fast()
