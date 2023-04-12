"""
Utilisation :

Modifier les 4 variables suivantes pour changer le flux calculé par l'algorithme
Vous pouvez voir les indices des stations dans le fichier paris_network.json (avec les numéros
des lignes en faisant une soustraction, ou directement si vous utilisez VSCode)
"""
import warnings
from time import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

from compute_flow_cost import compute_flow_cost
from util.util import Network, gen_matrix_A, write_json, read_json

# Disable warnings
warnings.filterwarnings("ignore", "Solving system with option")
warnings.filterwarnings("ignore", "Ill-conditioned matrix")

################################################################################
#                       CALCUL D"UN TRAJET SUR PARIS                           #
################################################################################
NOMBRE_DE_PASSAGERS = 1000
STATION_DEPART = 0  # 210 # Invalides
STATION_ARRIVEE = 100  # 68 # République
SEUIL_CONVERGENCE = 5  # La norme 1 de la différence entre 2 flux consécutifs doit être plus petite que ça


# Helper functions

def error_percentage(f1: np.ndarray, f2: np.ndarray) -> float:
    """Return the error percentage between two (consecutive) flows, relative to f1.
    Useful for checking the convergence of the algorithm in a relative (not absolute)"""
    error = np.sum(np.abs(f1 - f2))
    return error / np.sum(np.abs(f1))


class Paris:
    A: np.ndarray
    B: np.ndarray  # pour un seul couple pour l'instant, tout le reste sera nul
    c: np.ndarray  # coüts

    a: np.ndarray  # debug: coût linéaire arbitraire
    data: Network

    def __init__(self, data: Network, edge_distances: List[float]):
        """Chargement des données du graphe de Paris dans une matrice
        Pour les correspondances, je vais mettre un truc + x, genre 5000
        (il faut que ça soit plus gros que la plus grosse arête)
        """
        # Store graph data
        self.data = data

        # Création de la matrice A
        edges = (
                data["edges"]
                + data["metro_connections"]
                + data["rer_connections"]
                + data["trans_connections"]
        )
        self.edges = edges
        #############################################################################
        #                             SHARED VARIABLES                              #
        #############################################################################

        # Matrix that represents the network (shared between all methods)
        self.A = gen_matrix_A(vertices=len(data["stations"]), edges=edges)

        # COST COEFFS for the edge method
        # Coefficient multiplié par x pour chaque flux
        self.a = np.ones(len(edges))

        # Coefficient d'ordonnée à l'origine
        self.b = edge_distances + [5000] * (len(edges) - len(edge_distances))

        #############################################################################
        #                           NOT SHARED VARIABLES                            #
        #############################################################################
        # TODO : eventually, remove these variables
        # Création de la matrice B: on fait aller 1000 personnes d'où à où?
        self.B = np.zeros(len(data["stations"]))

        # Test avec les 2 premières stations
        self.B[STATION_DEPART] = -NOMBRE_DE_PASSAGERS
        self.B[STATION_ARRIVEE] = +NOMBRE_DE_PASSAGERS

    def compute_edge_costs(self, flow):
        """Calcule les coûts par arête"""
        return self.a * flow + self.b

    def solve(self, convergence_threshold=5,
              destination: Tuple[int, int, int] = (STATION_DEPART, STATION_ARRIVEE, NOMBRE_DE_PASSAGERS), log=True,
              output_file: str = None) -> np.ndarray:
        """Résout le problème avec la méthode des arêtes (long)"""

        setup_time = time()

        # Create b linprog vector
        b = np.zeros(len(self.data["stations"]))

        start, end, passengers = destination

        # Test avec les 2 premières stations
        b[start] = -passengers
        b[end] = +passengers

        # Initial cost (empty)
        edge_costs = np.zeros(len(self.edges))

        # Les flux + calcul du flux initial
        flow = linprog(
            edge_costs,
            A_eq=self.A,
            b_eq=self.B,
        )["x"]

        # Create last flow & flow list
        last_flow = np.zeros(flow.shape)
        flows = []

        i = 0

        setup_time = time() - setup_time
        loop_time = time()

        while np.sum(np.abs(flow - last_flow)) > convergence_threshold:

            step = 1 / (i + 2)

            if output_file is not None:
                flows.append(flow)

            # Update the cost for each edge
            edge_costs = self.compute_edge_costs(flow)

            last_flow = flow

            # Solve the linear problem
            gradient = linprog(
                edge_costs,
                A_eq=self.A,
                b_eq=self.B,
            )["x"]

            # Compute the next flow
            flow = (1 - step) * last_flow + step * gradient

            i += 1

            if log:
                error = error_percentage(flow, last_flow)
                print(
                    "Itération n°",
                    i,
                    "erreur :",
                    error,
                    "écart",
                    np.sum(np.abs(flow - last_flow)),
                )

        # Calcul des erreurs cumulées par rapport à la dernière valeur
        if output_file is not None:
            #  noinspection PyTypeChecker
            # np.savetxt(output_file, flows, delimiter=",")

            print("convergence après", i, "itérations")

            # loop_time = time() - loop_time
            # print("TOTAL TIME :", setup_time + loop_time, "s")
            # print("Setup took", setup_time, "s")
            # print("Loop time took", loop_time, "s")

            return flows

        # Return the last flow :
        return flow

    def solve_paths(self, n: int = 5,
                    couples: List[Tuple[int, int, int]] = ((STATION_DEPART, STATION_ARRIVEE, NOMBRE_DE_PASSAGERS),),
                    convergence_threshold=5,
                    log=True, log_all=False) -> np.ndarray:
        """Solve a problem with multiple (start, end) couples with different amounts of passengers using the smart
        path algorithm
        """
        #############################################################################
        #                           COMPUTE ALL PATHS                               #
        #############################################################################
        first_n_paths_time = time()  # Start time

        boolean_paths = np.empty((n * len(couples), len(self.edges)))  # Uninitialized, must be filled

        # Fill the paths
        for index, (start, end, _) in enumerate(couples):
            boolean_paths[index * n: (index + 1) * n] = self.first_paths(n, start, end, log=log)

        first_n_paths_time = time() - first_n_paths_time  # End time

        setup_time = time()  # Setup time
        #############################################################################
        #                           BUILD LINPROG MATRIX                            #
        #############################################################################
        # The only constraint is that the sum of passengers along all paths is always equal to the initial total number
        A = np.tile([1] * n + [0] * len(couples) * n, len(couples))[:-len(couples) * n].reshape(
            (len(couples), len(couples) * n))  # Sum passengers for each path
        b = np.array([passengers for start, end, passengers in couples])  # Total numbers of passengers

        #############################################################################
        #                             BUILD COST MATRIX                             #
        #############################################################################
        # The cost is computed for each edge as a * flow + b
        # Agregate a & b coefficients for every path
        # An A matrix must be built to account for overlapping edges
        # The b coefficient is constant and does not need to be adjusted

        # Boolean_paths is of size n * edges
        # Diagonal matrix of size edges * edges
        diagonal_a = np.diag(self.a)

        # The result matrix is of size n * n
        agregated_A_cost_matrix = boolean_paths @ diagonal_a @ boolean_paths.T

        # Same for b, but only a vector is needed
        agregated_B_cost_vector = boolean_paths @ self.b

        def compute_cost_vector(flow_vector: np.ndarray) -> np.ndarray:
            """Compute the cost vector (of size n) from the flow vector (of size n)"""
            return agregated_A_cost_matrix @ flow_vector + agregated_B_cost_vector

        #############################################################################
        #                   COMPUTE INITIAL COST WITHOUT FLOW                       #
        #############################################################################

        cost_vector = compute_cost_vector(np.zeros(n * len(couples)))

        # Solve the linear problem for this initial cost
        flow = linprog(
            cost_vector,  # Cost vector : minimise the dot product cost_vector @ flow
            A_eq=A,
            b_eq=b,
        )["x"]

        # Store last flow value
        last_flow = np.zeros(flow.shape)

        i = 0

        setup_time = time() - setup_time
        loop_time = time()

        all_flows = []

        while np.sum(np.abs(flow - last_flow)) > convergence_threshold:
            step = 1 / (i + 2)

            # Update the cost
            cost_vector = compute_cost_vector(flow)

            # Update last flow value
            last_flow = flow
            if log_all:
                all_flows.append(boolean_paths.T @ flow)

            # Solve the linear problem
            gradient = linprog(
                cost_vector,
                A_eq=A,
                b_eq=b,
            )["x"]

            # Compute the next flow
            flow = (1 - step) * last_flow + step * gradient

            i += 1

            # DEBUG : print error percentage to see the progress
            if log:
                error = error_percentage(flow, last_flow)
                print(
                    "Itération n°",
                    i,
                    "erreur :",
                    error,
                    "écart",
                    np.sum(np.abs(flow - last_flow)),
                )

        #############################################################################
        #               REBUILD THE LAST FLOW FROM PATHS TO EDGES                   #
        #############################################################################      

        # Rebuild flow
        converted_flow = boolean_paths.T @ flow

        if log_all:
            return all_flows

        if log:
            print("convergence après", i, "itérations")

            loop_time = time() - loop_time
            print("TOTAL TIME :", first_n_paths_time + setup_time + loop_time, "s")
            print("First", n, "paths took", first_n_paths_time, "s")
            print("Setup took", setup_time, "s")
            print("Loop time took", loop_time, "s")

            # Save in in a json file
            write_json("fast_last_flow.json", list(converted_flow))
        return converted_flow

    def first_paths(self, n: int, start: int, end: int, passengers: int = NOMBRE_DE_PASSAGERS,
                    log: bool = False) -> np.ndarray:
        """
        Evaluate the first n paths and log them Returns an array of n arrays representign circulation along the edges
        * start : the start station
        * end : the end station
        """

        #############################################################################
        #                          LINPROG A_EQ AND B_EQ                            #
        #############################################################################
        # Reuse A matrix
        A = self.A

        b = np.zeros(len(self.data["stations"]))
        b[start] = -passengers
        b[end] = +passengers

        #############################################################################
        #                       INITIAL FLOW & STORE FLOWS                          #
        #############################################################################

        # Compute initial cost for empty flow
        costs = self.compute_edge_costs(np.zeros(len(self.a)))

        # Compute initial flow
        flow = linprog(
            costs,
            A_eq=A,
            b_eq=b,
        )["x"]

        # Store consecutive flows in this variable
        flows = []

        for i in range(n):
            if log:
                print("Step", i + 1, "of", n, "...")

            step = 1 / (i + 2)

            # Update the cost for each edge
            costs = self.compute_edge_costs(flow)

            # Update last flow
            last_flow = flow

            # Solve the linear problem
            gradient = linprog(
                costs,
                A_eq=A,
                b_eq=b,
            )["x"]

            # Log this path
            flows.append(gradient)

            # Compute the next flow
            flow = (1 - step) * last_flow + step * gradient

        boolean_flows = np.array(flows) != 0

        if log:
            # Save paths
            print(n, "paths stored. Saving...")

            # De-numpify for json serialization
            flows = [x.tolist() for x in boolean_flows]

            success = write_json(f"first-{n}-paths.json", flows)
            print("Saving done with status", success)

        return boolean_flows

    def benchmark_convergence_thresholds(self, n: int = 5,
                                         couples: List[Tuple[int, int, int]] = (
                                                 (STATION_DEPART, STATION_ARRIVEE, NOMBRE_DE_PASSAGERS),),
                                         convergence_thresholds: Tuple[float] = (5,),
                                         log=False) -> List[float]:
        """Test convergence thresholds. The cost computed for each threshold will be returned at the end"""
        #############################################################################
        #                           COMPUTE ALL PATHS                               #
        #############################################################################
        if convergence_thresholds is None:
            convergence_thresholds = [5]

        boolean_paths = np.empty((n * len(couples), len(self.edges)))  # Uninitialized, must be filled

        # Fill the paths
        for index, (start, end, _) in enumerate(couples):
            boolean_paths[index * n: (index + 1) * n] = self.first_paths(n, start, end, log=log)

        setup_time = time()  # Setup time
        #############################################################################
        #                           BUILD LINPROG MATRIX                            #
        #############################################################################
        # The only constraint is that the sum of passengers along all paths is always equal to the initial total number
        A = np.tile([1] * n + [0] * len(couples) * n, len(couples))[:-len(couples) * n].reshape(
            (len(couples), len(couples) * n))  # Sum passengers for each path
        b = np.array([passengers for start, end, passengers in couples])  # Total numbers of passengers

        #############################################################################
        #                             BUILD COST MATRIX                             #
        #############################################################################
        # The cost is computed for each edge as a * flow + b
        # Agregate a & b coefficients for every path
        # An A matrix must be built to account for overlapping edges
        # The b coefficient is constant and does not need to be adjusted

        # Boolean_paths is of size n * edges
        # Diagonal matrix of size edges * edges
        diagonal_a = np.diag(self.a)

        # The result matrix is of size n * n
        agregated_A_cost_matrix = boolean_paths @ diagonal_a @ boolean_paths.T

        # Same for b, but only a vector is needed
        agregated_B_cost_vector = boolean_paths @ self.b

        def compute_cost_vector(flow_vector: np.ndarray) -> np.ndarray:
            """Compute the cost vector (of size n) from the flow vector (of size n)"""
            return agregated_A_cost_matrix @ flow_vector + agregated_B_cost_vector

        #############################################################################
        #                   COMPUTE INITIAL COST WITHOUT FLOW                       #
        #############################################################################

        cost_vector = compute_cost_vector(np.zeros(n * len(couples)))

        # Solve the linear problem for this initial cost
        flow = linprog(
            cost_vector,  # Cost vector : minimise the dot product cost_vector @ flow
            A_eq=A,
            b_eq=b,
        )["x"]

        # Store last flow value
        last_flow = np.zeros(flow.shape)

        i = 0
        #############################################################################
        #                  BUILD TOTAL COST FOR EACH THRESHOLD                      #
        #############################################################################
        total_costs = [0] * len(convergence_thresholds)
        convergence_thresholds = list(convergence_thresholds)
        convergence_thresholds.sort(reverse=True)
        print("convergence_thresholds",
              convergence_thresholds)  # TODO : debug. A mon avis il faudra log chacun des coûts pour voir leur évolution
        current_threshold_index = 0
        loop = True

        while loop:  # np.sum(np.abs(flow - last_flow)) > convergence_threshold:

            step = 1 / (i + 2)

            # Update the cost
            cost_vector = compute_cost_vector(flow)

            # Update last flow value
            last_flow = flow

            # TODO debug: affichage du coût correspondant pour en monitorer l'évolution au cours du temps
            # TODO : j'ai l'impression qu'il suffit de 2 itérations pour obtenir un bon coût.
            # Je vais comparer l'évolution avec le coût obtenu via l'algo lent, parce que c'est fou
            # Remarque : d'oû l'importance de différents graphes pour évaluer l'algorithme !
            # TODO: refaire tourner l'algo lent en monitorant les coûts à chaque itération, pour bien comprendre comment ça marche
            total_flow = boolean_paths.T @ flow
            total_cost = compute_flow_cost(total_flow)
            print("total cost for iteration", i, "is", total_cost)

            # Solve the linear problem
            gradient = linprog(
                cost_vector,
                A_eq=A,
                b_eq=b,
            )["x"]

            # Compute the next flow
            flow = (1 - step) * last_flow + step * gradient

            i += 1

            # DEBUG : print error percentage to see the progress
            if log:
                error = error_percentage(flow, last_flow)
                print(
                    "Itération n°",
                    i,
                    "erreur :",
                    error,
                    "écart",
                    np.sum(np.abs(flow - last_flow)),
                )
            # Threshold computation
            current_diff = np.sum(np.abs(flow - last_flow))
            if current_diff < convergence_thresholds[current_threshold_index]:
                if True:
                    print('Switching to next threshold :', current_threshold_index + 1, 'of',
                          len(convergence_thresholds))
                # Compute the cost for the total flow
                total_flow = boolean_paths.T @ flow
                total_costs[current_threshold_index] = compute_flow_cost(total_flow)
                # Increment the threshold index
                current_threshold_index += 1
                # If we reached the end of the thresholds, stop the loop
                if current_threshold_index == len(convergence_thresholds):
                    loop = False

        return total_costs


if __name__ == "__main__":
    graph_data = read_json("paris_network.json")
    edge_distances = read_json("edge_distances.json")

    p = Paris(graph_data, edge_distances)

    # p.solve()
    # p.solve_paths()
    # p.first_paths(n=5, start=0, end=100)

    # Essai avec les 2 chemins à la fois
    # p.solve_paths(n=5, couples=[(0, 100, NOMBRE_DE_PASSAGERS), (210, 68, NOMBRE_DE_PASSAGERS)])
    # p.solve_paths(n=5, couples=[(0, 100, NOMBRE_DE_PASSAGERS)])

    # Essai sur plusieurs couples avec beaucoup de trafic :
    p.solve_paths(n=5, couples=[
        (331, 280, NOMBRE_DE_PASSAGERS),
        (82, 266, NOMBRE_DE_PASSAGERS),
        (312, 294, NOMBRE_DE_PASSAGERS),
        (109, 249, NOMBRE_DE_PASSAGERS),
    ])

    # Boulogne Jean-Jaurès -> Jourdain : 331 - 280
    # Château Rouge -> Voltaire        : 82  - 266
    # Vaugirard -> Abbesses            : 312 - 294
    # Hoche -> Rue de la Pompe         : 109 - 249
