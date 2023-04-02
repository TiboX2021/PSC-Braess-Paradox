"""
Utilisation :

Modifier les 4 variables suivantes pour changer le flux calculé par l'algorithme
Vous pouvez voir les indices des stations dans le fichier paris_network.json (avec les numéros
des lignes en faisant une soustraction, ou directement si vous utilisez VSCode)
"""

################################################################################
#                       CALCUL D"UN TRAJET SUR PARIS                           #
################################################################################
NOMBRE_DE_PASSAGERS = 1000
STATION_DEPART = 0  # 210 # Invalides
STATION_ARRIVEE = 100  # 68 # République
SEUIL_CONVERGENCE = 5  # La norme 1 de la différence entre 2 flux consécutifs doit être plus petite que ça
# Mettre 50 voire plus pour que l'algo s'arrête plus tôt. Ça dépend aussi du nombre de passagers


"""
Showcase pour le paradoxe de braess, avec affichage des résultats et tout

    x/100       45
  _________A_________
 /         |         \ 
S          | 0.1      E
 \_________B_________/
    45        X/100


ordre des arêtes


S -> A
A -> E
S -> B
A -> B
B -> A
B -> E

"""

import warnings
from enum import Enum
from time import time
# Disable warnings
from typing import List, TypedDict, Callable, Dict, Tuple

import numpy as np
from scipy.optimize import linprog

from util.util import Network, gen_matrix_A, write_json, read_json

warnings.filterwarnings("ignore", "Solving system with option")
warnings.filterwarnings("ignore", "Ill-conditioned matrix")


# Helper functions


def error_percentage(f1: np.ndarray, f2: np.ndarray) -> float:
    """Return the error percentage between two (consecutive) flows, relative to f1.
    Useful for checking the convergence of the algorithm in a relative (not absolute)"""
    error = np.sum(np.abs(f1 - f2))
    return error / np.sum(np.abs(f1))


class GraphProblem(TypedDict):
    """Describes a flow problem, with a graph and a single (start, end) couple"""

    A: np.ndarray
    b: np.ndarray
    cost_function: Callable[[np.ndarray], np.ndarray]

    initial_flow: np.ndarray


class Braess:
    class BraessType(Enum):
        """Types of the graph used for showcasing the Braess Paradox"""

        WITHOUT_AB = 1
        WITH_AB = 2
        WITH_UPDATED_COSTS = 3

    CONVERGENCE_THRESHOLD = 50

    BRAESS_GRAPHS: Dict[BraessType, GraphProblem] = {
        BraessType.WITHOUT_AB: {
            "A": np.array(
                [
                    [
                        -1,
                        0,
                        -1,
                        0,
                    ],
                    [
                        1,
                        -1,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        1,
                        -1,
                    ],
                    [
                        0,
                        1,
                        0,
                        1,
                    ],
                ]
            ),
            "b": np.array([-4000, 0, 0, +4000]),
            "cost_function": lambda flow: np.array(
                [flow[0] / 100, 45, 45, flow[3] / 100]
            ),
            "initial_flow": np.array([4000, 4000, 0, 0]),
        },
        BraessType.WITH_AB: {
            "A": np.array(
                [
                    [
                        -1,
                        0,
                        -1,
                        0,
                        0,
                        0,
                    ],
                    [
                        1,
                        -1,
                        0,
                        -1,
                        1,
                        0,
                    ],
                    [
                        0,
                        0,
                        1,
                        1,
                        -1,
                        -1,
                    ],
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                    ],
                ]
            ),
            "b": np.array([-4000, 0, 0, +4000]),
            "cost_function": lambda flow: np.array(
                [flow[0] / 100, 45, 45, 0, 0, flow[5] / 100]
            ),
            "initial_flow": np.array([4000, 4000, 0, 0, 0, 0]),
        },
        BraessType.WITH_UPDATED_COSTS: {
            "A": np.array(
                [
                    [
                        -1,
                        0,
                        -1,
                        0,
                        0,
                        0,
                    ],
                    [
                        1,
                        -1,
                        0,
                        -1,
                        1,
                        0,
                    ],
                    [
                        0,
                        0,
                        1,
                        1,
                        -1,
                        -1,
                    ],
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                    ],
                ]
            ),
            "b": np.array([-4000, 0, 0, +4000]),
            "cost_function": lambda flow: np.array(
                [
                    flow[0] ** 2 / 200,
                    flow[1] * 45,
                    flow[2] * 45,
                    0,
                    0,
                    flow[5] ** 2 / 200,
                ]
            ),
            "initial_flow": np.array([4000, 4000, 0, 0, 0, 0]),
        },
    }

    def braess(graph_type: BraessType):
        """
        Algo du paradoxe de braess

        Convergence: the algorithm stops when the sum of the absolute value of the
        difference between two consecutive flow is less than the threshold.
        """
        graph = Braess.BRAESS_GRAPHS[graph_type]

        A = graph["A"]
        b = graph["b"]

        cost_function = graph["cost_function"]

        flow = graph["initial_flow"]
        last_flow = np.zeros(flow.shape)

        i = 0

        while np.sum(np.abs(flow - last_flow)) > Braess.CONVERGENCE_THRESHOLD:
            step = 1 / (i + 2)

            # Update the cost for each edge
            costs = cost_function(flow)

            last_flow = flow

            # Solve the linear problem
            next_flow = linprog(
                costs,
                A_eq=A,
                b_eq=b,
                options={
                    "rr": False,
                },
            )["x"]

            # Compute the next flow
            flow = (1 - step) * last_flow + step * next_flow

            i += 1

        # Printing results
        print()
        print("###############################################################")
        print("FINAL FLOW")
        print("###############################################################")
        print()
        print("Convergence reached at iteration", i + 1)
        print(
            "The last two consecutive flows have less than",
            Braess.CONVERGENCE_THRESHOLD,
            "people of difference in total",
        )
        print()
        Braess.pretty_print_braess(
            values=flow,
            AB=False if graph_type == Braess.BraessType.WITHOUT_AB else True,
        )
        print()
        print("Total cost :", int(np.round(Braess.standard_cost_function(flow))))

    @staticmethod
    def standard_cost_function(flow: np.ndarray) -> float:

        if len(flow) == 4:
            return flow[0] ** 2 / 100 + 45 * flow[1] + 45 * flow[2] + flow[3] ** 2 / 100
        elif len(flow) == 6:
            return flow[0] ** 2 / 100 + 45 * flow[1] + 45 * flow[2] + flow[5] ** 2 / 100
        else:
            raise Exception(f"Bad flow size :", len(flow))

    ################################################################################
    #                            Pretty printing                                   #
    ################################################################################

    @staticmethod
    def embed_number(number: int, size: int, char=" ") -> str:
        """Create a string of size <size> composed of characters <char>
        where the number is centered"""

        assert len(char) == 1, "char must be a string of size 1"

        n_size = len(str(number))

        if n_size >= size:
            return str(number)

        start = (size - n_size) // 2  # beginning index of the number

        return f"{start * char}{number}{(size - start - n_size) * char}"

    @staticmethod
    def pretty_print_braess(values: List[int], AB: bool) -> None:
        """Pretty print the Braess paradox graph with the given values.
        * AB: whether to show the A <-> B edge or not. Showing it will require one more value"""

        n_values = 6 if AB else 4

        values = np.round(values, decimals=0).astype(int)

        assert (
                len(values) == n_values
        ), f"{n_values} values required, only {len(values)} provided"

        print(
            f"{Braess.embed_number(values[0], size=12)}{Braess.embed_number(values[1], size=11)}\n"
            "  _________A_________  \n"
            " /         |         \\ \n"
            f"S          |{Braess.embed_number(abs(values[2] - values[3]), size=10) if AB else ' ' * 10}E\n"
            " \\_________B_________/ \n"
            f"{Braess.embed_number(values[2 + AB * 2], size=12)}{Braess.embed_number(values[3 + AB * 2], size=11)}\n"
        )


class Paris:
    A: np.ndarray
    B: np.ndarray  # pour un seul couple pour l'instant, tout le reste sera nul
    c: np.ndarray  # coüts

    a: np.ndarray  # debug: coût linéaire arbitraire
    data: Network

    def __init__(self, data: Network, edge_distances: List[float]):
        """Chargement des données du graphe de Paris dans une matrice
        TODO : améliorer les coûts. Dans une première version, je vais mettre distance de l'arête + x.
        Pour les correspondances, je vais mettre un truc + x, genre 5000 (il faut que ça soit plus gros que la plus grosse arête)
        """
        self.data = data

        # Création de la matrice A
        edges = (
                data["edges"]
                + data["metro_connections"]
                + data["rer_connections"]
                + data["trans_connections"]
        )
        #############################################################################
        #                             SHARED VARIABLES                              #
        #############################################################################

        # Matrix that represents the network (shared between all methods)
        self.A = gen_matrix_A(vertices=len(data["stations"]), edges=edges)

        # COSTS
        # Création du vecteur de coûts (initialisation à 0)
        self.c = np.zeros(len(edges))

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

        # DEBUG : store flows
        self.flows = []

    def compute_edge_costs(self, flow):
        """Calcule les coûts par arête"""
        return self.a * flow + self.b

    def solve(self, log=True):
        """Résout le problème"""

        setup_time = time()
        # debug
        # print(self.c.shape, self.A.shape, self.B.shape)

        # Les flux + calcul du flux initial
        self.flow = linprog(
            self.c,
            A_eq=self.A,
            b_eq=self.B,
            options={
                "rr": False,
            },
        )["x"]
        self.last_flow = np.zeros(self.flow.shape)

        i = 0

        setup_time = time() - setup_time
        loop_time = time()

        while np.sum(np.abs(self.flow - self.last_flow)) > SEUIL_CONVERGENCE:

            step = 1 / (i + 2)

            if log:
                self.flows.append(self.flow)

            # Update the cost for each edge
            costs = self.compute_edge_costs(self.flow)

            self.last_flow = self.flow

            # Solve the linear problem
            gradient = linprog(
                costs,
                A_eq=self.A,
                b_eq=self.B,
                options={
                    "rr": False,
                },
            )["x"]

            # Compute the next flow
            self.flow = (1 - step) * self.last_flow + step * gradient

            i += 1

            error = error_percentage(self.flow, self.last_flow)
            print(
                "Itération n°",
                i,
                "erreur :",
                error,
                "écart",
                np.sum(np.abs(self.flow - self.last_flow)),
            )

        print("convergence après", i, "itérations")

        loop_time = time() - loop_time
        print("TOTAL TIME :", setup_time + loop_time, "s")
        print("Setup took", setup_time, "s")
        print("Loop time took", loop_time, "s")

        # Calcul des erreurs cumulées par rapport à la dernière valeur
        if log:
            np.savetxt("out.csv", self.flows, delimiter=",")

    def solve_paths(self, n: int = 5, couples: List[Tuple[int, int]] = (STATION_DEPART, STATION_ARRIVEE), log=True):
        """Résout le problème avec les n premiers chemins
        """

        # TODO : do this first process in the function that extracts paths
        # TODO : comme on va réutiliser ça pour faire plusieurs points de départ et d'arrivée, il faudrait faire en sorte que ça dépende moins de l'initialisation
        first_n_paths_time = time()  # Start time

        first_n_paths = self.first_paths(n)

        first_n_paths_time = time() - first_n_paths_time

        # DEBUG
        print(first_n_paths.shape)

        # Remove duplicates
        # first_n_paths = np.unique(first_n_paths, axis=0)

        setup_time = time()  # Setup time

        # Extract boolean paths
        # For each path, store its edges as a boolean value : this edge belongs / does not belong
        boolean_paths: np.ndarray = first_n_paths != 0

        #############################################################################
        #                           BUILD LINPROG MATRIX                            #
        #############################################################################
        # The only constraint is that the sum of passengers along all paths is always equal to the initial total number

        self.A = np.ones((1, len(first_n_paths)))  # Sum all passengers for all paths
        self.B = NOMBRE_DE_PASSAGERS  # Total number of passengers

        #############################################################################
        #                             BUILD COST MATRIX                             #
        #############################################################################
        # The cost is computed for each edge as a * flow + b
        # Agregate a & b coefficients for every path
        # An A matrix must be built to account for overlapping edges. The b coefficient is constant and does not need to be adjusted

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

        cost_vector = compute_cost_vector(np.zeros(n))

        # Solve the linear problem for this initial cost
        self.flow = linprog(
            cost_vector,  # Cost vector : minimise the dot product cost_vector @ flow
            A_eq=self.A,
            b_eq=self.B,
            options={
                "rr": False,
            },
        )["x"]

        # Store last flow value
        self.last_flow = np.zeros(self.flow.shape)

        i = 0

        setup_time = time() - setup_time
        loop_time = time()

        while np.sum(np.abs(self.flow - self.last_flow)) > SEUIL_CONVERGENCE:
            step = 1 / (i + 2)

            # Update the cost
            cost_vector = compute_cost_vector(self.flow)

            # Update last flow value
            self.last_flow = self.flow

            # Solve the linear problem
            gradient = linprog(
                cost_vector,
                A_eq=self.A,
                b_eq=self.B,
                options={
                    "rr": False,
                },
            )["x"]

            # Compute the next flow
            self.flow = (1 - step) * self.last_flow + step * gradient

            i += 1

            # DEBUG : print error percentage to see the progress
            error = error_percentage(self.flow, self.last_flow)
            print(
                "Itération n°",
                i,
                "erreur :",
                error,
                "écart",
                np.sum(np.abs(self.flow - self.last_flow)),
            )

        print("convergence après", i, "itérations")

        loop_time = time() - loop_time
        print("TOTAL TIME :", first_n_paths_time + setup_time + loop_time, "s")
        print("First", n, "paths took", first_n_paths_time, "s")
        print("Setup took", setup_time, "s")
        print("Loop time took", loop_time, "s")

        #############################################################################
        #               REBUILD THE LAST FLOW FROM PATHS TO EDGES                   #
        #############################################################################      
        if log:
            # Rebuild flow
            converted_flow = boolean_paths.T @ self.flow

            # Save in in a json file
            write_json("test_last_flow.json", list(converted_flow))

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

        if log:
            # Save paths
            print(n, "paths stored. Saving...")

            # De-numpify for json serialization
            flows = [x.tolist() for x in flows]

            success = write_json(f"first-{n}-paths.json", flows)
            print("Saving done with status", success)

        return np.array(flows)


if __name__ == "__main__":
    # pretty_print_braess([1, 2, 3, 4, 5], True)

    # Braess.braess(Braess.BraessType.WITHOUT_AB)
    # Braess.braess(Braess.BraessType.WITH_AB)
    # Braess.braess(Braess.BraessType.WITH_UPDATED_COSTS)

    graph_data = read_json("paris_network.json")
    edge_distances = read_json("edge_distances.json")

    p = Paris(graph_data, edge_distances)

    # p.solve()
    # p.solve_paths()
    # p.first_paths(n=5, start=0, end=100)
