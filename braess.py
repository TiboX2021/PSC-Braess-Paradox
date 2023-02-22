"""
TODO pour le futur:

* trouver un algo pour les chemins non débiles et le tester. Comparer à l'aide de l'algo qui prend beaucoup de temps.
* Analyser les données envoyées par Cohen
* faire un truc pour la représentation matplotlib des données quand on affiche le résultat pour le graphe de Paris. Il faut faire un truc très clean:
visualisation: un plot interactif de matplotlib? Dans ce cas, il faut toutes les données

Convergence basée sur un pourcentage d'erreur? Le donner.

mettre un système pour log les flux sur Paris (pour les analyser derrière)


essayer un pas qui décroît plus vite
"""


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

# Disable warnings
from typing import List, TypedDict, Callable, Dict
from enum import Enum
import numpy as np
from scipy.optimize import linprog
import warnings
import json
from util.util import Network, gen_matrix_A
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', 'Solving system with option')
warnings.filterwarnings('ignore', 'Ill-conditioned matrix')

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
                    [-1,  0, -1,  0, ],
                    [1, -1,  0,  0, ],
                    [0,  0,  1, -1, ],
                    [0,  1,  0,  1, ],
                ]
            ),
            "b": np.array([-4000, 0, 0, +4000]),
            "cost_function": lambda flow: np.array([flow[0]/100, 45, 45, flow[3]/100]),

            "initial_flow": np.array([4000, 4000, 0, 0]),
        },

        BraessType.WITH_AB: {
            "A": np.array(
                [
                    [-1,  0, -1,  0,  0,  0, ],
                    [1, -1,  0, -1,  1,  0, ],
                    [0,  0,  1,  1, -1, -1, ],
                    [0,  1,  0,  0,  0,  1, ],
                ]
            ),
            "b": np.array([-4000, 0, 0, +4000]),
            "cost_function": lambda flow: np.array([flow[0]/100, 45, 45, 0, 0, flow[5]/100]),

            "initial_flow": np.array([4000, 4000, 0, 0, 0, 0]),
        },
        BraessType.WITH_UPDATED_COSTS: {
            "A": np.array(
                [
                    [-1,  0, -1,  0,  0,  0, ],
                    [1, -1,  0, -1,  1,  0, ],
                    [0,  0,  1,  1, -1, -1, ],
                    [0,  1,  0,  0,  0,  1, ],
                ]
            ),
            "b": np.array([-4000, 0, 0, +4000]),
            "cost_function": lambda flow: np.array([flow[0]**2/200, flow[1]*45, flow[2]*45, 0, 0, flow[5]**2/200]),

            "initial_flow": np.array([4000, 4000, 0, 0, 0, 0]),
        }
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
            next_flow = linprog(costs, A_eq=A, b_eq=b,
                                options={"rr": False, })['x']

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
        print("The last two consecutive flows have less than",
              Braess.CONVERGENCE_THRESHOLD, "people of difference in total")
        print()
        Braess.pretty_print_braess(
            values=flow, AB=False if graph_type == Braess.BraessType.WITHOUT_AB else True)
        print()
        print("Total cost :", int(np.round(
            Braess.standard_cost_function(flow))))

    @staticmethod
    def standard_cost_function(flow: np.ndarray) -> float:

        if len(flow) == 4:
            return flow[0]**2/100 + 45 * flow[1] + 45 * flow[2] + flow[3] ** 2/100
        elif len(flow) == 6:
            return flow[0]**2/100 + 45 * flow[1] + 45 * flow[2] + flow[5]**2/100
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

        assert len(
            values) == n_values, f"{n_values} values required, only {len(values)} provided"

        print(
            f"{Braess.embed_number(values[0], size=12)}{Braess.embed_number(values[1], size=11)}\n"
            "  _________A_________  \n"
            " /         |         \ \n"
            f"S          |{Braess.embed_number(abs(values[2] - values[3]), size=10) if AB else ' ' * 10}E\n"
            " \_________B_________/ \n"
            f"{Braess.embed_number(values[2 + AB * 2], size=12)}{Braess.embed_number(values[3 + AB * 2], size=11)}\n"
        )


class Paris:

    A: np.ndarray
    B: np.ndarray  # pour un seul couple pour l'instant, tout le reste sera nul
    c: np.ndarray  # coüts

    a: np.ndarray  # debug: coût linéaire arbitraire

    SEUIL = 5

    def __init__(self, data: Network, edge_distances: List[float]):
        """Chargement des données du graphe de Paris dans une matrice
        TODO : améliorer les coûts. Dans une première version, je vais mettre distance de l'arête + x.
        Pour les correspondances, je vais mettre un truc + x, genre 5000 (il faut que ça soit plus gros que la plus grosse arête)
        """

        # Création de la matrice A
        edges = data["edges"] + data["metro_connections"] + \
            data["rer_connections"] + data["trans_connections"]

        self.A = gen_matrix_A(vertices=len(data["stations"]), edges=edges)

        # Création de la matrice B: on fait aller 1000 personnes d'où à où?
        self.B = np.zeros(len(data["stations"]))

        # Test avec les 2 premières stations
        self.B[0] = -1000
        self.B[100] = +1000

        # Création du vecteur de coûts (initialisation à 0)
        self.c = np.zeros(len(edges))

        # Coefficient multiplié par x pour chaque flux
        self.a = np.ones(len(edges))

        # Coefficient d'ordonnée à l'origine
        self.b = edge_distances + [5000] * (len(edges) - len(edge_distances)) 


        # DEBUG : store flows
        self.flows = []

    def costs(self, flow):
        """Calcule les coûts par arête"""
        return self.a * flow + self.b

    def solve(self, log=True):
        """Résout le problème"""

        # TODO: debug
        print(self.c.shape, self.A.shape, self.B.shape)

        # Les flux + calcul du flux initial
        self.flow = linprog(self.c, A_eq=self.A, b_eq=self.B,
                            options={"rr": False, })['x']
        self.last_flow = np.zeros(self.flow.shape)

        i = 0

        while np.sum(np.abs(self.flow - self.last_flow)) > Paris.SEUIL:

            step = 1 / (i + 2)

            if log:
                self.flows.append(self.flow)

            # Update the cost for each edge
            costs = self.costs(self.flow)

            self.last_flow = self.flow

            # Solve the linear problem
            gradient = linprog(costs, A_eq=self.A, b_eq=self.B,
                               options={"rr": False, })['x']

            # Compute the next flow
            self.flow = (1 - step) * self.last_flow + step * gradient

            i += 1

            error = error_percentage(self.flow, self.last_flow)
            print("Itération n°", i, "erreur :", error, "écart", np.sum(
                np.abs(self.flow - self.last_flow)))

        print("convergence après", i, "itérations")

        # Calcul des erreurs cumulées par rapport à la dernière valeur
        if log:
            np.savetxt("out.csv", self.flows, delimiter=",")


if __name__ == "__main__":

    # pretty_print_braess([1, 2, 3, 4, 5], True)

    # Braess.braess(Braess.BraessType.WITHOUT_AB)
    # Braess.braess(Braess.BraessType.WITH_AB)
    # Braess.braess(Braess.BraessType.WITH_UPDATED_COSTS)

    # Test avec paris
    f = open("paris_network.json")
    graph_data = json.loads(f.read())
    f.close()

    f = open("edge_distances.json")
    edge_distances = json.loads(f.read())
    f.close()

    p = Paris(graph_data, edge_distances)

    p.solve()
