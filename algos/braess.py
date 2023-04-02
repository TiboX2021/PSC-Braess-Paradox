from enum import Enum
from typing import List, TypedDict, Callable, Dict

import numpy as np
from scipy.optimize import linprog

"""
Showcase pour le paradoxe de braess, avec affichage des résultats et tout

    x/100       45
  _________A_________
 /         |         \\ 
S          | 0.1      E
 \\_________B_________/
    45        X/100


ordre des arêtes


S -> A
A -> E
S -> B
A -> B
B -> A
B -> E

"""


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

    @staticmethod
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


if __name__ == "__main__":
    # pretty_print_braess([1, 2, 3, 4, 5], True)

    # Braess.braess(Braess.BraessType.WITHOUT_AB)
    # Braess.braess(Braess.BraessType.WITH_AB)
    # Braess.braess(Braess.BraessType.WITH_UPDATED_COSTS)
    pass
