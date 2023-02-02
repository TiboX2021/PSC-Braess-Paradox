"""
Utility : comme le format du fichier paris_network
"""
from typing import TypedDict, Tuple, List, Dict
import numpy as np


class Network(TypedDict):

    lines: Dict[str, Tuple[int, int]]
    stations: List[str]
    edges: List[Tuple[int, int]]
    metro_connections: List[Tuple[int, int]]
    rer_connections: List[Tuple[int, int]]
    trans_connections: List[Tuple[int, int]]


"""
Conversion d'une liste de sommets & arrêtes en la matrice utilisée pour les algos Franck-Wolfe
"""


def gen_matrix_A(vertices: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Generate matrix A
    Franck-Wolfe algorithm with flux represented as number of people per edge

    vertices: they go from 0 to "vertices - 1" index
    edges: list of oriented edges from one vertice to another
    """

    n_edges = len(edges)

    A = np.zeros((vertices, n_edges), dtype=int)

    # Fill the matrix for each edge, in order
    for i, (start, end) in enumerate(edges):

        A[start, i] = -1
        A[end, i] = +1

    return A
