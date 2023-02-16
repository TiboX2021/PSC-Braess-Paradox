"""
Utility : data format for many of the .json data files.
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

# Typing for paris_gps.json
GPS = Dict[str, Tuple[float, float]]


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

def read_json(filename : str, encoding = "utf-8") -> any:
    """Read data from a json file in one line"""
    from json import loads

    with open(filename, encoding=encoding) as f:
        return loads(f.read())

def rad(angle: float) -> float:
    """Convert a degree angle to radians"""
    from math  import pi
    return angle * pi / 180


def spherical_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distance in meters between two point with (latitude, longitude) coordinates
    This function needs angles in degrees (not radians)
    We take 6371 km as the Earth radius
    """
    from math import acos, sin, cos

    rad_lat1 = rad(lat1)
    rad_lon1 = rad(lon1)
    rad_lat2 = rad(lat2)
    rad_lon2 = rad(lon2)

    return acos( sin(rad_lat1) * sin(rad_lat2) + cos(rad_lat1) * cos(rad_lat2) * cos(rad_lon2 - rad_lon1)) * 6371000