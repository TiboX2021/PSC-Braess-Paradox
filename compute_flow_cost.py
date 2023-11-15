"""
Compute the total cost of a flow

Current method :

Cost for an edge : distance + x
Cost for a connection : 5000 + x
"""
from typing import List

import numpy as np

from util.util import read_json, Network, read_argv

# Add caching to speedup subsequent calls
__b_coeffs = None
__a_coeffs = None


def compute_flow_cost(flow: np.ndarray):
    """Compute the total cost for a flow"""
    global __b_coeffs, __a_coeffs

    # Read network data
    edge_distances: List = read_json("edge_distances.json")
    network: Network = read_json("paris_network.json")

    total_len = len(network["edges"]) + len(network["metro_connections"]) + len(network["rer_connections"]) + len(
        network["trans_connections"])

    if __b_coeffs is None or __a_coeffs is None:
        __b_coeffs = np.array(
            edge_distances + [5000] * (total_len - len(edge_distances)))
        __a_coeffs = np.ones(total_len) * 0.005  # en baissant le scaling ?

    # IMPROVEMENT : only add the b_coeff where there are people? Else this is just a constant cost...
    # En réalité il faudrait écrire ça comme ça !
    total_cost = np.sum(__a_coeffs * flow * flow + __b_coeffs * flow)
    return total_cost


if __name__ == "__main__":
    filename, = read_argv(1)

    # Read network data
    edge_distances: List = read_json("edge_distances.json")
    network: Network = read_json("paris_network.json")

    total_len = len(network["edges"]) + len(network["metro_connections"]) + len(network["rer_connections"]) + len(
        network["trans_connections"])

    b_coeffs = np.array(
        edge_distances + [5000] * (total_len - len(edge_distances)))
    a_coeffs = np.ones(total_len)

    # Read flow data
    flow = np.array(read_json(filename))

    # IMPROVEMENT : only add the b_coeff where there are people? Else this is just a constant cost...
    # En réalité il faudrait écrire ça comme ça !
    total_cost = np.sum(a_coeffs * flow * flow + b_coeffs * flow)

    print("Total cost for this flow :")
    print(total_cost)
