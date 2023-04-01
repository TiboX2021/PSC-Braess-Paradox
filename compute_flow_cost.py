"""
Compute the total cost of a flow

Current method :

Cost for an edge : distance + x
Cost for a connection : 5000 + x
"""
import numpy as np
from typing import List
from util.util import read_json, Network, read_argv
import sys

if __name__ == "__main__":

    filename, = read_argv(1)
   
    # Read network data
    edge_distances: List = read_json("edge_distances.json")
    network : Network = read_json("paris_network.json")

    total_len = len(network["edges"]) + len(network["metro_connections"]) + len(network["rer_connections"]) + len(network["trans_connections"])

    b_coeffs = np.array(edge_distances + [5000] * (total_len - len(edge_distances)))
    a_coeffs = np.ones(total_len)


    # Read flow data
    flow = np.array(read_json(filename))

    total_cost = np.sum(a_coeffs * flow + b_coeffs)

    print("Total cost for this flow :")
    print(total_cost)