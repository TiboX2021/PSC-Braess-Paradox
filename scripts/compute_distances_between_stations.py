"""
Compute all the distances between the stations as a way to weight the graph edges

ATTENTION : il se trouve que les stations Saint-Fargeau et Malesherbes existent sur la ligne 3 et le RER D mais n'ont AUCUN lien.
Par la suite, il faudra indexer les stations avec leur index int et non leur nom.
En attendant, j'ai renommé les stations du RER D Malesherbes2 et Saint-Fargeau2.
"""
from util.util import Network, GPS, read_json, spherical_distance
import json
from numpy import mean, std


def compute_edge_lengths(gps = "paris_gps.json", graph = "paris_network.json"):

    # Open and read data files
    gps_data: GPS = read_json(gps)
    graph_data: Network = read_json(graph)

    stations_names = graph_data["stations"]

    distances = []

    # Go through all edges (but not connections)
    for (start, end) in graph_data["edges"]:

        # Get coordinates for both stations
        lat1, lon1 = gps_data[stations_names[start]]
        lat2, lon2 = gps_data[stations_names[end]]

        distance = spherical_distance(lat1, lon1, lat2, lon2)

        distances.append(distance)
        
    # Save results in temporary file
    f = open("edge_distances.json", "w", encoding="utf-8")
    f.write(json.dumps(distances, indent=4, ensure_ascii=False))
    f.close()

    # Some stats
    print("Longueur totale (m)", sum(distances))
    print("Longueur moyenne", mean(distances))
    print("Écart-type :", std(distances))
    print("Maximum :", max(distances))

    # Detect abnormal cases
    _std = std(distances)
    _mean = mean(distances)

    for distance, (start, end) in zip(distances, graph_data["edges"]):
        if distance == 0:
            print("Null distance between stations", start, f"({stations_names[start]}) and", end, f"({stations_names[end]}) (undetected connection?)")
        elif distance > _mean + 2 * _std:
            print("Abnormally high distance between stations", start, f"({stations_names[start]}) and", end, f"({stations_names[end]}) (wrong edge?)")


if __name__ == "__main__":
    
    compute_edge_lengths()