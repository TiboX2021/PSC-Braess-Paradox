"""
Compute all the distances between the stations as a way to weight the graph edges

Output

Longueur totale (m) 1941297.8404170615
Longueur moyenne 1553.0382723336488
Écart-type : 3370.3873391757434
Maximum : 61019.42025202223
Abnormally high distance between stations 602 (Le Coudray-Montceaux) and 603 (Saint-Fargeau) (wrong edge?)
Abnormally high distance between stations 603 (Saint-Fargeau) and 602 (Le Coudray-Montceaux) (wrong edge?)
Abnormally high distance between stations 603 (Saint-Fargeau) and 604 (Ponthierry - Pringy) (wrong edge?)
Abnormally high distance between stations 604 (Ponthierry - Pringy) and 603 (Saint-Fargeau) (wrong edge?)

les 4 précédents sont normaux normalement


Abnormally high distance between stations 623 (Boigneville) and 624 (Malesherbes) (wrong edge?)
Abnormally high distance between stations 624 (Malesherbes) and 623 (Boigneville) (wrong edge?)

Ces 2 là sont normales. Il va falloir finir le reste à la main

TODO: finir les corrections à la main. En général ce sont des coordonnées gps mauvaises qui ont été prises par le script automatique
ATTENTION : ON DIRAIT QU'IL Y A DES CONFUSIONS ENTRE LES NOMS DE STATIONS. Il est possible que 2 stations aient des noms identiques?
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