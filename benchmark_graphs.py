"""
Benchmark graphs for the algorithm
Todolist:
- see carnets.binets.fr pour les différents trucs mentionnés pendant la réunion
"""
from franck_wolfe import Paris
from util.util import read_json

if __name__ == "__main__":
    graph_data = read_json("paris_network.json")
    edge_distances = read_json("edge_distances.json")

    p = Paris(graph_data, edge_distances)

    # Exemple de cas d'utilisation. Il faudra faire ça, timer et afficher les graphes résultats
    # TODO : noms à trouver pour les 2 algorithmes
    # 1) algo général/exhaustif/complet
    # 2) algo réduit/rapide/etc
    NOMBRE_DE_PASSAGERS = 1000

    p.solve_paths(n=5, couples=[
        (331, 280, NOMBRE_DE_PASSAGERS),
        (82, 266, NOMBRE_DE_PASSAGERS),
        (312, 294, NOMBRE_DE_PASSAGERS),
        (109, 249, NOMBRE_DE_PASSAGERS),
    ])
