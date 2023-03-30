"""
TODO : on a un point de départ et un point d'arrivée
Il faut évaluer les chemins non débiles

2 options données par salamitou :
* faire un djikstra et regarder la distance min arrivée / départ (avec les capacités)
Imaginons que certains points sont trop éloignés, alors on les "retire du graphe"
=> mais après, il faut reconstituer les chemins dans le sous graphe sélectionné ! Ça risque d'être compliqué

* faire en fonction des correspondances. 2 remarques :
1) les correspondances sont très coûteuses
2) le réseau de métro est fait de telle sorte qu'on n'ait jamais à changer trop de fois

Alors: il faut retravailler le graphe et tranformer toutes les stations connectées sans correspondances
=> union-find

Une fois qu'on a des clusters de stations, (imaginons : 1 ou 2 correspondances max)
on regarde par quels clusters passer.

Puis : on se résuit à ces clusters pour les chemins


=> REMARQUE : ON N'EST PAS EN TRAIN DE TROUVER DES CHEMINS, MAIS ON RÉDUIT LE GRAPHE!
C'EST DÉJÀ UNE AVANCÉE EN SOI!


Une fois qu'on a les chemins réduits
=> tester, sur les x premières itérations : coup sur coup, ça va faire passer par les chemins estimès optimaux
=> à moins qu'il y ait des chemins qui saturent vite, ça pourrait être très intéressant
C'EST UNE AUTRE MÉTHODE, QU'IL FAUT COMBINER OU NON!

"""
from util.util import read_json, Network, write_json
from typing import List, TypedDict, Callable, Dict
from enum import Enum
import numpy as np
from scipy.optimize import linprog
import warnings
import json
from util.util import Network, gen_matrix_A

################################################################################
#                       CALCUL D"UN TRAJET SUR PARIS                           #
################################################################################
NOMBRE_DE_PASSAGERS = 1000
STATION_DEPART = 0
STATION_ARRIVEE = 100
SEUIL_CONVERGENCE = 5  # La norme 1 de la différence entre 2 flux consécutifs doit être plus petite que ça
# Mettre 50 voire plus pour que l'algo s'arrête plus tôt. Ça dépend aussi du nombre de passagers



def error_percentage(f1: np.ndarray, f2: np.ndarray) -> float:
    """Return the error percentage between two (consecutive) flows, relative to f1.
    Useful for checking the convergence of the algorithm in a relative (not absolute)"""
    error = np.sum(np.abs(f1 - f2))
    return error / np.sum(np.abs(f1))



# TODO : méthode simple : faire quelques itérations, et à chaque itération regarder la marice obtenue pour faire un chemin
# TODO : en extraire un flow affichable

# TODO : pour éviter d'avoir à recopier le code, compartimenter Braess

# XXX : le but c'est de transformer et compartimenter ça
class Paris:

    A: np.ndarray
    B: np.ndarray  # pour un seul couple pour l'instant, tout le reste sera nul
    c: np.ndarray  # coüts

    a: np.ndarray  # debug: coût linéaire arbitraire


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
        self.B[STATION_DEPART] = -NOMBRE_DE_PASSAGERS
        self.B[STATION_ARRIVEE] = +NOMBRE_DE_PASSAGERS

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

        # Les flux + calcul du flux initial
        self.flow = linprog(self.c, A_eq=self.A, b_eq=self.B,
                            options={"rr": False, })['x']
        self.last_flow = np.zeros(self.flow.shape)

        i = 0

        while np.sum(np.abs(self.flow - self.last_flow)) > SEUIL_CONVERGENCE:

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

    def first_paths(self, n: int) -> None:
        """
        Evaluate the first n paths and log them
        """

        # Les flux + calcul du flux initial
        self.flow = linprog(self.c, A_eq=self.A, b_eq=self.B,
                            options={"rr": False, })['x']
        self.last_flow = np.zeros(self.flow.shape)

        for i in range(n):
            print("Step", i+1, "of", n, "...")

            step = 1 / (i + 2)

            # Update the cost for each edge
            costs = self.costs(self.flow)

            self.last_flow = self.flow

            # Solve the linear problem
            gradient = linprog(costs, A_eq=self.A, b_eq=self.B,
                               options={"rr": False, })['x']
        
            # Log this path
            self.flows.append(gradient)

            # Compute the next flow
            self.flow = (1 - step) * self.last_flow + step * gradient

        # Save paths
        print(n, "paths stored. Saving...")

        # De-numpify for json serialization
        flows = [x.tolist() for x in self.flows]

        success = write_json(f"first-{n}-paths.json", flows)
        print("Saving done with status", success)



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

    # Log 5 first paths
    p.first_paths(5)
