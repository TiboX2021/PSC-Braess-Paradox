"""
Test des premiers algorithmes sur un graphe simple pour voir comment ça marche

Objectif:
* possibilité de déterminer à l'avance un certain nombre de chemins qui peuvent être pris
* algos de résolution de l'équilibre social et égoïste, ainsi que leur temps
Franck-Wolfe vs descente de gradient avec projection
* comment calculer l'équilibre social, l'équilibre égoïste


NOTES POUR PLUS TARD:
on peut rejeter les chemins dont la longueur à vide est k fois plus grande que la longueur à vide
du chemin le plus court, par exemple (en supposant qu'au bout de k fois plus, les gens ne se déplaceront plus)
"""
from scipy.optimize import linprog
import numpy as np

"""
Cas SIMPLE (aucune optimisation préalable du réseau à faire)

Pour chaque arête:
* sommet de départ
* sommet d'arrivée
* coeff a
* coeff b
* la fonction de coût est de la forme ax + b (pour simplifier)

######### LE GRAPHE #########

    _________(B)_________
   /    x           0    \ 
(A)                      (C)
   \_____________________/
              5

#############################


Supposons que 10 personnes veulent voyager de A à C
* équilibre social ?
* équilibre égoïste ?
"""


"""
(j'ai pas fini de réfléchir sur l'algo de Franck Wolf)
(Voir l'algo suivant avec les chemins d'Aymeric, lui fonctionne)

Algorithme de Franck Wolfe tel qu'on l'a décidé
(minimiser la fonction de coût, résoudre un problème linéaire, puis avancer d'une certaine grandeur)

Remarque: hypothèse de non atomicité: on raisonne avec des grandeurs continues,
le flux n'est donc pas forcément constitué de valeurs entières.


Définition des différents éléments:

couts - coût par arête
flux - le flux de passagers

A une itération donnée, on cherche à avoir un flux qui minimise le produit scalaire (couts ; flux)
(ie: somme des flux_par_arête * cout_par_arête)


Le flux est la solution d'un problème linéaire: il doit y avoir une constante point de départ / point d'arrivée

Le problème linéaire A * flux = b représente deux types de contraintes:
* les arêtes forment des chemins existants
(ie: il n'y a pas plus de passagers qui partent que ceux qui arrivent pour chaque station, sauf départ/arrivée)
* le flux donnée fait bien avancer les passagers depuis leur point de départ vers le point d'arrivée.

C'est très difficile à axiomatiser, on peut facilement déterminer A et b tels que le flux est valide et part/arrive aux bons points,
mais rien ne garantit que les couples départ/arrivée sont respectés. Dans l'article donné par Alain, pour parvenir à cette axiomatisation,
on devait utiliser une matrice carrée A de taille sommets*arêtes.

REMARQUE: dans le cas où il n'y a qu'un couple (départ/arrivée), ce n'est pas nécessaire d'ajouter cette contrainte.

Ci dessous une version incomplète où on garantit uniquement la validité des chemins et des points d'arrivée/départ, et où la matrice A
est de taille (sommets, arêtes), donc bien plus petite.
PISTE: regarder le 3ème algo tout en bas, j'essaye de régler ce problème dedans.

Les arêtes du flux :

flux: |A -> B
      |B -> C
      |A -> C

La matrice A, telle que A * flux donne le bilan algébrique (arrivée - départ) pour chaque sommet.
Cette matrice n'est pas nécessairement carrée (1 colonne = 1 sommet, 1 ligne = 1 arête)

A: |-1  0 -1
   |+1 -1  0
   | 0 +1 +1

La matrice b indique, pour chaque sommet, le nombre de passagers qui entrent et sortent.
Dans notre cas simplifié, 10 passagers entrent en A, et sortent en C. On indique donc le résultat du bilan algébrique:

b: | -10
   |   0
   | +10

En effet: A se trouve avec un déficit de 10 passagers, car 10 passagers partent et aucun n'arrive. Le contraire pour C.
"""


def Franck_Wolfe():

    couts_a = np.array([1, 0, 0])
    couts_b = np.array([0, 0, 5])

    # Flux initial: on l'initialise de manière aléatoire
    # Dans notre cas simple, comme 10 personnes vont de A à C,
    # on peut leur faire emprunter par défaut l'arête C
    flux = np.array([0, 0, 10])

    # Matrice A
    A = np.array([
        [-1, 0, -1],
        [+1, -1, 0],
        [0, 1, 1],
    ])

    # Vecteur b:
    b = np.array([-10, 0, +10])

    print("#######################################################################################")
    print("METHODE DE FRANCK WOLFE")
    print("#######################################################################################")

    for i in range(10):

        # Calcul du pas décroissant:
        pas = 1 / (i + 2)

        # Calcul du coût marginal pour chaque arête
        # (la fonction de coût total vaut la somme des intégrales de ces trucs là)
        # np.sum(flux * (a * flux / 2 + b))
        couts = couts_a * flux + couts_b

        # Maintenant que chaque arête a un coût, on calcule un nouvel équilibre à coeffs constants qui dépend de ces coûts
        result = linprog(couts, A_eq=A, b_eq=b)["x"]

        # Déplacement vers le flux suivant
        flux = (1 - pas) * flux + pas * result

        print(flux, " -- COUT TOTAL :", np.dot(flux, couts))


"""
Là, on a fait un truc nécessairement compliqué, puisqu'il était général.
Partons du principe qu'on a une liste de chemins "non débiles"

Avant tout: on sait que nos personnes vont exclusivement aller de A à C
Il nous suffit donc d'énumérer les chemins qui permettent d'aller de A à C
et qui ne sont pas débiles.

chemins existants:
* A -> C
* A -> B -> C

il nous faudra donc ensuite faire tourner un algo d'optimisation linéaire non plus sur toutes les arêtes,
mais sur ces chemins. Ils pourront être encodés dans une grande matrice, à voir.

Cette méthode est plus simplifiée, mais elle sera plus facile à faire tourner dans la mesure où les couples
"origine-destination" y sont hard codés.

ie: la condition d'égalité sera simple: dans notre cas, la somme sur les 2 chemins doit faire 10

flux: vecteur des fréquentations de chaque chemin
cout: un vecteur bien défini qui fait un produit scalaire avec chaque chemin. Ce vecteur de coût dépendra de chaque chemin
(en termes d'optimisation linéaire, ça ne change pas)

Egalité AX = B:

X est à priori le flot des chemins

on doit simplement conserver les arrivées / destinations

"""


"""
MISE EN PLACE
Définition des variables:

Chemins possibles: 2
* A -> C
* A -> B -> C

Arrivée-Destination envisagées: 1 seule
* A vers B

flux: |0
      |0
flux[i] contient le nombre de personnes empruntant le chemin i

Relation d'optimisation linéaire:
A * flux = b

Nombre de personnes suivant chaque couple arrivée / destination (ici, il n'y a qu'un couple)
b: |10

Chaquu des deux chemins concerne le premier couple arrivée / destination
A: |1 1

Ainsi, A * flux donne le vecteur |x avec x le nombre de personnes allant de A à B


Le vecteur de coûts C est un peu plus complexe à définir car il dépend des chemins
Remarque: ici, comme le coût est une fonction affine, on peut le calculer en utilisant des matrices (c'est pas obligatoire)

* chemin 1: le coût total vaut x
* chemin 2: le coût total vaut 5

donc avec le flux:

coût = |1  0|  x  |f1 + |0
       |0  0|     |f2 + |5
    
     = c1 * flux + c2

REMARQUE:
on s'attend à obtenir un résultat du type [5, 5], puisque c'est l'équilibre social.

REMARQUE SUR LE PAS:
la détermination du flux suivant à chaque itération est TRES VIOLENTE, il faut un pas fortement
décroissant pour espérer que ça converge. Un pas en 1 / 2**k fait bien le travail. Dans le cas où le graphe est
assez bien équilibré (le flux est bien réparti entre plusieurs chemins, il n'y a pas clairement un chemin dominant)
"""


def Aymeric():

    # Initialisation: on suppose que tous suivent le chemin numéro 1: A -> C
    # On ne prend en compte que 10 voyageurs
    flux = np.array([10, 0])

    A = np.array([[1, 1]])
    b = np.array([10])

    # Initialisation des coûts en fonction du flux
    c1 = np.array([
        [1, 0],
        [0, 0]
    ])

    c2 = np.array([0, 5])

    # Coût initial
    cout = np.dot(c1, flux) + c2

    print("#######################################################################################")
    print("METHODE DES CHEMINS D'AYMERIC")
    print("#######################################################################################")

    for i in range(10):  # Nombre fixe d'itérations pour pouvoir tester. TODO: arrêter les itérations quand le delta entre itérations consécutives est suffisamment faible

        # Calcul du coût
        cout = np.dot(c1, flux) + c2

        # Résolution du problème d'optimisation
        result = linprog(cout, A_eq=A, b_eq=b)["x"]

        # Calcul du pas optimal (en 1 / 2+i)
        pas = 1 / (i + 2)

        # Déplacement du flux :
        flux = (1 - pas) * flux + pas * result

        # Affichage du résultat
        print(flux)


"""
#####################################################################################################
PISTE D'AMELIORATION POUR LA METHODE DE FRANCK WOLFE CLASSIQUE
#####################################################################################################

On utilise un graphe un peu plus complexe, de sorte à pouvoir avoir 2 couples (départ, arrivée)
vraiment distincts. On forme le graphe de sorte à ce que le 1er algorithme donne une fausse solution
en "découplant" les couples (A, C), (B, D) pour former des chemins plus optimaux, car il n'est pas
assez restrictif.


faire un graphe à croisement, faire franck wolfe
"""


"""
deuxième truc à faire: le graphe du paradoxe de braess: on verra tout de suite si c'est le coût social ou égoïste

il faudra ajuster les fonctions de coût à ce qu'on a dit!!
pb: le truc avec le produit scalaire, ça va limiter?


Exemple du paradoxde de Braess pour wikipédia
    x/100       45
  _________A_________
 /         |         \ 
S          | 0.1      E
 \_________B_________/
    45        X/100

Pour 4000 conducteurs voulant aller de S à E, sans puis avec l'arête A-B.
Les poids correspondent au temps de voyage.

Je vais le faire avec la représentation arêtes, la représentation Aymeric, et différentes fonction de coût


Liste des arêtes

S -> A
A -> E
S -> B
A -> B
B -> A
B -> E


Liste des sommets

S A B E

cet ordre est utilisé pour la création de la matrice A

#################################################################################################################

Resultats : [4000.  129.    0. 3871.    0. 3871.]  -- COUT TOTAL : 315870.89541483147
# Problème: dans le problème linéaire, le fait d'avoir des arêtes à 2 sens perturbe la résolution
# Il faudra voir comment gérer ça? ça risque de poser pas mal de problèmes
# TEST: je met un coût de 0.1 pour éviter les allers-retours


AVEC L'ARETE AB
Après 100 itérations: 
[4000.   40.    0. 3960.    0. 3960.]  -- COUT TOTAL : 319009.8681855296

    4000       40
  _____>___A_____>____
 /         |          \ 
S          | 3960      E
 \____>____B_____>____/
    0         3960


SANS L'ARETE AB
[1980. 1980. 2020. 2020.]  -- COUT TOTAL : 259999.90221832233

    1980       1980
  _____>___A_____>____
 /                    \ 
S                      E
 \____>____B_____>____/
    2020       2020


AVEC FONCTION DE COUT MISE A JOUR
[2990. 1010. 1010. 2600.  620. 2990.]  -- COUT TOTAL ANCIEN : 270026.61957459163

    2990       1010
  _____>___A_____>____
 /         |          \ 
S          | 1980      E
 \____>____B_____>____/
    1010         2990


=> AVEC CES FONCTIONS DE COUT, ON OBSERVE UN PARADOXE DE BRAESS

=> OBJECTIF : CHANGER LA FONCTION DE COUT GLOBALE POUR OBTENIR UN COUT PLUS FAIBLE
# Remarque: il faut ensuite recalculer le cout avec les fonctions précédentes

"""


# Représentation arêtes: les gens veulent aller de 1 vers B
# Remarque: il faut faire une structure pour automatiser tout ça

# AVEC ARETE A-B

def BraessWith():
    from util import gen_matrix_A

    # DEBUG:  comme liste de sommets et d'arrêtes
    vertices = 4  # number of vertices
    edges = [  # oriented edges in order
        (0, 1),
        (1, 3),
        (0, 2),
        (1, 2),
        (2, 1),
        (2, 3),
    ]

    other_A = gen_matrix_A(vertices, edges)

    print("Generated A matrix :")
    print(other_A)

    # Matrice des connexions sommet-arête (dans l'ordre énoncé plus tôt)
    # [[-1  0 -1  0  0  0]
    # [ 1 -1  0 -1  1  0]
    # [ 0  0  1  1 -1 -1]
    # [ 0  1  0  0  0  1]]

    # combinaison linéaire de rows qui s'annule alors que ça ne s'annule pas sur b?
    # L1 + L2 : [0, -1, -1, -1, +1, 0]
    #  + L3 : [0, 0, -1, -1, +1, 1]
    # + L4
    # en gros: L1 + L2 + L3 + L4 s'annule, mais c'est normal. Ca donne pareil pour b_eq.

    # Vecteur B : 4000 départs de S (-4000) vers E (+4000)
    b = np.array([-4000, 0, 0, +4000])

    # Première valeur du flux (pour calculer les coûts)
    # On fait passer tout le monde par S -> A -> E au début
    flux = np.array([4000, 4000, 0, 0, 0, 0])

    # Vecteurs pour le calcul du cout:
    cout_a = np.array([1/100, 0, 0, 0, 0, 1/100])
    cout_b = np.array([0, 45, 45, 0.1, 0.1, 0])

    # Méthode de Franck Wolfe standard
    for i in range(100):

        # Calcul du pas décroissant:
        pas = 1 / (i + 2)

        # Calcul du coût marginal pour chaque arête
        # (la fonction de coût total vaut la somme des intégrales de ces trucs là)
        # np.sum(flux * (a * flux / 2 + b))
        couts = cout_a * flux + cout_b

        # Maintenant que chaque arête a un coût, on calcule un nouvel équilibre à coeffs constants qui dépend de ces coûts
        all_ = linprog(couts, A_eq=other_A, b_eq=b,
                       options={"disp": True, "rr": False, })
        # problème de rr. Il faut vérifier le range de la matrice
        # PROBLEM APPEARS TO BE INFEASIBLE

        result = all_["x"]
        print("success :", all_["success"])
        print("status :", all_["status"])

        # Déplacement vers le flux suivant
        flux = (1 - pas) * flux + pas * result

        print(np.round(flux), " -- COUT TOTAL :", np.dot(flux, couts))

    # Resultats : [4000.  129.    0. 5530. 1659. 3871.]  -- COUT TOTAL : 315483.84502733883
    # Corrigé   : [4000.  129.    0. 3871.    0. 3871.]  -- COUT TOTAL : 315870.89541483147 (30 iter)
    # bis repeti[4000.   40.    0. 3960.    0. 3960.]  -- COUT TOTAL : 319009.8681855296 (100 iter)


# SANS ARETE A-B

def BraessWithout():
    from util import gen_matrix_A

    # DEBUG:  comme liste de sommets et d'arrêtes
    vertices = 4  # number of vertices
    edges = [  # oriented edges in order
        (0, 1),
        (1, 3),
        (0, 2),
        # (1, 2),
        # (2, 1),
        (2, 3),
    ]

    other_A = gen_matrix_A(vertices, edges)

    print("Generated A matrix :")
    print(other_A)

    # Matrice des connexions sommet-arête (dans l'ordre énoncé plus tôt)
    # [[-1  0 -1  0  0  0]
    # [ 1 -1  0 -1  1  0]
    # [ 0  0  1  1 -1 -1]
    # [ 0  1  0  0  0  1]]

    # combinaison linéaire de rows qui s'annule alors que ça ne s'annule pas sur b?
    # L1 + L2 : [0, -1, -1, -1, +1, 0]
    #  + L3 : [0, 0, -1, -1, +1, 1]
    # + L4
    # en gros: L1 + L2 + L3 + L4 s'annule, mais c'est normal. Ca donne pareil pour b_eq.

    # Vecteur B : 4000 départs de S (-4000) vers E (+4000)
    b = np.array([-4000, 0, 0, +4000])

    # Première valeur du flux (pour calculer les coûts)
    # On fait passer tout le monde par S -> A -> E au début
    flux = np.array([4000, 4000, 0, 0])

    # Vecteurs pour le calcul du cout:
    cout_a = np.array([1/100, 0, 0, 1/100])
    cout_b = np.array([0, 45, 45, 0])

    # Premier calcul du coût:
    # cout = np.sum(cout_a * flux + cout_b)

    # Méthode de Franck Wolfe standard
    for i in range(100):

        # Calcul du pas décroissant:
        pas = 1 / (i + 2)

        # Calcul du coût marginal pour chaque arête
        # (la fonction de coût total vaut la somme des intégrales de ces trucs là)
        # np.sum(flux * (a * flux / 2 + b))
        couts = cout_a * flux + cout_b

        # Maintenant que chaque arête a un coût, on calcule un nouvel équilibre à coeffs constants qui dépend de ces coûts
        all_ = linprog(couts, A_eq=other_A, b_eq=b,
                       options={"disp": True, "rr": False, })
        # problème de rr. Il faut vérifier le range de la matrice
        # PROBLEM APPEARS TO BE INFEASIBLE

        result = all_["x"]
        print("success :", all_["success"])
        print("status :", all_["status"])

        # Déplacement vers le flux suivant
        flux = (1 - pas) * flux + pas * result

        print(np.round(flux), " -- COUT TOTAL :", np.dot(flux, couts))

        # 100 iter :


# AVEC ARETE A-B & CHANGEMENT DE LA FONCTION DE COUT POUR OBTENIR UN MEILLEUR EQUILIBRE
"""
Nouvelles fonctions de coût : on intègre les coûts précédents de 0 à x
    x/100       45
  _________A_________
 /         |         \ 
S          | 0.1      E
 \_________B_________/
    45        X/100


    x²/200      45*x
  _________A_________
 /         |         \ 
S          | 0.1*x    E
 \_________B_________/
    45*x      X²/200

S -> A
A -> E
S -> B
A -> B
B -> A
B -> E
"""


def BraessWithUpdated():
    from util import gen_matrix_A

    # DEBUG:  comme liste de sommets et d'arrêtes
    vertices = 4  # number of vertices
    edges = [  # oriented edges in order
        (0, 1),
        (1, 3),
        (0, 2),
        (1, 2),
        (2, 1),
        (2, 3),
    ]

    other_A = gen_matrix_A(vertices, edges)

    print("Generated A matrix :")
    print(other_A)

    # Matrice des connexions sommet-arête (dans l'ordre énoncé plus tôt)
    # [[-1  0 -1  0  0  0]
    # [ 1 -1  0 -1  1  0]
    # [ 0  0  1  1 -1 -1]
    # [ 0  1  0  0  0  1]]

    # combinaison linéaire de rows qui s'annule alors que ça ne s'annule pas sur b?
    # L1 + L2 : [0, -1, -1, -1, +1, 0]
    #  + L3 : [0, 0, -1, -1, +1, 1]
    # + L4
    # en gros: L1 + L2 + L3 + L4 s'annule, mais c'est normal. Ca donne pareil pour b_eq.

    # Vecteur B : 4000 départs de S (-4000) vers E (+4000)
    b = np.array([-4000, 0, 0, +4000])

    # Première valeur du flux (pour calculer les coûts)
    # On fait passer tout le monde par S -> A -> E au début
    flux = np.array([4000, 4000, 0, 0, 0, 0])

    # Vecteurs pour le calcul du cout:
    cout_a = np.array([1/100, 0, 0, 0, 0, 1/100])
    cout_b = np.array([0, 45, 45, 0.1, 0.1, 0])

    new_cout_a = np.array([1/200, 0, 0, 0, 0, 1/200])
    new_cout_b = np.array([0, 45, 45, 0.1, 0.1, 0])
    new_cout_c = np.array([0, 0, 0, 0, 0, 0])  # celui là n'a pas d'utilité

    # Méthode de Franck Wolfe standard
    for i in range(100):

        # Calcul du pas décroissant:
        pas = 1 / (i + 2)

        # Calcul du coût marginal pour chaque arête
        # (la fonction de coût total vaut la somme des intégrales de ces trucs là)
        # np.sum(flux * (a * flux / 2 + b))
        # couts = cout_a * flux + cout_b
        # Nouvelle manière de calculer le cout
        couts = new_cout_a * flux * flux + new_cout_b * flux + new_cout_c

        # Maintenant que chaque arête a un coût, on calcule un nouvel équilibre à coeffs constants qui dépend de ces coûts
        all_ = linprog(couts, A_eq=other_A, b_eq=b,
                       options={"disp": True, "rr": False, })
        # problème de rr. Il faut vérifier le range de la matrice
        # PROBLEM APPEARS TO BE INFEASIBLE

        result = all_["x"]
        print("success :", all_["success"])
        print("status :", all_["status"])

        # Déplacement vers le flux suivant
        flux = (1 - pas) * flux + pas * result

        old_cost = cout_a * flux + cout_b
        print(np.round(flux), " -- COUT TOTAL ANCIEN :", np.dot(flux, old_cost))


if __name__ == "__main__":

    BraessWithout()
