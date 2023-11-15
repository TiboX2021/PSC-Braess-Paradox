# PSC Paradoxe de Braess

Selfish equilibrium simulation for the Braess paradox on the Paris subway network.

See the result images in [`images/`](./images/).

## Contenu du projet

### Fichiers

#### Résultats des algos

- **out1.csv** : flux successifs sur le graphe de Paris pour des fonctions de coût qui valent _x_
- **out_v2.csv** : flux successifs sur le graphe de Paris pour des fonctions de coût qui valent _distance + x_ avec une distance de 5000 pour les correspondances (les autres arêtes font 1000m en moyenne).

#### Données du graphe

- **paris_gps.json** : graphe complet (RER + Métro) de Paris. Le type des données de ce graphe est _Network_, cf util/util.py
- **edge_distances.json** : liste des longueurs en mètres des arêtes du graphe de paris (seulement _edges_, pas les correspondances), stockées dans une liste. _Calculé avec le script *compute_distances_between_stations.py*, rangé dans \_scripts_.
- **paris_gps.json** : localisations gps de toutes les stations du graphe complet de paris. Le type des donnés est _GPS_, cf util/util.py
- **last_flow.json** : dernier flux de _out_v2.csv_, extrait avec _extract_last_flow.py_ de _scripts_ parce que la lecture des fichiers csv est très longue à cause de leur taille (~100 Mo). C'est ce genre de fichiers qui est utilisé pour afficher le résultat avec matplotlib

#### Scripts utiles

- **affichage_flux.py** : affiche un flux (le dernier flux, par exemple). Il faut changer la variable _FICHIER_DU_FLUX_A_AFFICHER_ au début du code pour afficher un autre flux. Pensez à bien mettre ce fichier dans le même dossier que le script, ou à préciser un chemin complet pour le fichier
- **braess.py** : script qui permet de faire une simulation sur le graphe. Elle dure 30-40 min sur mon pc parce que l'algorithme effectue un grand nombre d'itérations pour beaucoup de précisions. Les flux successifs sont stockés dans un fichier _out.csv_ à la fin de l'exécution. Utiliser le script _extract_last_flow.py_ pour générer un fichier json qui ne contient que le dernier flot afin de l'afficher. Les paramètres à modifier sont indiqués au début du fichier.

### API

Wrappers pour certaines apis dont on a besoin (ex: google pour chercher automatiquement la géolocalisation des stations sur internet). J'utilise ça pour aller chercher des données sur internet

### UTIL

Code utilitaire pour le réutiliser dans d'autres scripts

### SCRIPTS

Scripts de data scraping et bidouillage, que j'ai utilisés pour aller chercher les stations et leur localisation sur internet, par exemple

### ALGOS

Algorithmes

### TESTS

Petits scripts de test pour afficher / analyser les données

- braess.py : aglos de base sur le graphe du paradoxe de Braess
- util.py : fonctions utilitaires pour braess.py, avec du typing pour les données de paris_network.json
- psc_util.py : fonctions utilitaires que je réutilise dans d'autres fichiers, notamment pour faire du datascraping
- nettoyage.py : piste pour enlever les arêtes inutiles d'un graphe (pas fini)
- main.py : code qui fait tourner les algos de nettoyage.py
- code_épuré_pour_présentation.py : code simplifié qui a été inséré dans le rapport intermédiaire
- bidouillage.py : fonctions pour traiter les résultats du fichier out.csv, qui contient les flows de chaque itération de l'algo standard utilise sur un couple départ-arrivée sur le graphe de Paris

- paris_network.json : graphe complet du métro de Paris
- TODO: paris_gps.json : les données de géolocalisation pour chaque station du réseau. Je suis en train de compléter ça

ATTENTION A UTILISER encoding="utf-8" POUR OUVRIR LES FICHIERS

## Cloner le projet

Pour pouvoir copier le projet sur votre pc, vous pouvez le cloner avec git ou télécharger un .zip
