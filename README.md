# PSC Paradoxe de Braess

## Contenu du projet

* braess.py : aglos de base sur le graphe du paradoxe de Braess
* util.py : fonctions utilitaires pour braess.py, avec du typing pour les données de paris_network.json
* psc_util.py : fonctions utilitaires que je réutilise dans d'autres fichiers, notamment pour faire du datascraping
* nettoyage.py : piste pour enlever les arêtes inutiles d'un graphe (pas fini)
* main.py : code qui fait tourner les algos de nettoyage.py
* code_épuré_pour_présentation.py : code simplifié qui a été inséré dans le rapport intermédiaire
* bidouillage.py : fonctions pour traiter les résultats du fichier out.csv, qui contient les flows de chaque itération de l'algo standard utili'se sur un couple départ-arrivée sur le graphe de Paris

Le reste est en bordel, j'essayerai de mettre un peu d'ordre dans tout ça.