# PSC Paradoxe de Braess

## Contenu du projet

### API

Wrappers pour certaines apis dont on a besoin (ex: google pour chercher automatiquement la géolocalisation des stations sur internet)

### UTIL

Code utilitaire pour le réutiliser dans d'autres scripts

### SCRIPTS

Scripts de data scraping, que j'ai utilisés pour aller chercher les stations et leur localisation sur internet, par exemple

### ALGOS

Algorithmes

### TESTS

Petits scripts de test pour afficher / analyser les données


* braess.py : aglos de base sur le graphe du paradoxe de Braess
* util.py : fonctions utilitaires pour braess.py, avec du typing pour les données de paris_network.json
* psc_util.py : fonctions utilitaires que je réutilise dans d'autres fichiers, notamment pour faire du datascraping
* nettoyage.py : piste pour enlever les arêtes inutiles d'un graphe (pas fini)
* main.py : code qui fait tourner les algos de nettoyage.py
* code_épuré_pour_présentation.py : code simplifié qui a été inséré dans le rapport intermédiaire
* bidouillage.py : fonctions pour traiter les résultats du fichier out.csv, qui contient les flows de chaque itération de l'algo standard utilise sur un couple départ-arrivée sur le graphe de Paris

* paris_network.json : graphe complet du métro de Paris
* TODO: paris_gps.json : les données de géolocalisation pour chaque station du réseau. Je suis en train de compléter ça

ATTENTION A UTILISER encoding="utf-8" POUR OUVRIR LES FICHIERS