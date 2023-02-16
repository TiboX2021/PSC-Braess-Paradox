"""
Certaines stations ont des coordonnées incohérentes, il faut vérifier et corriger à la main les stations dont les données gps ne seraient pas correctes

latitude (hauteur): 48.3 - 49.27
longitude (largeur): 1.98 - 2.82

Résultats :
Le Mée
Les Noues
Les Saules
Luxembourg

"""
import json
from typing import Dict, Tuple

def get_names_of_eccentric_stations(min_lat: float, max_lat: float, min_lon: float, max_lon: float, datafile = "paris_gps.json"):
    
    f = open(datafile, encoding="utf-8")
    data: Dict[str, Tuple[float, float]] = json.loads(f.read())
    f.close()

    for name, (latitude, longitude) in data.items():
        
        if latitude < min_lat or latitude > max_lat or longitude < min_lon or longitude > max_lon:
            print(name)



if __name__ == "__main__":

    get_names_of_eccentric_stations(
        min_lat=48.3,
        max_lat=49.27,
        min_lon=1.98,
        max_lon=2.82,
    )
