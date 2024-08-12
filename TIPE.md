# TIPE

## Timeline

- semaine du 17 juin 2024: idée d'utiliser un réseau de neurones pour controller les paramètres d'une simulation de particules avec une première approche en tête: utiliser l'algorithme de descente de gradient pour entraîner le réseau de neurones
- jeudi 4 juillet: rencontre à l'université de La Rochelle avec Mr Imekraz, enseignant-chercheur et l'un de ses collègues pour en savoir plus sur l'algorithme de descente de gradient
- semaine du 15 juillet:
  - nouvelle idée: utiliser plutôt un algorithme génétique
  - nouveau but: entraîner le réseau à amener un groupe de particules vers une cible placée aléatoirement dans le plan
- semaine du 19 juillet: tentatives avec l'algorithme génétique
  - réussite de l'algorithme pour des tâches relativement simples: par exemple regrouper les particules
  - recherche d'une fonction evaluation donnant des résultats satisfaisant
- 8 août: augmenter les forces exercées par les particules si celles-ci sont en nombre réduit
- 12 août: amélioration considérable de la vitesse de simulation en n'effectuant seulement les calculs nécessaires (découpage du plan en cellules et application des forces sur les particules des cellules voisines seulement)
