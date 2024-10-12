# Plan présentation

- Présentation de la simulation
  - origine du logiciel
  - Fonctionnement général
    - Classes et forces
  - Améliorations apportées
    - multithreading
    - Intégration de Verlet
    - Partitionnement spatial
- Présentation de l'algorithme d'entraînement
  - Étapes de recherche
    - Rencontre à l'université avec Mr Imekraz et Mr ??? pour parler du l'algorithme de descente de gradient
  - Principe de l'algorithme génétique

# MCOT

## Simulation de particules et réseau de neurones

_ancrage:_ L'intelligence artificielle permet de avancées scientifiques majeures et constitue une transition dans le traitement de problèmes complexes. Les réseaux de neurones notamment convertissent des données brutes en données exploitables et une simulation est souvent utilisée pour les entraîner.

_motivation:_ Je suis fasciné par le Jeu de La Vie de Conway. Une simulation de particules peut présenter des caractéristiques similaires. J’ai aussi un grand intérêt pour l’intelligence artificielle. C'est pourquoi mon TIPE traite d'une simulation de particules et de l'entraînement d’un réseau de neurones pour en contrôler les paramètres.

### Positionnement thématique:

- **_INFORMATIQUE_** (Informatique Théorique, Informatique pratique)
- **_PHYSIQUE_** (Mécanique)

### Mots clés

| Mots clés en français    | Mots clés en anglais |
| ------------------------ | -------------------- |
| Simulation de particules | Particle Simulation  |
| Algorithme génétique     | Genetic Algorithm    |
| Intégration de Verlet    | Verlet Integration   |
| Forces centrales         | Central Forces       |
|                          | Spatial Partitioning |

## DOT

- semaine du 17 juin 2024: idée d'utiliser un réseau de neurones pour contrôler les paramètres d'une simulation de particules pour effectuer une action sur les particules avec une première approche en tête: utiliser l'algorithme de descente de gradient pour entraîner le réseau de neurones
- jeudi 4 juillet: rencontre à l'université de La Rochelle avec Mr Imekraz, enseignant-chercheur et l'un de ses collègues pour en savoir plus sur l'algorithme de descente de gradient
- semaine du 15 juillet:
  - nouvelle idée: utiliser plutôt un algorithme génétique
  - nouveau but: entraîner le réseau à amener un groupe de particules vers une cible placée aléatoirement dans le plan
- semaine du 19 juillet: tentatives avec l'algorithme génétique
  - réussite de l'algorithme pour des tâches relativement simples: par exemple regrouper les particules
  - recherche d'une fonction évaluation donnant des résultats satisfaisants
- 12 août: amélioration considérable de la vitesse de simulation en effectuant seulement les calculs nécessaires (découpage du plan en cellules et application des forces sur les particules des cellules voisines seulement)
- 20 août: Modification de l'objectif: entraîner le réseau à faire se déplacer un groupe de particule dans une direction cible (variable au cours du temps)
- 7 septembre: changement de la façon de disposer les particules à l'instant initial: passage d'une disposition initiale aléatoire à une disposition faisant apparaître les particules en groupe selon leur classe
- 16 septembre: Les particules sont disposées de manière à toujours être dans la direction de la direction cible

## Liste des références bibliographiques

| Thème du site         | URL                                                               |
| --------------------- | ----------------------------------------------------------------- |
| Intégration de Verlet | https://femto-physique.fr/analyse-numerique/methode-de-verlet.php |
