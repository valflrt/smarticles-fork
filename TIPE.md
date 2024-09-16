# Plan présentation

- Présentation de la simulation
  - origine du logiciel
  - Fonctionnement général
    - Classes et forces
  - Améliorations apportées
    - Intégration de Verlet
    - Partition de l'espace
- Présentation de l'algorithme d'entraînement
  - Étapes de recherche
    - Rencontre à l'université avec Mr Imekraz et Mr ??? pour parler du l'algorithme de descente de gradient
  - Principe de l'algorithme génétique

# MCOT

## Simulation de particules et réseau de neurones

_ancrage_ L'intelligence artificielle permet de avancées scientifiques majeures et constitue une transition dans le traitement de problèmes complexes. Les réseaux de neurones notamment permettent de convertir des données brutes en données exploitables et il est souvent nécessaire pour les entraîner de mettre en place une simulation pour permettre un entraînement rapide.

_motivation_ Je suis depuis longtemps fasciné par les phénomènes d’émergence dont un exemple iconique est le Jeu de La Vie de John Conway, j’ai découvert qu’il était possible de d’observer de tels phénomènes dans un simulation de particules. J’ai aussi un grand intérêt pour l’intelligence artificielle. C’est pour cette raison que j’ai décidé d'intégrer ces thèmes dans mon sujet de TIPE qui s'intéresse donc à une simulation de particules et l'entraînement d’un réseau de neurones pour en modifier les paramètres.

### Positionnement thématique:

- **_INFORMATIQUE_** (Informatique Théorique, Informatique pratique)
- **_PHYSIQUE_** (Mécanique)

### Mots clés

| Mots clés en français    | Mots clés en anglais |
| ------------------------ | -------------------- |
| Simulation de particules | Particle Simulation  |
| Forces centrales         | Central Forces       |
| Algorithme génétique     | Genetic Algorithm    |
| Phénomènes d'émergence   | Emergence Phenomenon |
| Intégration de Verlet    | Verlet Integration   |

## DOT

- semaine du 17 juin 2024: idée d'utiliser un réseau de neurones pour controller les paramètres d'une simulation de particules pour effectuer une action sur les particules avec une première approche en tête: utiliser l'algorithme de descente de gradient pour entraîner le réseau de neurones
- jeudi 4 juillet: rencontre à l'université de La Rochelle avec Mr Imekraz, enseignant-chercheur et l'un de ses collègues pour en savoir plus sur l'algorithme de descente de gradient
- semaine du 15 juillet:
  - nouvelle idée: utiliser plutôt un algorithme génétique
  - nouveau but: entraîner le réseau à amener un groupe de particules vers une cible placée aléatoirement dans le plan
- semaine du 19 juillet: tentatives avec l'algorithme génétique
  - réussite de l'algorithme pour des tâches relativement simples: par exemple regrouper les particules
  - recherche d'une fonction evaluation donnant des résultats satisfaisant
- 12 août: amélioration considérable de la vitesse de simulation en n'effectuant seulement les calculs nécessaires (découpage du plan en cellules et application des forces sur les particules des cellules voisines seulement)
- 7 septembre: changement de la façon de disposer les particules à l'instant initial: passage d'une disposition initiale aléatoire à une disposition faisant apparaître les particules en groupe selon leur classe
- 16 septembre: Les particules sont disposées de manière à toujours être dans la direction que l'angle/la direction cible
