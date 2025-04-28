# projet ELE6209A
projet SLAM pour le cours ELE6209A

Ce projet implémente un algorithme EKF-SLAM (Extended Kalman Filter - Simultaneous Localization and Mapping) en Python, appliqué à un jeu de données de robot mobile contenant des informations GPS, odométriques et LiDAR.

Deux méthodes d'exécution sont proposées :

* Mode Visualisation : pour exécuter SLAM en ligne et tracer la carte avec les incertitudes.
* Mode Analyse de Performance : pour évaluer la consistance et la précision de l'algorithme.

## Prérequis

Python 3.9+

Bibliothèques :
* numpy
* matplotlib
* scipy
* tqdm
* seaborn (pour la visualisation avancée)
* pandas

Installation rapide:

```bash
pip install -r requirements.txt
```

## Exécution

### 1. Mode Visualisation Simple (main.py)

Permet de :

* Lancer l'algorithme EKF-SLAM,
* Suivre la trajectoire du robot,
* Visualiser la carte estimée et les ellipses d'incertitude sur les landmarks.
Exemple minimal pour lancer :
```bash
python main.py
```
Fonctionnement :

* Chargement des fichiers de données GPS, DRS (odométrie) et LiDAR.
* Exécution de l'algorithme SLAM.
* Affichage de la trajectoire du robot et des landmarks.
* Sauvegarde de la carte (foo.png) et de la matrice de covariance (cov.png).


### 2. Mode Analyse Complète (ekf_slam.py)
Permet de :

* Evaluer la consistance (NIS Test),
* Analyser la performance de position,
* Suivre l’évolution de la covariance des landmarks,
* Mesurer les temps de calcul par étape.
Exemple de lancement :

```bash
python EKF_Slam.py
```

Fonctionnement :

* Production de figures de performance :
  * Carte finale avec incertitudes
  * Evolution de la précision
  * Test du NIS
  * Covariances spécifiques de certains landmarks
  * Courbes de temps de calcul