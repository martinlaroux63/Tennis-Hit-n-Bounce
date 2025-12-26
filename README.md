
# Tennis Tracking Analytics : Détection Hybride d'Événements

Ce projet implémente un moteur d'analyse de données de tracking de tennis (format JSON) capable de détecter, classifier et valider les événements clés du jeu : les frappes (**Hits**) et les rebonds (**Bounces**).

L'architecture repose sur une approche hybride combinant une **analyse cinématique non-supervisée** (basée sur les lois de la physique) et un **modèle de Machine Learning** (Random Forest) pour maximiser la robustesse face aux données bruitées.

---

##  Installation & Démarrage

1. **Prérequis** : Python 3.8+
2. **Installation des dépendances** :
   ```bash
   pip install -r requirements.txt


## Architecture & Méthodologie

Le cœur du projet réside dans la traduction de données de position brutes (x,y) en signatures physiques interprétables.


**1. Moteur Physique & Cinématique (Non-Supervisé)**

Script : main.py

Plutôt que de traiter les données comme une simple série temporelle, j'ai modélisé la balle comme un objet physique soumis à des forces. L'algorithme suit ce pipeline :

  #A. Traitement du Signal (Signal Processing)

Les données brutes de tracking contiennent du bruit et des occlusions.

Interpolation Linéaire : Comblement des frames manquantes pour assurer la continuité mathématique nécessaire aux dérivées.

Filtre Savitzky-Golay : Contrairement à une moyenne mobile qui "écrase" les pics d'intensité, ce filtre (fenêtre de 7, polynôme d'ordre 2) lisse le bruit tout en préservant l'amplitude des impulsions (essentiel pour détecter la force réelle d'un impact).

 #B. Paramètres Physiques & Arbre de Décision

La classification repose sur des descripteurs cinématiques que j'ai conçus pour distinguer les signatures vectorielles :

----Détection des Frappes (Hits) :

Inversion Vectorielle (Produit Scalaire < 0) : Je calcule le produit scalaire entre le vecteur vitesse entrant (t[-3:0]) et sortant (t[1:4]) : une valeur négative indique une rupture de direction > 90° (caractéristique d'un coup de raquette).

Magnitude d'Accélération : Un seuil de force (> 4g) est appliqué pour filtrer les mouvements passifs.

----Détection des Rebonds (Bounces) :

J'ai introduit une notion de Ratio d'accélération vertical (Rv =|ay| / force_choc ) 

Si Rv>0.75, cela signifie que l'accélération est dominée par la composante Y (le sol repousse la balle vers le haut), signature unique d'un rebond.

Stabilité Latérale : Contrairement à une frappe, un rebond conserve la direction du vecteur vitesse en X (pas d'inversion latérale).






**2. Moteur Machine Learning (Supervisé)**

Script : train_model.py et main.py

Pour valider les détections et gérer les cas limites (bruit extrême ou trajectoires complexes), un classifieur Random Forest a été implémenté.

Feature Engineering Avancé : * Création de fenêtres glissantes (Lag/Lead features) pour donner au modèle le contexte temporel (dynamique 10 frames avant et après l'instant T).

Calcul de la variation angulaire pour détecter les cassures de trajectoire subtiles.

Gestion du Déséquilibre (Imbalance) : * Le jeu de tennis est composé à 95% de frames "Air" (rien ne se passe).

Utilisation de class_weight=custom_weight dans le Random Forest pour pénaliser les erreurs sur les classes minoritaires (Hits/Bounces) et forcer le modèle à les apprendre.

 **Visualisation**
 
Le module main.py inclut un moteur de rendu Matplotlib qui superpose les événements détectés à la trajectoire de profondeur (Y) pour validation visuelle :

Lignes Vertes : Frappes (Hits) détectées par rupture de flux.

Lignes Rouges : Rebonds (Bounces) détectés par signature verticale.


## Workflow & Utilisation des Fichiers
Ce projet fonctionne en deux phases distinctes (Entraînement et Inférence) :

1. Entraînement (train_model.py)

Ce script est responsable de la création du "cerveau" du système.

Entrée : Une liste de fichiers JSON annotés (ex: ball_data_1.json, etc.).

Processus : Il extrait les features cinématiques, gère le déséquilibre des classes, entraîne le Random Forest et affiche le rapport de performance (Précision/Rappel).

Sortie : Il génère le fichier binaire trained_model_rf.joblib.

2. Inférence & Analyse (main.py)

Ce script est utilisé pour analyser un nouveau match.

Fonction unsupervised_hit_bounce_detection : Exécute l'analyse purement physique (sans ML) et affiche les graphiques.

Fonction supervized_hit_bounce_detection : Charge le modèle ML pour prédire les événements frame par frame.

Sortie : Visualisation Matplotlib et logs de détection.

## Le fichier .joblib (Modèle Sérialisé)

Le fichier trained_model_rf.joblib contient le classifieur Random Forest entraîné et gelé. Il permet de faire des prédictions sans avoir à ré-entraîner le modèle à chaque fois.

Comment s'en servir dans un autre script ?

Pour utiliser ce modèle, vous devez impérativement lui fournir les mêmes features que celles utilisées lors de l'entraînement (Vitesse, Accélération, Lag, Lead, etc.).

 **Structure du Projet**
 
main.py : Pipeline d'inférence, algorithme physique et visualisation.

train_model.py : Pipeline d'entraînement ML (Extraction de features, Split, Training, Validation).

ball_data_*.json : Données brutes de tracking (Input).

trained_model_rf.joblib : Modèle sérialisé prêt pour la production.
