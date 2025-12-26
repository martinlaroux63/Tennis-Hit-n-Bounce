#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 19:15:07 2025
@author: martinlaroux
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import glob

# ==============================================================================
# 1. FEATURE ENGINEERING PIPELINE
# ==============================================================================
def extract_features(df_input, window_size=10):
    """
    Transformation des coordonnées brutes en vecteurs de features cinématiques.
    Inclut le lissage, les dérivées et l'enrichissement temporel (Lag/Lead).
    """
    # 1. Nettoyage des données
    df = df_input.copy()
    
    # Conversion type & Interpolation
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Interpolation linéaire : essentielle pour calculer des gradients cohérents sur les frames manquantes
    df['x'] = df['x'].interpolate(method='linear', limit_direction='both')
    df['y'] = df['y'].interpolate(method='linear', limit_direction='both')
    
    # Fallback sur 0 pour les valeurs résiduelles NaN
    df[['x', 'y']] = df[['x', 'y']].fillna(0)

    # 2. Réduction de bruit (Savitzky-Golay)
    # Permet de lisser la trajectoire tout en conservant les points d'inflexion rapides (chocs)
    try:
        x_smooth = savgol_filter(df['x'], window_length=7, polyorder=2)
        y_smooth = savgol_filter(df['y'], window_length=7, polyorder=2)
    except Exception:
        # Fallback pour les séquences trop courtes pour la fenêtre de filtre
        x_smooth = df['x'].values
        y_smooth = df['y'].values
    
    # 3. Calcul des Dérivées Cinématiques
    vx = np.gradient(x_smooth)
    vy = np.gradient(y_smooth)
    ax = np.gradient(vx)
    ay = np.gradient(vy)
    acc_totale = np.sqrt(ax**2 + ay**2) # Norme de l'accélération (Proxy de la force d'impact)
    
    # 4. Features Angulaires
    # L'angle et surtout sa variation sont discriminants pour les rebonds vs trajectoires courbes
    angle = np.arctan2(vy, vx)
    
    # Dérivée de l'angle (Vitesse angulaire)
    angle_change = pd.Series(angle).diff().fillna(0)
    # Normalisation pour gérer la discontinuité trigonométrique (-pi/pi)
    angle_change = angle_change.apply(lambda x: (x + np.pi) % (2 * np.pi) - np.pi)
    
    # 5. Agrégation dans le DataFrame
    features = pd.DataFrame(index=df.index)
    features['vx'] = vx
    features['vy'] = vy
    features['acc_mag'] = acc_totale
    features['ax'] = ax
    features['ay'] = ay
    features['y_pos'] = y_smooth 
    
    # Feature métier : Ratio Vertical (acc_y / acc_total)
    # Utile pour distinguer un rebond (force sol->ciel) d'une frappe à plat
    features['ratio_vertical'] = np.abs(ay) / (acc_totale + 0.0001)

    features['angle'] = angle
    features['angle_change'] = np.abs(angle_change) 

    # 6. Contextualisation Temporelle (Sliding Window)
    # Ajout des états passés (Lag) et futurs (Lead) pour capter la dynamique autour de l'instant t
    cols_to_shift = ['vx', 'vy', 'acc_mag','angle_change']
    
    for shift in range(1, window_size + 1):
        # Passé (t - shift)
        for col in cols_to_shift:
            features[f'{col}_lag_{shift}'] = features[col].shift(shift)
        
        # Futur (t + shift)
        for col in cols_to_shift:
            features[f'{col}_lead_{shift}'] = features[col].shift(-shift)
            
    features = features.fillna(0)
    return features

# ==============================================================================
# 2. PRÉPARATION DU DATASET
# ==============================================================================
def load_training_data(json_files):
    X_list = []
    y_list = []
    
    print(f"Chargement du dataset : {len(json_files)} fichiers sources détectés.")

    for file_path in json_files:
        if not os.path.exists(file_path):
            print(f"    [WARN] Fichier manquant : {file_path}")
            continue
            
        # print(f" - Parsing {file_path}...") # Désactivé pour réduire le bruit dans les logs
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Conversion structurée
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = df.index.astype(int)
        df = df.sort_index()
        
        # Extraction des Features (X)
        feats = extract_features(df)
        
        # Extraction des Labels (y) - 'air' par défaut
        labels = df['action'].fillna('air')
        
        # Stratégie de conservation : Pas de sous-échantillonnage ici.
        # Le déséquilibre de classe sera géré au niveau du modèle (class_weight).
        
        X_list.append(feats)
        y_list.append(labels)
        
    if not X_list:
        raise ValueError("Dataset vide. Vérifiez les chemins.")

    # Concaténation finale
    X_full = pd.concat(X_list)
    y_full = pd.concat(y_list)
    
    return X_full, y_full

# ==============================================================================
# 3. ENTRAÎNEMENT (RANDOM FOREST)
# ==============================================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"PIPELINE D'ENTRAÎNEMENT : DÉTECTION D'ÉVÉNEMENTS")
    print(f"{'='*60}")

    # --- A. CONFIGURATION ---
    # Sélection des données sources
    files_to_train = sorted(glob.glob("ball_data_*.json"))
    
    # --- B. CHARGEMENT ET SPLIT ---
    X, y = load_training_data(files_to_train)
    print(f"\n[INFO] Données chargées.")
    print(f"| Total Frames : {len(X)}")
    print(f"| Distribution des classes :\n{y.value_counts()}")
    
    # Split Train/Test standard (70/30) avec mélange
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    
    # --- C. MODÉLISATION ---
    print("\n Démarrage de l'entraînement (Random Forest)...")
    
    # Utilisation de class_weight='balanced' :
    # Pénalise davantage les erreurs sur les classes minoritaires (Hit/Bounce) 
    # pour compenser la prédominance de la classe 'Air'.
    custom_weight={'hit':10,'bounce':10,'air':1}
    clf = RandomForestClassifier(n_estimators=100, class_weight=custom_weight, random_state=42, n_jobs=-1)
    
    clf.fit(X_train, y_train)
    
    
    # --- D. ÉVALUATION ---
    print(f"\n{'='*20} RAPPORT DE PERFORMANCE {'='*20}")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("\n[Matrice de Confusion (Air / Hit / Bounce)]")
    try:
        print(confusion_matrix(y_test, y_pred, labels=['air', 'hit', 'bounce']))
    except:
        print(confusion_matrix(y_test, y_pred))
    
    # --- E. SÉRIALISATION ---
    output_filename = "trained_model_rf.joblib"
    joblib.dump(clf, output_filename)
    print(f"\n>>> Modèle exporté : {output_filename}")
    print(f"{'='*60}\n")
    
    
    
    
    
    
