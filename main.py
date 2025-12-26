#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 11:50:07 2025

@author: martinlaroux
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import joblib


def unsupervised_hit_bounce_detection(json_file_path):
    """
    Pipeline de détection non-supervisée basée sur la cinématique.
    
    Approche :
    1. Prétraitement du signal (Interpolation & Lissage Savitzky-Golay).
    2. Calcul des dérivées physiques (Vitesse, Accélération, Jerk).
    3. Arbre de décision heuristique basé sur la signature physique des impacts.
    
    Args:
        json_file_path (str): Chemin du fichier de tracking brut.
        
    Returns:
        tuple: (liste_hits, liste_bounces) - Indices des événements détectés.
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSUS DÉMARRÉ : Analyse Cinématique Non-Supervisée")
    print(f"Fichier cible : {json_file_path}")
    print(f"{'='*60}")
    
    # ---------------------------------------------------------
    # 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
    # ---------------------------------------------------------
    if not os.path.exists(json_file_path):
        print(f"[ERREUR] Le fichier {json_file_path} est introuvable.")
        return [], []

    with open(json_file_path, 'r') as m:
        data = json.load(m)

    # Garantie de l'ordre chronologique des frames pour l'analyse temporelle
    frames = list(data.keys())
    
    # --- Analyse de la qualité du tracking ---
    count_invisible = 0
    total_frames = 0
    for frame in frames:
        if data[frame]['visible'] == False:
            count_invisible += 1
            total_frames += 1
        else:
            total_frames += 1
    
    ratio = count_invisible/total_frames
    print(f"[*] Qualité du signal : {100-ratio*100:.2f}% de frames visibles.")
    
    # --- Standardisation des données ---
    X = []
    Y = []
    for frame in frames:
        if data[frame]['visible']:
            X.append(data[frame]['x'])
            Y.append(data[frame]['y'])
        else:
            # Injection de NaN pour permettre une interpolation linéaire ultérieure
            X.append(np.nan)
            Y.append(np.nan)

    # ---------------------------------------------------------
    # 2. INTERPOLATION ET LISSAGE 
    # ---------------------------------------------------------
    # Interpolation linéaire pour assurer la continuité des dérivées
    X_propres = pd.Series(X).interpolate(method='linear', limit_direction='both').tolist()
    Y_propres = pd.Series(Y).interpolate(method='linear', limit_direction='both').tolist()

    X_arr = np.array(X_propres)
    Y_arr = np.array(Y_propres)

    # Application d'un filtre Savitzky-Golay :
    # Choix stratégique vs Moyenne mobile : préserve mieux l'amplitude des pics (impacts brefs)
    X_smooth = savgol_filter(X_arr, window_length=7, polyorder=2)
    Y_smooth = savgol_filter(Y_arr, window_length=7, polyorder=2)

    # ---------------------------------------------------------
    # 3. CALCULS PHYSIQUES 
    # ---------------------------------------------------------
    # Vitesse (Dérivée 1ère)
    vx2 = np.gradient(X_smooth)
    vy2 = np.gradient(Y_smooth)

    # Accélération (Dérivée 2nde) - Indicateur clé des chocs
    ax2 = np.gradient(vx2)
    ay2 = np.gradient(vy2)

    # Magnitude de l'accélération totale (Norme du vecteur)
    acc_totale2 = np.sqrt(ax2**2 + ay2**2)

    # ---------------------------------------------------------
    # 4. DÉTECTION DES PICS CANDIDATS
    # ---------------------------------------------------------
    # Identification des changements brusques de mouvement via proéminence des pics
    pics_indices, _ = find_peaks(acc_totale2, prominence=1.3)
    print(f"[*] Candidats détectés (pics d'accélération) : {len(pics_indices)}")

    # ---------------------------------------------------------
    # 5. FILTRE DE DÉBUT ET FIN DE JEU
    # ---------------------------------------------------------
    # Exclusion des mouvements parasites avant/après le jeu effectif
    vitesse_mag = np.sqrt(vx2**2 + vy2**2)
    seuil_mouvement = 0.5 
    indices_mouvement = np.where(vitesse_mag > seuil_mouvement)[0]
    
    if len(indices_mouvement) > 0:
        start_game_idx = indices_mouvement[0] + 5
        end_game_idx = indices_mouvement[-1] - 5
    else:
        start_game_idx = 0
        end_game_idx = len(frames)

    # ---------------------------------------------------------
    # 6. CLASSIFICATION (Hit vs Bounce vs Air)
    # ---------------------------------------------------------
    hits = []
    bounces = []
    
    LARGE_WINDOW = 2
    last_event_type = None 
    
    last_hit_frame_idx = -999 
    MIN_FRAMES_ENTRE_HITS = 10 # Cooldown temporel (~0.3s) pour éviter les doubles détections

    for i in pics_indices:
        # A. Filtrage Temporel (Hors zone de jeu active)
        if i < start_game_idx or i > end_game_idx:
            continue
            
        frame_idx_str = frames[i]
        frame_idx_int = int(frame_idx_str)
        
        # Protection contre les effets de bord du tableau
        if i < 4 or i > len(frames) - 5:
            continue

        # B. Analyse Vectorielle (Moyenne lissée sur 3 frames)
        v_in_x = np.mean(vx2[i-3:i])
        v_in_y = np.mean(vy2[i-3:i])
        
        v_out_x = np.mean(vx2[i+1:i+4])
        v_out_y = np.mean(vy2[i+1:i+4])
        
        # Détection de changement de sens
        changement_direction_x = (np.sign(v_in_x) != np.sign(v_out_x))
        
        # C. Produit Scalaire
        # < 0 implique un angle > 90° (choc violent / raquette)
        dot_product = (v_in_x * v_out_x) + (v_in_y * v_out_y)
        
        # D. Analyse Topologique (Trajectoire Y)
        # Vérification des extrema locaux sur l'axe profondeur
        is_local_y_max = (Y_smooth[i] > Y_smooth[i-1]) and (Y_smooth[i] > Y_smooth[i+1])
        is_local_y_min = (Y_smooth[i] < Y_smooth[i-1]) and (Y_smooth[i] < Y_smooth[i+1])
        is_local_extremun = is_local_y_max or is_local_y_min 
        
        # E. Analyse Topologique Globale (Fenêtre élargie)
        y_window = Y_smooth[i - LARGE_WINDOW : i + LARGE_WINDOW + 1]
        valeur_actuelle = Y_smooth[i]
        is_global_max = valeur_actuelle == np.max(y_window)
        is_global_min = valeur_actuelle == np.min(y_window)
        is_extremum_trajectory = is_global_max or is_global_min
        
        # F. Analyse des Forces (Composantes du vecteur accélération)
        force_choc = acc_totale2[i]
        acc_x = abs(ax2[i])
        acc_y = abs(ay2[i])
        
        # Ratio vertical : Discrimine le rebond (choc purement Y) de la frappe (choc X+Y)
        ratio_vertical = acc_y / (force_choc + 0.00001)
        
        
        # --- ARBRE DE DÉCISION ---

        # BRANCHE 1 : DÉTECTION DE FRAPPE (HIT)
        # Critères : Inversion vectorielle violente OU extremum de trajectoire marqué OU changement de direction selon x
        if (dot_product < 0 and force_choc > 4 ) or is_extremum_trajectory or changement_direction_x:
            ecart_frames = frame_idx_int - last_hit_frame_idx
                
            if ecart_frames > MIN_FRAMES_ENTRE_HITS:
                hits.append(frame_idx_str) 
                last_event_type = 'hit'
                last_hit_frame_idx = frame_idx_int 
            else:
                # Suppression des doublons rapprochés
                continue
            

        # BRANCHE 2 : DÉTECTION DE REBOND (BOUNCE)
        else:
            # Critères : Accélération majoritairement verticale & conservation du sens latéral
            is_vertical_shock = ratio_vertical > 0.75
            stable_x_direction = (v_in_x * v_out_x) > 0
            
            if is_vertical_shock and stable_x_direction:
                # Seuil de force relaxé pour les rebonds
                if force_choc > 2.1: 
                    # Logique séquentielle : Impossible d'avoir 2 bounces de suite
                    if last_event_type == 'bounce':
                        continue 
                    else:
                        bounces.append(frame_idx_str)
                        last_event_type = 'bounce'

    # ---------------------------------------------------------
    # 7. VISUALISATION (DEBUG)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(frames, Y_smooth, color='blue', label='Trajectoire Y (Lissée)')

    # Marquage des événements Hits
    label_added_hit = False 
    for frame_hit in hits:
        if not label_added_hit:
            plt.axvline(x=frame_hit, color='green', linestyle='--', linewidth=1.5, label='Hit (Détecté)')
            label_added_hit = True
        else:
            plt.axvline(x=frame_hit, color='green', linestyle='--', linewidth=1.5)

    # Marquage des événements Bounces
    label_added_bounce = False
    for frame_bounce in bounces:
        if not label_added_bounce:
            plt.axvline(x=frame_bounce, color='red', linestyle='--', linewidth=1.5, label='Bounce (Détecté)')
            label_added_bounce = True
        else:
            plt.axvline(x=frame_bounce, color='red', linestyle='--', linewidth=1.5)

    plt.title(f"Analyse Physique : {os.path.basename(json_file_path)}")
    plt.xlabel("Index Frame")
    plt.ylabel("Position Y (px)")
    indices_ticks = np.arange(0, len(frames), 50)
    labels_ticks = [frames[i] for i in indices_ticks]
    plt.xticks(indices_ticks, labels_ticks, fontsize=8)
    plt.legend()
    
    
    plt.show()

    print(f"\n>>> RÉSULTATS ANALYSE PHYSIQUE")
    print(f"| Hits    : {len(hits)}")
    print(f"| Bounces : {len(bounces)}")
    print(f"{'='*60}\n")
    
    return hits, bounces




# =========================================================
# FEATURE EXTRACTION (CONSISTANCE TRAIN/INFERENCE)
# =========================================================
def extract_features_for_prediction(df_input, window_size=10):
    """
    Génération des features pour l'inférence.
    Est strictement identique au pipeline d'entraînement.
    """
    df = df_input.copy()
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['x'] = df['x'].interpolate(method='linear', limit_direction='both')
    df['y'] = df['y'].interpolate(method='linear', limit_direction='both')
    df[['x', 'y']] = df[['x', 'y']].fillna(0)

    try:
        x_smooth = savgol_filter(df['x'], window_length=7, polyorder=2)
        y_smooth = savgol_filter(df['y'], window_length=7, polyorder=2)
    except Exception:
        x_smooth = df['x'].values
        y_smooth = df['y'].values
    
    vx = np.gradient(x_smooth)
    vy = np.gradient(y_smooth)
    ax = np.gradient(vx)
    ay = np.gradient(vy)
    acc_totale = np.sqrt(ax**2 + ay**2)
    
    # Feature engineering angulaire
    angle = np.arctan2(vy, vx)
    angle_change = pd.Series(angle).diff().fillna(0)
    # Normalisation angulaire (-pi, pi)
    angle_change = angle_change.apply(lambda x: (x + np.pi) % (2 * np.pi) - np.pi)
    
    features = pd.DataFrame(index=df.index)
    features['vx'] = vx
    features['vy'] = vy
    features['acc_mag'] = acc_totale
    features['ax'] = ax
    features['ay'] = ay
    features['y_pos'] = y_smooth
    features['ratio_vertical'] = np.abs(ay) / (acc_totale + 0.0001)
    features['angle'] = angle
    features['angle_change'] = np.abs(angle_change)

    # Création des features temporelles (Lag/Lead) pour donner du contexte au modèle
    cols_to_shift = ['vx', 'vy', 'acc_mag','angle_change']
    for shift in range(1, window_size + 1):
        for col in cols_to_shift:
            features[f'{col}_lag_{shift}'] = features[col].shift(shift)
            features[f'{col}_lead_{shift}'] = features[col].shift(-shift)
            
    return features.fillna(0)


# =========================================================
# DÉTECTION SUPERVISÉE (ML + FILTRE PHYSIQUE)
# =========================================================
def supervized_hit_bounce_detection(json_path, model_path='trained_model_rf.joblib'):
    """
    Prédiction Random Forest 
    """
    if not os.path.exists(json_path) or not os.path.exists(model_path):
        return {}

    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = df.index.astype(int)
    df = df.sort_index()

    clf = joblib.load(model_path)
    X_pred = extract_features_for_prediction(df) 

    # Inférence probabiliste
    probas = clf.predict_proba(X_pred)
    classes = list(clf.classes_)
    
    # Mapping sécurisé des indices de classes
    try:
        idx_hit = classes.index('hit')
    except ValueError:
        idx_hit = -1
    try:
        idx_bounce = classes.index('bounce')
    except ValueError:
        idx_bounce = -1
    
    # --- POST-PROCESSING & HYPERPARAMÈTRES ---
    # Seuils ajustés pour maximiser la Précision (éviter les Faux Positifs)
    SEUIL_HIT = 0.30 
    SEUIL_BOUNCE = 0.30
    
    MIN_FRAMES_COOLDOWN = 10
    MIN_FORCE_PHYSIQUE = 0.4 # Filtre pour éliminer le bruit détecté comme "Hit"
    
    acc_mag_series = X_pred['acc_mag']
    last_event_frame = -999
    final_predictions = []
    
    for i, frame_id in enumerate(df.index):
        # Récupération des scores bruts
        score_hit = probas[i][idx_hit] if idx_hit != -1 else 0
        score_bounce = probas[i][idx_bounce] if idx_bounce != -1 else 0
        
        acc_actuelle = acc_mag_series.iloc[i]
        current_pred = 'air'
        
        # 1. Décision basée sur le score ML
        if score_hit > SEUIL_HIT:
            current_pred = 'hit'
        elif score_bounce > SEUIL_BOUNCE:
            current_pred = 'bounce'
            
        # 2. Validation Logique (Filtres métiers)
        if current_pred != 'air':
            # Validation Physique : Y a-t-il vraiment eu un choc ?
            if acc_actuelle < MIN_FORCE_PHYSIQUE:
                current_pred = 'air'
            # Validation Temporelle : Respect du temps de latence physique
            elif (frame_id - last_event_frame) < MIN_FRAMES_COOLDOWN:
                current_pred = 'air'
            else:
                # Événement validé
                last_event_frame = frame_id

        final_predictions.append(current_pred)
    
    # Enrichissement du JSON de sortie
    pred_series = pd.Series(final_predictions, index=df.index)
    for frame_id in data.keys():
        idx_int = int(frame_id)
        if idx_int in pred_series.index:
            data[frame_id]["pred_action"] = pred_series.loc[idx_int]
        else:
            data[frame_id]["pred_action"] = "air"
            
    return data

if __name__ == "__main__":
    print(">>> Système prêt. Fonctions chargées en mémoire.")
    
    
