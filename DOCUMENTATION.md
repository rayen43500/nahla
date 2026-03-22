# Documentation Complète 🛡️ Système de Détection d'Intrusions Réseau IoT

## 🎯 Objectif du Projet
Ce projet implémente un système de détection automatique d'intrusions réseau spécifiquement adapté aux environnements et flux IoT (capteurs, passerelles, appareils connectés). Basé sur l'apprentissage profond (Deep Learning), il est capable de :
1. Identifier divers types d'attaques (DDoS, scanning, malware, spoofing).
2. Fonctionner en temps quasi-réel grâce à une API FastAPI.
3. Afficher une robustesse aux nouveaux environnements IoT à travers de la détection d'anomalies de type Zero-Day.

## 🏗️ Architecture et Logique du Pipeline

Le projet est divisé en plusieurs scripts logiques qui forment un pipeline MLOps complet (Machine Learning Operations).

### 1. Collecte et Extraction des Données
- **`download_datasets.py`** : Télécharge automatiquement des datasets publics de référence (CICIDS2017, NSL-KDD, IoT-23).
- **`feature_extraction.py`** : Convertit des flux bruts ou fichiers `.pcap` en caractéristiques tabulaires (IPs, durées, ports, nombre de paquets, flags TCP).

### 2. Prétraitement & Augmentation
- **`data_prep.py`** : 
  - Gère les valeurs manquantes et sépare le jeu de données (70% Train, 15% Val, 15% Test) avec stratification.
  - Encode les colonnes catégorielles (ex. protocole TCP/UDP) et normalise les caractéristiques numériques avec `RobustScaler` (pour limiter l'effet des valeurs extrêmes aberrantes de trafic).
- **`data_augmentation.py`** : Applique de l'augmentation synthétique (ex. SMOTE) pour équilibrer le dataset si une attaque est sous-représentée.

### 3. Modélisation : Approches Implémentées
Pour répondre aux exigences de pointe, `models.py` propose plusieurs architectures performantes :
* **LSTM (Long Short-Term Memory)** : Réseau récurrent bidirectionnel idéal pour capturer les anomalies séquentielles dans les flux temporels.
* **1D CNN (Convolutional Neural Network)** : Modèle exploitant des filtres parallèles (convolution spatiale) pour détecter rapidement les motifs structurels et signatures d'attaques.
* **Hybride CNN-LSTM** : Combine l'extraction spatiale du CNN avec la mémoire séquentielle du LSTM.
* **Autoencoder (Détecteur d'anomalies)** : Modèle non-supervisé entraîné sur du trafic bénin. Les attaques créent des erreurs de reconstruction élevées (pratique pour l'analyse Zero-Day).
* **Baselines Classiques (ML)** : Random Forest, XGBoost et SVM entraînés dans `train_baselines.py` pour valider l'avantage comparatif du Deep Learning.

### 4. Entraînement et Optimisation
- **`train.py`** : Moteur d'entraînement dynamique intégrant l'Early Stopping (arrêt précoce si sur-apprentissage) et l'adaptation du taux d'apprentissage (ReduceLROnPlateau).
- **`optimize.py`** : Script d'ajustement des hyperparamètres (Optuna) et d'export en format ONNX (quantification des modèles pour une inférence plus légère).

### 5. Évaluation et Visualisation
- **`evaluate.py`** : Calcule finement la Précision, le Rappel, le F1-Score (pondéré et macro), les matrices de confusion, et inclut une modélisation Zero-Day en masquant empiriquement des classes.
- **`visualize.py`** : Synthétise les résultats en images PNG (Courbes ROC, PR, matrices de confusion) et crée un rendu final comparant ML classique et Deep Learning.

### 6. API & Déploiement en Temps Réel
- **`api.py`** : Expose un endpoint REST `/predict` via FastAPI. Le script charge le meilleur modèle en RAM et renvoie le statut (bénin ou malveillant) des flux traités en quelques millisecondes.

---

## 🚀 Guide Rapide : Comment Utiliser ce Programme ?

Ce projet est entièrement complet.

### Étape 1 : Activer l'environnement
Ouvrez PowerShell depuis le dossier du projet, et exécutez :
```powershell
.\venv\Scripts\activate
```

### Étape 2 : Préparer de nouvelles données (Facultatif)
Pour préparer un dataset brut :
```powershell
python src/data_prep.py --input data/raw/votre_dataset.csv --outdir data/preprocessed/
```

### Étape 3 : Entraîner un modèle
L'entraînement est automatisé et s'adapte à vos données :
```powershell
# Entraîner un algorithme LSTM
python src/train.py --data-dir data/preprocessed/ --epochs 10 --model-type lstm

# Entraîner un CNN
python src/train.py --data-dir data/preprocessed/ --epochs 10 --model-type cnn

# Entraîner les modèles classiques (Random Forest, SVM, XGBoost)
python src/train_baselines.py --data-dir data/preprocessed/
```

### Étape 4 : Évaluer et Comparer
Pour exécuter l'évaluation complète sur les jeux de test et générer les graphiques de résultat de chaque algorithme :
```powershell
# Exemple pour le LSTM
python src/evaluate.py --data-dir data/preprocessed --model-path models\lstm_best.pt --model-type lstm

# Produire le rapport et les matrices de confusion
python src/visualize.py --results-dir results --models-dir models --output-dir results --comparison
```
*Le tableau comparatif s'inscrira dans le fichier `results/comparison_report.md`.*

### Étape 5 : Lancer le Système de Détection Temps Réel (API)
Pour utiliser le modèle final et détecter les intrusions à la volée, démarrez le serveur :
```powershell
python -m uvicorn src.api:app --reload --port 8000
```
- Rendez-vous sur `http://localhost:8000/docs` via votre navigateur pour tester l'API intégrée visuellement, ou envoyez un POST sur `/predict` avec vos données réseau.
