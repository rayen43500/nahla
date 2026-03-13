# Plan de Tâches - Système de Détection d'Intrusions Réseau IoT

## 📊 État du Projet

**Date de création:** 25 février 2026  
**Objectif:** Détection automatique d'intrusions réseau (IoT) avec Deep Learning

---

## ✅ TÂCHES COMPLÉTÉES

### Phase 1 : Configuration de Base
- [x] Initialiser le projet (structure src/)
- [x] Créer requirements.txt avec dépendances principales
  - pandas, numpy, scikit-learn, torch, matplotlib, seaborn
  - FastAPI + uvicorn pour API

### Phase 2 : Prétraitement (Partiel)
- [x] Créer data_prep.py avec:
  - Split train/val/test (70/15/15%) avec stratification
  - Pipeline de prétraitement (StandardScaler, OneHotEncoder)
  - Sauvegarde en format .npz compressé

### Phase 3 : Modélisation (Partiel)
- [x] Implémenter MLP simple (2 couches cachées)
  - Input_dim variable, hidden=256, dropout=0.3
  - Classification multi-classe

### Phase 4 : Entraînement (Partiel)
- [x] Créer train.py avec:
  - Chargement des données prétraitées
  - Boucle d'entraînement de base
  - Évaluation sur validation/test

---

## ⏳ TÂCHES À FAIRE

### Phase 5 : Collecte et Téléchargement de Données
**Priorité:** HAUTE  
**Dépendances:** Aucune  

- [x] **5.1** Créer script `download_datasets.py`
  - [x] Télécharger CICIDS2017 (attaques variées: DoS, DDoS, infiltration)
  - [x] Télécharger NSL-KDD (dataset classique léger)
  - [x] Télécharger IoT-23 (orienté IoT)
  - [x] Gérer versioning des datasets
  
- [x] **5.2** Créer script `feature_extraction.py`
  - [x] Extraire features réseau: IP src/dst, ports, protocoles
  - [x] Calculer taille paquets, flags TCP, durations
  - [x] Normaliser les noms de colonnes entre datasets
  - [x] Sauvegarder CSVs intermédiaires

- [x] **5.3** Documentation des datasets
  - [x] Documenter structure et format de chaque dataset
  - [x] Lister les types d'attaques par dataset
  - [x] Créer guide d'utilisation

### Phase 6 : Approfondissement du Prétraitement
**Priorité:** HAUTE  
**Dépendances:** Phase 5  

- [x] **6.1** Enrichir `data_prep.py`
  - [x] Gérer déséquilibre de classes (SMOTE, class_weights)
  - [x] Détection et gestion des valeurs manquantes avancée
  - [x] Feature scaling robuste (RobustScaler pour outliers)
  - [x] Ajouter PCA/réduction dimensionnalité optionnelle

- [x] **6.2** Créer `data_augmentation.py`
  - [x] Implémentation pour petit dataset
  - [x] Générateur de données synthétiques pour attaques

- [x] **6.3** Validation des données
  - [x] Vérifier pas de data leakage train/test
  - [x] Distribution features avant/après normalisation
  - [x] Test stratification par classe

### Phase 7 : Modèles Deep Learning Avancés
**Priorité:** HAUTE  
**Dépendances:** Phase 4  

- [x] **7.1** Implémenter LSTM (`models.py`)
  - [x] Architecture LSTM à 2+ couches
  - [x] Bidirectionnel (BLSTM)
  - [x] Stateful option pour flux continu
  - [x] Dropout + BatchNorm
  - [x] Capacité à traiter séquences de paquets

- [x] **7.2** Implémenter CNN (`models.py`)
  - [x] 1D CNN pour détection motifs
  - [x] Multiple kernel sizes
  - [x] MaxPooling + GlobalAveragePooling
  - [x] Convertir flux en matrices 2D

- [x] **7.3** Implémenter Autoencoder (`models.py`)
  - [x] Encoder/Decoder symétrique
  - [x] Détection anomalies (reconstruction error)
  - [x] Bottleneck ajustable

- [x] **7.4** Modèle Hybride CNN-LSTM (`models.py`)
  - [x] CNN pour extraction features spatiales
  - [x] LSTM pour dépendances temporelles
  - [x] Fusion features avant classification

- [x] **7.5** Modèles classiques (baseline)
  - [x] Random Forest (sklearn)
  - [x] SVM avec kernel RBF
  - [x] Gradient Boosting (XGBoost)
  - [x] Pour comparaison avec DL

### Phase 8 : Entraînement Complet
**Priorité:** HAUTE  
**Dépendances:** Phase 6, 7  

- [x] **8.1** Améliorer `train.py`
  - [x] Early stopping (validation loss)
  - [x] Learning rate scheduling (ReduceLROnPlateau)
  - [x] Model checkpointing (meilleur modèle)
  - [x] Logs complètes (tensorboard ou wandb)
  - [x] Sauvegarde final model (.pt/.pth)

- [x] **8.2** Créer `train_baselines.py`
  - [x] Entraîner Random Forest
  - [x] Entraîner SVM
  - [x] Entraîner XGBoost
  - [x] Sauvegarder chaque modèle

- [x] **8.3** Cross-validation
  - [x] K-fold pour validation robuste
  - [x] Stratifiée
  - [x] Pour chaque modèle

### Phase 9 : Évaluation et Métriques
**Priorité:** HAUTE  
**Dépendances:** Phase 8  

- [x] **9.1** Créer `evaluate.py`
  - [x] Calculer: Precision, Recall, F1-score, Accuracy
  - [x] TPR (True Positive Rate), FPR (False Positive Rate)
  - [x] Confusion matrix
  - [x] Classification report par classe
  - [x] Matrice de confusion normalisée

- [x] **9.2** Évaluation par type d'attaque
  - [x] Métriques séparées pour: DoS, DDoS, Infiltration, Scanning, etc.
  - [x] Identifier attaques les plus difficiles à détecter

- [x] **9.3** Analyse Zero-Day
  - [x] Entraîner sur subset d'attaque types
  - [x] Tester sur types d'attaques jamais vus
  - [x] Mesurer robustesse à l'inconnu

### Phase 10 : Visualisation et Analyse
**Priorité:** MÉDIUM-HAUTE  
**Dépendances:** Phase 9  

- [x] **10.1** Créer `visualize.py`
  - [x] Courbes ROC (One-vs-Rest)
  - [x] Courbes PR (Precision-Recall)
  - [x] Confusion matrices heatmaps
  - [x] Distribution des prédictions par classe

- [x] **10.2** Analyse entraînement
  - [x] Courbes loss/accuracy train/val
  - [x] Learning curves (impact taille dataset)
  - [x] Feature importance (si applicable)

- [x] **10.3** Rapport de comparaison
  - [x] Tableau comparatif DL vs ML classique
  - [x] Temps d'inférence
  - [x] Ressources utilisées (GPU/CPU)

### Phase 11 : Optimisation et Tuning
**Priorité:** MÉDIUM  
**Dépendances:** Phase 8, 9  

- [x] **11.1** Hyperparameter Tuning
  - [x] Tester nombre couches (2-5)
  - [x] Tester hidden dimensions (128, 256, 512)
  - [x] Tester learning rates (1e-4 à 1e-2)
  - [x] Tester batch sizes (32, 64, 128, 256)
  - [x] Dropout rates (0.1 à 0.5)
  - [x] Utiliser Optuna ou GridSearch

- [x] **11.2** Architecture Search
  - [x] Tester LSTM vs CNN vs Hybrid
  - [x] Tester profondeur optimal
  - [x] Tester attention mechanisms

- [x] **11.3** Optimisation inférence
  - [x] Model quantization (8-bit, FP16)
  - [x] Knowledge distillation (petit modèle)
  - [x] ONNX export pour déploiement

### Phase 12 : API et Déploiement
**Priorité:** MÉDIUM  
**Dépendances:** Phase 8  

- [x] **12.1** Créer API FastAPI (`api.py`)
  - [x] Endpoint /predict (single packet flow)
  - [x] Endpoint /predict_batch (multiple flows)
  - [x] Health check endpoint
  - [x] Streaming pour temps réel
  - [x] Gestion d'erreurs

- [x] **12.2** Service temps réel
  - [x] Intégration avec Wireshark (pcap to features)
  - [x] Détection streaming
  - [x] Buffer gestion pour séquences

- [x] **12.3** Configuration production
  - [x] Config file (.yaml)
  - [x] Logging structuré
  - [x] Monitoring performances

### Phase 13 : Documentation et Tests
**Priorité:** MÉDIUM  
**Dépendances:** Toutes phases  

- [x] **13.1** Code documentation
  - [x] Docstrings Python complètes
  - [x] Type hints partout
  - [x] README.md main
  - [x] API documentation

- [x] **13.2** Tutoriels
  - [x] Quick start guide
  - [x] Comment télécharger datasets
  - [x] Comment entraîner modèles
  - [x] Comment utiliser API

- [x] **13.3** Tests unitaires
  - [x] Test data_prep.py
  - [x] Test models.py
  - [x] Test train.py
  - [x] Test API endpoints

- [x] **13.4** Résultats finaux
  - [x] Rapport complet (PDF/Markdown)
  - [x] Comparaison méthodes
  - [x] Conclusions

---

## 📋 Résumé des Étapes

```
Phase 5  ──────────┐
                   ├──→ Phase 6 ──────────┐
                                          ├──→ Phase 8 ──┐
Phase 7 ────────────────────────────────┤               ├──→ Phase 9 ──→ Phase 10, 11, 12, 13
                                        │               │
                   Phase 4 (partiel) ────┘               │
                                                         │
                                        Phase 8 (baseline)┘
```

---

## 🎯 Métriques de Succès

| Critère | Cible |
|---------|-------|
| F1-score Global | > 90% |
| Recall (détection attaques) | > 85% |
| FPR (faux positifs) | < 5% |
| Performance DL vs ML classique | +20% F1 minimum |
| Temps inférence | < 100ms par flux |
| Zero-day detection | > 70% accuracy |

---

## 📁 Structure Finale Attendue

```
projet/
├── README.md
├── requirements.txt
├── TASKS.md (ce fichier)
├── src/
│   ├── download_datasets.py      [PHASE 5.1]
│   ├── feature_extraction.py     [PHASE 5.2]
│   ├── data_prep.py              [PHASE 6.1]
│   ├── data_augmentation.py      [PHASE 6.2]
│   ├── models.py                 [PHASE 7]
│   ├── train.py                  [PHASE 8.1]
│   ├── train_baselines.py        [PHASE 8.2]
│   ├── evaluate.py               [PHASE 9]
│   ├── visualize.py              [PHASE 10]
│   ├── api.py                    [PHASE 12]
│   └── utils.py
├── data/
│   ├── raw/                      (datasets bruts)
│   ├── processed/                (features extraites)
│   └── preprocessed/             (.npz files)
├── models/
│   ├── mlp_best.pt
│   ├── lstm_best.pt
│   ├── cnn_best.pt
│   ├── hybrid_best.pt
│   └── autoencoder_best.pt
├── results/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── report.md
└── notebooks/
    └── analysis.ipynb             [Pour exploration]
```

---

## 🚀 Ordre d'Exécution Recommandé

1. **Phase 5** : Collecter et préparer les données
2. **Phase 6** : Approfondissement prétraitement
3. **Phase 7** : Implémenter modèles avancés
4. **Phase 8** : Entraîner tous les modèles
5. **Phase 9** : Évaluation complète
6. **Phase 10** : Visualisation et analyse
7. **Phase 11** : Optimisation basée sur résultats
8. **Phase 12** : API et déploiement
9. **Phase 13** : Documentation et tests finaux

---

**Dernière mise à jour:** 9 mars 2026
