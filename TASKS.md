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

- [ ] **6.2** Créer `data_augmentation.py`
  - [ ] Implémentation pour petit dataset
  - [ ] Générateur de données synthétiques pour attaques

- [ ] **6.3** Validation des données
  - [ ] Vérifier pas de data leakage train/test
  - [ ] Distribution features avant/après normalisation
  - [ ] Test stratification par classe

### Phase 7 : Modèles Deep Learning Avancés
**Priorité:** HAUTE  
**Dépendances:** Phase 4  

- [ ] **7.1** Implémenter LSTM (`models.py`)
  - [ ] Architecture LSTM à 2+ couches
  - [ ] Bidirectionnel (BLSTM)
  - [ ] Stateful option pour flux continu
  - [ ] Dropout + BatchNorm
  - [ ] Capacité à traiter séquences de paquets

- [ ] **7.2** Implémenter CNN (`models.py`)
  - [ ] 1D CNN pour détection motifs
  - [ ] Multiple kernel sizes
  - [ ] MaxPooling + GlobalAveragePooling
  - [ ] Convertir flux en matrices 2D

- [ ] **7.3** Implémenter Autoencoder (`models.py`)
  - [ ] Encoder/Decoder symétrique
  - [ ] Détection anomalies (reconstruction error)
  - [ ] Bottleneck ajustable

- [ ] **7.4** Modèle Hybride CNN-LSTM (`models.py`)
  - [ ] CNN pour extraction features spatiales
  - [ ] LSTM pour dépendances temporelles
  - [ ] Fusion features avant classification

- [ ] **7.5** Modèles classiques (baseline)
  - [ ] Random Forest (sklearn)
  - [ ] SVM avec kernel RBF
  - [ ] Gradient Boosting (XGBoost)
  - [ ] Pour comparaison avec DL

### Phase 8 : Entraînement Complet
**Priorité:** HAUTE  
**Dépendances:** Phase 6, 7  

- [ ] **8.1** Améliorer `train.py`
  - [ ] Early stopping (validation loss)
  - [ ] Learning rate scheduling (ReduceLROnPlateau)
  - [ ] Model checkpointing (meilleur modèle)
  - [ ] Logs complètes (tensorboard ou wandb)
  - [ ] Sauvegarde final model (.pt/.pth)

- [ ] **8.2** Créer `train_baselines.py`
  - [ ] Entraîner Random Forest
  - [ ] Entraîner SVM
  - [ ] Entraîner XGBoost
  - [ ] Sauvegarder chaque modèle

- [ ] **8.3** Cross-validation
  - [ ] K-fold pour validation robuste
  - [ ] Stratifiée
  - [ ] Pour chaque modèle

### Phase 9 : Évaluation et Métriques
**Priorité:** HAUTE  
**Dépendances:** Phase 8  

- [ ] **9.1** Créer `evaluate.py`
  - [ ] Calculer: Precision, Recall, F1-score, Accuracy
  - [ ] TPR (True Positive Rate), FPR (False Positive Rate)
  - [ ] Confusion matrix
  - [ ] Classification report par classe
  - [ ] Matrice de confusion normalisée

- [ ] **9.2** Évaluation par type d'attaque
  - [ ] Métriques séparées pour: DoS, DDoS, Infiltration, Scanning, etc.
  - [ ] Identifier attaques les plus difficiles à détecter

- [ ] **9.3** Analyse Zero-Day
  - [ ] Entraîner sur subset d'attaque types
  - [ ] Tester sur types d'attaques jamais vus
  - [ ] Mesurer robustesse à l'inconnu

### Phase 10 : Visualisation et Analyse
**Priorité:** MÉDIUM-HAUTE  
**Dépendances:** Phase 9  

- [ ] **10.1** Créer `visualize.py`
  - [ ] Courbes ROC (One-vs-Rest)
  - [ ] Courbes PR (Precision-Recall)
  - [ ] Confusion matrices heatmaps
  - [ ] Distribution des prédictions par classe

- [ ] **10.2** Analyse entraînement
  - [ ] Courbes loss/accuracy train/val
  - [ ] Learning curves (impact taille dataset)
  - [ ] Feature importance (si applicable)

- [ ] **10.3** Rapport de comparaison
  - [ ] Tableau comparatif DL vs ML classique
  - [ ] Temps d'inférence
  - [ ] Ressources utilisées (GPU/CPU)

### Phase 11 : Optimisation et Tuning
**Priorité:** MÉDIUM  
**Dépendances:** Phase 8, 9  

- [ ] **11.1** Hyperparameter Tuning
  - [ ] Tester nombre couches (2-5)
  - [ ] Tester hidden dimensions (128, 256, 512)
  - [ ] Tester learning rates (1e-4 à 1e-2)
  - [ ] Tester batch sizes (32, 64, 128, 256)
  - [ ] Dropout rates (0.1 à 0.5)
  - [ ] Utiliser Optuna ou GridSearch

- [ ] **11.2** Architecture Search
  - [ ] Tester LSTM vs CNN vs Hybrid
  - [ ] Tester profondeur optimal
  - [ ] Tester attention mechanisms

- [ ] **11.3** Optimisation inférence
  - [ ] Model quantization (8-bit, FP16)
  - [ ] Knowledge distillation (petit modèle)
  - [ ] ONNX export pour déploiement

### Phase 12 : API et Déploiement
**Priorité:** MÉDIUM  
**Dépendances:** Phase 8  

- [ ] **12.1** Créer API FastAPI (`api.py`)
  - [ ] Endpoint /predict (single packet flow)
  - [ ] Endpoint /predict_batch (multiple flows)
  - [ ] Health check endpoint
  - [ ] Streaming pour temps réel
  - [ ] Gestion d'erreurs

- [ ] **12.2** Service temps réel
  - [ ] Intégration avec Wireshark (pcap to features)
  - [ ] Détection streaming
  - [ ] Buffer gestion pour séquences

- [ ] **12.3** Configuration productio
  - [ ] Config file (.yaml)
  - [ ] Logging structuré
  - [ ] Monitoring performances

### Phase 13 : Documentation et Tests
**Priorité:** MÉDIUM  
**Dépendances:** Toutes phases  

- [ ] **13.1** Code documentation
  - [ ] Docstrings Python complètes
  - [ ] Type hints partout
  - [ ] README.md main
  - [ ] API documentation

- [ ] **13.2** Tutoriels
  - [ ] Quick start guide
  - [ ] Comment télécharger datasets
  - [ ] Comment entraîner modèles
  - [ ] Comment utiliser API

- [ ] **13.3** Tests unitaires
  - [ ] Test data_prep.py
  - [ ] Test models.py
  - [ ] Test train.py
  - [ ] Test API endpoints

- [ ] **13.4** Résultats finaux
  - [ ] Rapport complet (PDF/Markdown)
  - [ ] Comparaison méthodes
  - [ ] Conclusions

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

**Dernière mise à jour:** 25 février 2026
