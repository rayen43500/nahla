# Système de Détection d'Intrusions Réseau IoT avec Deep Learning

## 📌 Vue d'ensemble

Ce projet développe un système de détection automatique d'intrusions réseau dans des environnements IoT (capteurs, passerelles industrielles) en utilisant des techniques de Deep Learning (LSTM, CNN, Autoencoder, Hybrid).

**Objectif Principal:** Atteindre un F1-score > 90% avec des modèles DL supérieurs aux méthodes ML classiques (Random Forest, SVM).

---

## 🎯 Objectifs du Projet

### Types d'attaques ciblées
- DDoS (Distributed Denial of Service)
- DoS (Denial of Service)
- Scanning/Port scanning
- Malware/Intrusion
- IP Spoofing
- Bottom-up attacks

### Datasets utilisés
1. **CICIDS2017** : Attaques variées (DoS, DDoS, infiltration)
2. **NSL-KDD** : Dataset classique léger, bon pour prototypage
3. **IoT-23** : Spécifique aux environnements IoT

### Technologies
- **Langages:** Python 3.8+
- **Frameworks DL:** PyTorch, TensorFlow/Keras
- **ML classique:** Scikit-learn, XGBoost
- **Visualisation:** Matplotlib, Seaborn
- **API:** FastAPI + Uvicorn
- **Données:** Pandas, NumPy

---

## 📊 État du Projet

### ✅ Composants Existants

| Fichier | Description | État |
|---------|-------------|------|
| `src/data_prep.py` | Split train/val/test, normalisation | 70% |
| `src/models.py` | Architecture MLP simple | 20% |
| `src/train.py` | Boucle entraînement basique | 50% |
| `requirements.txt` | Dépendances principales | ✅ |

### ⏳ À Développer

Voir [TASKS.md](TASKS.md) pour détail complet des 13 phases.

**Résumé:**
- Phase 5 : Collecte datasets (CICIDS2017, NSL-KDD, IoT-23)
- Phase 6 : Prétraitement avancé (SMOTE, PCA, data augmentation)
- Phase 7 : Modèles DL avancés (LSTM, CNN, Autoencoder, Hybrid)
- Phase 8 : Entraînement complet + baselines ML
- Phase 9 : Évaluation (précision, recall, F1, ROC, etc.)
- Phase 10 : Visualisations et analyses
- Phase 11 : Hyperparameter tuning + optimisation
- Phase 12 : API temps réel FastAPI
- Phase 13 : Documentation et tests

---

## 🚀 Quick Start

### 1. Installation

```bash
# Cloner/naviguer vers le projet
cd projet

# Créer virtualenv (optionnel mais recommandé)
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

### 2. Préparer les données

```bash
# À implémenter (Phase 5):
# python src/download_datasets.py
# python src/feature_extraction.py --input data/raw/cicids2017.csv --output data/processed/

# Exemple avec données existantes:
python src/data_prep.py --input data/raw/network_traffic.csv --outdir data/preprocessed/
```

### 3. Entraîner le modèle

```bash
python src/train.py \
  --data-dir data/preprocessed/ \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-3 \
  --hidden 256
```

### 4. Évaluer

```bash
# À implémenter (Phase 9):
# python src/evaluate.py --model models/mlp_best.pt --data data/preprocessed/test.npz
```

### 5. Visualiser résultats

```bash
# À implémenter (Phase 10):
# python src/visualize.py --results results/metrics.json
```

---

## 📂 Structure du Projet

```
projet/
├── README.md                 (ce fichier)
├── TASKS.md                  (plan de tâches détaillé)
├── requirements.txt          (dépendances Python)
│
├── src/                      (code source)
│   ├── data_prep.py         (prétraitement données) ✅
│   ├── data_augmentation.py (à implémenter - Phase 6.2)
│   ├── download_datasets.py (à implémenter - Phase 5.1)
│   ├── feature_extraction.py (à implémenter - Phase 5.2)
│   ├── models.py            (architectures DL) - MLP seulement ⚠️
│   ├── train.py             (entraînement) ⚠️
│   ├── train_baselines.py   (à implémenter - Phase 8.2)
│   ├── evaluate.py          (à implémenter - Phase 9)
│   ├── visualize.py         (à implémenter - Phase 10)
│   ├── api.py               (à implémenter - Phase 12)
│   └── utils.py             (à implémenter)
│
├── data/                     (données - à créer)
│   ├── raw/                 (datasets bruts)
│   ├── processed/           (features extraites)
│   └── preprocessed/        (.npz files)
│
├── models/                   (modèles entraînés - à créer)
│   ├── mlp_best.pt
│   ├── lstm_best.pt
│   ├── cnn_best.pt
│   └── hybrid_best.pt
│
├── results/                  (résultats - à créer)
│   ├── metrics.json
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── report.md
│
└── notebooks/               (exploration - optionnel)
    └── analysis.ipynb
```

---

## 🔄 Pipeline Complet (Une fois terminé)

```
Télécharger Datasets (Phase 5)
        ↓
Extraction Features Réseau (Phase 5.2)
        ↓
Prétraitement & Normalisation (Phase 6)
        ↓
Split Train/Val/Test (Phase 6)
        ↓
┌─────────────────────────────┐
├─ Entraîner DL (Phase 8):   ≥
│  - MLP                       ├→ Évaluation (Phase 9)
│  - LSTM                      ├→ Visualisation (Phase 10)
│  - CNN                       ├→ Optimisation (Phase 11)
│  - Hybrid CNN-LSTM          └→ API (Phase 12)
│                             │
├─ Entraîner ML (Phase 8.2):  ├
│  - Random Forest             ├
│  - SVM                       ├
│  - XGBoost                   ≥
└─────────────────────────────┘
        ↓
Rapport Comparatif (Phase 13)
        ↓
Documentation & Tests (Phase 13)
```

---

## 📊 Métriques et Critères de Succès

### Métriques Principales
- **F1-score global:** Cible > 90%
- **Recall (TPR):** Cible > 85% (minimiser attaques non détectées)
- **FPR:** Cible < 5% (minimiser fausses alarmes)
- **Précision:** Cible > 85%
- **Accuracy:** Cible > 90%

### Par Type d'Attaque
Chacun des types d'attaques doit avoir:
- Recall > 80%
- FPR < 10%

### Performance Zero-Day
- Detection accuracy sur attaques non vues: > 70%

### Comparaison DL vs ML
- F1 DL doit être +15% minimum par rapport ML classique

---

## 🛠️ Configuration Requise

### Matériel Recommandé
- **GPU:** NVIDIA (CUDA 11.8+) - idéal pour entraînement
- **RAM:** 8 GB minimum, 16+ GB recommandé
- **Stockage:** 50 GB+ (datasets complets)

### Logiciels
- Python 3.8 - 3.11
- CUDA Toolkit (optionnel, pour GPU)
- cuDNN (optionnel, pour GPU)

---

## 📝 Utilisation

### Exemple 1: Préparation des données

```python
from src.data_prep import split_data, build_preprocessor, transform_and_save
import pandas as pd

df = pd.read_csv('data.csv')
train, val, test = split_data(df, label_col='label')
preprocessor = build_preprocessor(train, label_col='label')
preprocessor.fit(train.drop(columns=['label']))
transform_and_save(preprocessor, train, 'label', 'data/train.npz')
```

### Exemple 2: Entraîner MLP

```bash
python src/train.py \
  --data-dir data/preprocessed/ \
  --epochs 50 \
  --batch-size 64 \
  --lr 5e-4 \
  --hidden 512 \
  --dropout 0.4
```

### Exemple 3: API temps réel (Phase 12)

```bash
python -m uvicorn src.api:app --reload --port 8000
```

```python
# Client
import requests
response = requests.post('http://localhost:8000/predict', json={
    "flow": [192.168.1.100, 10.0.0.5, 80, 443, "TCP", ...]
})
print(response.json())  # {"prediction": "Normal", "confidence": 0.95}
```

---

## 📚 Références et Ressources

### Datasets
- **CICIDS2017:** https://www.unb.ca/cic/datasets/ids-2017.html
- **NSL-KDD:** https://www.unb.ca/cic/datasets/nsl-kdd.html
- **IoT-23:** https://www.unb.ca/cic/datasets/iot-dataset.html

### Papers Pertinents
- LSTM pour détection anomalies réseau
- CNN pour feature learning flux réseau
- Autoencoders pour zero-day detection
- Comparaison DL vs ML pour IDS

### Tutorials
- PyTorch LSTM: https://pytorch.org/tutorials/
- FastAPI: https://fastapi.tiangolo.com/
- Scikit-learn IDS: https://scikit-learn.org/

---

## 🤝 Contribution

Pour ajouter une fonctionnalité:
1. Créer une branche `feature/name`
2. Implémenter selon les spécifications Phase X
3. Tester et documenter
4. Mettre à jour TASKS.md

---

## ⚠️ Notes Importantes

1. **Data Leakage:** Toujours fit préprocessor sur train set uniquement
2. **Stratification:** Utiliser stratified split vu déséquilibre des classes
3. **GPU:** Vérifier CUDA disponible avec `torch.cuda.is_available()`
4. **Temps:** Entraînement LSTM/CNN peut prendre heures (GPU recommandé)
5. **Datasets:** Vérifier droits d'accès et licences avant téléchargement

---

## 📞 Support

Pour questions ou issues:
1. Consulter TASKS.md pour contexte de la phase
2. Vérifier requirements.txt pour dépendances
3. Lancer tests unitaires (Phase 13.3)

---

## 📄 Licence

À définir

---

**Dernière mise à jour:** 25 février 2026  
**Responsable:** Projet Machine Learning IoT Security  
**Status:** En cours de développement - Phase 5





python -m uvicorn src.api:app --reload --port 8000

python src/train.py 
 python src/data_prep.py --input data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --outdir data/preprocessed --label Label --smote

 Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

 .\.venv\Scripts\Activate.ps1
python src\data_prep.py --input "data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --outdir data/preprocessed --label Label --smote --smote-k-neighbors 3 --merge-webattacks --min-samples-per-class 50

python src\train.py --data-dir data/preprocessed --epochs 20 --batch-size 256