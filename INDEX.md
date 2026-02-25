# 🎯 INDEX DU PROJET - Détection d'Intrusions Réseau IoT

## 📚 Guide de Navigation Rapide

```
DÉBUT ICI ──→ README.md (5-10 min) → Vue d'ensemble + Quick Start

        ↓

COMPRENDRE ──→ SUMMARY.md (5 min) → Ce qui est fait, ce qui reste

        ↓

PLANIFIER ──→ TASKS.md (10-15 min) → 13 Phases détaillées

        ↓

SURVEILLER ──→ STATUS.md (5 min) → Progression actuelle, timeline

        ↓

CONFIGURER ──→ config.ini → 30+ paramètres du projet

        ↓

CODER ──→ src/ (voir ci-dessous)
```

---

## 📂 Structure Fichiers

### 📄 DOCUMENTATION (Lisez ces fichiers first!)

| Fichier | Lecture | Contenu |
|---------|---------|---------|
| **README.md** ⭐ | 5 min | Vue d'ensemble + How-to |
| **SUMMARY.md** ⭐ | 5 min | Résumé phase 1 complétée |
| **TASKS.md** ⭐ | 15 min | Plan complet: 13 phases |
| **STATUS.md** | 5 min | Progression détaillée |
| **INDEX.md** | 3 min | CE FICHIER - Navigation |

### 🔧 CONFIGURATION

| Fichier | Type | Contenu |
|---------|------|---------|
| **config.ini** | INI | 30+ paramètres (à adapter) |
| **.gitignore** | Git | Fichiers à ne pas commiter |
| **requirements.txt** | Python | Dépendances Python |

### 💻 CODE SOURCE (src/)

| Fichier | Status | Complétude | Ligne |
|---------|--------|-----------|-------|
| **data_prep.py** | ⚠️ | 70% | 90 |
| **models.py** | ⚠️ | 20% | 15 |
| **train.py** | ⚠️ | 50% | 135 |
| **utils.py** | ✅ | 100% | 280 |

---

## 🚀 Par Où Commencer?

### Option 1: Débutant (Nouveau sur le projet)
```
1. Lire README.md (5 min)
2. Parcourir SUMMARY.md (5 min)
3. Consulter TASKS.md quand besoin (15 min)
4. Voir STATUS.md pour progression (5 min)
Total: ~30 minutes
```

### Option 2: Développeur (Implémenter Phase X)
```
1. Voir Phase X dans TASKS.md
2. Voir STATUS.md pour contexte
3. Consulter config.ini pour paramètres
4. Coder dans src/
5. Mettre à jour TASKS.md et STATUS.md
```

### Option 3: Manager (Suivi du projet)
```
1. Lire SUMMARY.md (résumé rapide)
2. Consulter STATUS.md (progression)
3. Vérifier TASKS.md quand besoin
4. Adapter config.ini si nécessaire
```

---

## 📊 État Actuel en 30 Secondes

```
Project: Détection Intrusions Réseau IoT avec DL
Status: 📍 Phase 1 COMPLÉTÉE (Fondations)
Progress: ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 30%

Next: Phase 5 (Collecte Données) ← COMMENCER ICI
Timeline: 4-5 semaines pour version complète
Target: F1-Score > 90%
```

Voir [STATUS.md](STATUS.md) pour timeline détaillée.

---

## 📝 Fichiers Clés à Modifier par Phase

| Phase | Fichier à Créer/Modifier | Effort |
|-------|--------------------------|--------|
| 5 | `src/download_datasets.py` (CRÉER) | 1 jour |
| 5 | `src/feature_extraction.py` (CRÉER) | 0.5 jour |
| 6 | `src/data_prep.py` (AMÉLIORER) | 1.5 jour |
| 6 | `src/data_augmentation.py` (CRÉER) | 1 jour |
| 7 | `src/models.py` (ENRICHIR) | 3 jours |
| 8 | `src/train.py` (AMÉLIORER) | 2 jours |
| 8 | `src/train_baselines.py` (CRÉER) | 1 jour |
| 9 | `src/evaluate.py` (CRÉER) | 2 jours |
| 10 | `src/visualize.py` (CRÉER) | 2 jours |
| 12 | `src/api.py` (CRÉER) | 2 jours |

---

## 💡 Astuces de Navigation

### Pour Trouver...

**"Comment faire X?"** → Voir README.md  
**"Qu'est-ce qui me manque?"** → Voir TASKS.md  
**"Où en sommes-nous?"** → Voir STATUS.md  
**"Quelle est la config?"** → Voir config.ini  
**"Comment ça marche ensemble?"** → Voir SUMMARY.md  

### Pour Démarrer l'Entraînement

```bash
# 1. Préparer les données (une fois Phase 5 complétée)
python src/data_prep.py --input data/final.csv --outdir data/preprocessed/

# 2. Entraîner MLP basique
python src/train.py --data-dir data/preprocessed/ --epochs 20

# 3. (Bientôt) Entraîner LSTM
python src/train.py --model lstm --data-dir data/preprocessed/

# 4. (Bientôt) Évaluer tous les modèles
python src/evaluate.py --models all --data data/preprocessed/

# 5. (Bientôt) Lancer l'API
uvicorn src.api:app --reload
```

---

## 🎯 Objectifs Clés à Retenir

| Objectif | Cible | Où suivre |
|----------|-------|-----------|
| F1-Score Global | > 90% | config.ini, STATUS.md |
| Recall (Détection) | > 85% | config.ini |
| FPR (Faux positifs) | < 5% | config.ini |
| Zero-Day Detection | > 70% | TASKS.md Phase 9.3 |
| DL vs ML | +15% F1 min | TASKS.md Phase 10.3 |

---

## 📞 FAQ Rapides

**Q: Par où je commence?**  
A: Lire README.md (5 min) puis SUMMARY.md (5 min)

**Q: Qu'est-ce qui est fait?**  
A: Phase 1 = Configuration. Voir STATUS.md pour détail.

**Q: Qu'est-ce qu'il reste à faire?**  
A: 12 phases! Voir TASKS.md pour full scope.

**Q: Combien de temps?**  
A: 4-5 semaines estimées. Voir timeline dans STATUS.md.

**Q: Comment contribuer?**  
A: Voir Phase X dans TASKS.md, implémenter, cocher tâches.

**Q: Où sauvegarder les données?**  
A: `data/raw/`, `data/processed/`, `data/preprocessed/`. Voir config.ini.

**Q: Comment entraîner?**  
A: Phase 5-6 d'abord (données), puis Phase 7-8. Voir README.md.

**Q: GPU nécessaire?**  
A: Non obligatoire, mais recommandé pour LSTM/CNN. Voir config.ini.

---

## 📈 Dépendances Entre Phases

```
Phase 1 (Config)        ✅ COMPLÉTÉE
    ↓
Phase 5 (Données)       ← DÉMARRER ICI
    ↓
Phase 6 (Prétraitement avancé)
    ↓
Phase 7 (Modèles avancés)  ← En parallèle: Phase 8 peut utiliser phase 4
    ↓
Phase 8 (Entraînement complet)
    ↓
Phase 9 (Évaluation)
    ↓
Phase 10 (Visualisation)
    ↓
Phase 11 (Optimisation)
    ↓
Phase 12 (API)
    ↓
Phase 13 (Docs & Tests)
```

---

## 🔍 Checklist du Premier Jour

- [ ] Lire README.md
- [ ] Consulter SUMMARY.md
- [ ] Parcourir TASKS.md (sections Phase 5-6)
- [ ] Vérifier config.ini
- [ ] Consulter STATUS.md pour timeline
- [ ] Préparer environnement (venv+ requirements.txt)
- [ ] Tester code existant (`python src/data_prep.py --help`)

---

## 🎓 Ressources d'Apprentissage

### Deep Learning
- PyTorch Tutorials: https://pytorch.org/tutorials/
- LSTM Guide: [search: PyTorch LSTM RNN sequences]
- CNN pour time-series: [search: CNN 1D time series]

### Détection intrusions
- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl-kdd.html
- IoT-23: https://www.unb.ca/cic/datasets/iot-dataset.html

### Outils
- FastAPI: https://fastapi.tiangolo.com/
- Scikit-learn: https://scikit-learn.org/
- TensorFlow: https://tensorflow.org/

---

## ✨ Prochaines Étapes (Action Items)

**IMMÉDIAT (Aujourd'hui - Phase 5):**
1. Créer `src/download_datasets.py` pour télécharger CICIDS2017, NSL-KDD
2. Créer `src/feature_extraction.py` pour parser les données
3. Générer datasets prétraités

**COURT TERME (2-3 jours - Phase 6):**
1. Enrichir `src/data_prep.py` avec SMOTE, RobustScaler
2. Créer `src/data_augmentation.py` pour augmentation données

**MOYEN TERME (1-2 semaines - Phase 7-8):**
1. Enrichir `src/models.py` (LSTM, CNN, Autoencoder)
2. Améliorer `src/train.py` (learning rate scheduler, logging)
3. Créer `src/train_baselines.py` (Random Forest, SVM, XGBoost)

---

## 📋 Légende Statuts

| Symbole | Signification |
|---------|---------------|
| ✅ | Complété |
| ⚠️ | Partiellement complété |
| ⏳ | À faire |
| 🔴 | Priorité HAUTE |
| 🟠 | Priorité MÉDIUM-HAUTE |
| 🟡 | Priorité MÉDIUM |

---

## 🤝 Collaboration Notes

- Voir TASKS.md pour tâches granulaires (peut être assignées)
- Mettre à jour STATUS.md après chaque phase (% complétude)
- Cocher tâches dans TASKS.md au fur et à mesure
- Adapter config.ini si paramètres changent

---

## 📞 Support / Questions

1. **"Je suis perdu"** → Lire ce fichier + README.md
2. **"Qu'est-ce que je dois faire?"** → Voir TASKS.md Phase X
3. **"Où en sommes-nous?"** → Voir STATUS.md
4. **"Comment configurer?"** → Voir config.ini avec commentaires

---

**Version:** 0.1.0  
**Date:** 25 février 2026  
**Status:** Phase 1 Complétée ✅

**🚀 Prêt à développer! Commencez par [README.md](README.md) →**

---

Généré par: Équipe Développement  
Dernière mise à jour: 25/02/2026
