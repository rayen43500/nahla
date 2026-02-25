#!/bin/bash
# QUICK START - Détection Intrusions Réseau IoT
# Commandes pour démarrer le projet rapidement

# =============================================================================
# 1. SETUP INITIAL
# =============================================================================

# Créer virtualenv
python -m venv venv

# Activer virtualenv (Windows)
venv\Scripts\activate
# OU (Linux/Mac):
# source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt


# =============================================================================
# 2. DOCUMENTATION (À LIRE AUTO-MINIMUM 30 MIN)
# =============================================================================

# Navigation rapide
echo "📚 LIRE EN PREMIER:"
echo "1. INDEX.md (3 min) - Navigation rapide"
echo "2. README.md (5 min) - Vue d'ensemble"
echo "3. SUMMARY.md (5 min) - Phase 1 complétée"
echo "4. TASKS.md (15 min) - Plan 13 phases"
echo "5. STATUS.md (5 min) - Progression actuellen"


# =============================================================================
# 3. TESTER LE CODE EXISTANT
# =============================================================================

# Voir aide data_prep
python src/data_prep.py --help

# Voir aide train
python src/train.py --help


# =============================================================================
# 4. EXEMPLE: PRÉPARATION DONNÉES (APRÈS PHASE 5)
# =============================================================================

# Créer répertoires
mkdir -p data/raw data/processed data/preprocessed
mkdir -p models results logs

# Préparer les données (nécessite CSV d'entrée)
# Exemple avec données dummy:
# python src/data_prep.py --input data/raw/network_data.csv --outdir data/preprocessed/


# =============================================================================
# 5. EXEMPLE: ENTRAÎNEMENT (APRÈS PHASE 6)
# =============================================================================

# Entraîner MLP basique
# python src/train.py --data-dir data/preprocessed/ --epochs 20 --batch-size 128


# =============================================================================
# 6. VIS-À-VIS DES RÉSULTATS (À VENIR - PHASE 9+)
# =============================================================================

# À implémenter:
# python src/evaluate.py --model models/mlp_best.pt --data data/preprocessed/
# python src/visualize.py --results results/metrics.json


# =============================================================================
# 7. LANCER API (À VENIR - PHASE 12)
# =============================================================================

# À implémenter:
# uvicorn src.api:app --reload --port 8000


# =============================================================================
# CONFIGURATION
# =============================================================================

# Adapter paramètres globaux dans config.ini:
cat config.ini
# OU
nano config.ini  # Éditer si besoin


# =============================================================================
# FICHIERS CLÉS À CONNAÎTRE
# =============================================================================

echo "📁 FICHIERS CLÉS:"
echo ""
echo "📄 Documentation:"
echo "  - INDEX.md (navigation raccourcie)"
echo "  - README.md (guide complet)"
echo "  - TASKS.md (145+ tâches détaillées)"
echo "  - STATUS.md (progression par phase)"
echo "  - SUMMARY.md (résumé phase 1)"
echo ""
echo "🔧 Configuration:"
echo "  - config.ini (30+ paramètres)"
echo "  - requirements.txt (dépendances)"
echo "  - .gitignore (fichiers à ignorer)"
echo ""
echo "💻 Code:"
echo "  - src/data_prep.py (prétraitement 70%)"
echo "  - src/models.py (modèles 20%)"
echo "  - src/train.py (entraînement 50%)"
echo "  - src/utils.py (utilitaires 100%)"


# =============================================================================
# STATISTIQUES DU PROJET
# =============================================================================

echo ""
echo "📊 STATISTIQUES:"
echo "  - Phases: 13 (30% complétées - Phase 1)"
echo "  - Tâches: 145+ (17 complétées)"
echo "  - Fichiers doc: 6 (INDEX, README, TASKS, STATUS, SUMMARY, ce fichier)"
echo "  - Fichiers config: 3 (config.ini, requirements.txt, .gitignore)"
echo "  - Fichiers code: 4 (data_prep.py, models.py, train.py, utils.py)"
echo "  - Total lignes: ~2500 (doc + code)"
echo "  - Timeline estimée: 4-5 semaines"
echo "  - Cible F1-score: > 90%"


# =============================================================================
# PROCHAINES ÉTAPES RECOMMANDÉES
# =============================================================================

echo ""
echo "🚀 COMMENCER PAR:"
echo ""
echo "1. Lire INDEX.md (ce guide de navigation)"
echo "2. Lire README.md (vue d'ensemble)"
echo "3. Lire SUMMARY.md (résumé)"
echo "4. Consulter TASKS.md pour Phase 5 (Données)"
echo "5. Implémenter Phase 5:"
echo "   - download_datasets.py (CICIDS2017, NSL-KDD, IoT-23)"
echo "   - feature_extraction.py (extraire features réseau)"
echo "6. Continuer avec Phase 6+ selon TASKS.md"
echo ""
echo "⏱️  Temps estimé pour Phase 5: 1-2 jours"


# =============================================================================
# NOTES IMPORTANTES
# =============================================================================

echo ""
echo "⚠️  IMPORTANT:"
echo "  - Lire TASKS.md ENTIÈREMENT avant de coder"
echo "  - Phase 5 dépend de CICIDS2017, NSL-KDD, IoT-23"
echo "  - GPU recommandé pour LSTM/CNN (pas obligatoire pour MLP)"
echo "  - Config.ini pour paramètres (ne pas modifier code)"
echo "  - Mettre à jour STATUS.md après chaque phase"
echo "  - Cocher TASKS.md au fur et à mesure"


# =============================================================================
# DÉBOGAGE
# =============================================================================

# Vérifier PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Vérifier GPU disponible
python -c "import torch; print(f'GPU disponible: {torch.cuda.is_available()}')"

# Vérifier scikit-learn
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"

# Vérifier tous imports
python -c "import numpy, pandas, sklearn, torch, matplotlib, seaborn; print('✅ Toutes les dépendances OK')"


# =============================================================================
# COMMANDES UTILES
# =============================================================================

# Lister fichiers du projet
# Windows:
# tree /F
# Linux/Mac:
# find . -type f -not -path './.git/*' | sort

# Taille du repo
# du -sh .

# Nombre de lignes de code
# find src -name "*.py" | xargs wc -l

# Nombre de lignes doc
# find . -name "*.md" | xargs wc -l
