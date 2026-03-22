# 📖 Guide Pratique et Rôles des Fichiers

Ce document répond de manière simple et directe à vos 4 questions principales : comment utiliser vos données, comment lancer le projet, à quoi sert chaque fichier, et comment tester votre modèle.

---

## 1. Pour entrer un Dataset, j'utilise quoi ?

Pour fournir de nouvelles données au système (un nouveau dataset CSV de trafic réseau), vous devez utiliser le script de préparation : **`src/data_prep.py`**.

**Comment faire ?**
1. Placez votre fichier CSV contenant le trafic réseau dans le dossier `data/raw/`.
2. Ouvrez votre terminal (PowerShell) et activez l'environnement : `.\venv\Scripts\activate`
3. Lancez ce script en lui indiquant le chemin de votre fichier CSV :
   ```powershell
   python src/data_prep.py --input data/raw/votre_fichier.csv --outdir data/preprocessed/ --label Label --smote
   ```
> **Explication :** Ce script va lire votre CSV, nettoyer les données (Standardisation, gestion des valeurs manquantes), équilibrer les attaques minoritaires artificiellement (grâce à `--smote`), et créer des matrices NumPy prêtes pour l'entraînement dans le dossier `data/preprocessed/`.

---

## 2. Pour runner (lancer) le projet, j'utilise quoi ?

Le projet complet tourne autour d'une **API Temps Réel** prête à l'emploi. Pour faire tourner le cœur du projet de détection, le script à utiliser est **`src/api.py`** via Uvicorn.

**Comment faire ?**
1. Toujours dans votre environnement virtuel (`.\venv\Scripts\activate`) :
2. Démarrez le serveur avec la commande suivante :
   ```powershell
   python -m uvicorn src.api:app --reload --port 8000
   ```
> **Explication :** Uvicorn va lancer le fichier `api.py`. Cela va charger en mémoire votre meilleur modèle d'Intelligence Artificielle entraîné (ex. `mlp_best.pt` ou `lstm_best.pt`) et ouvrir un serveur local.
> Vous pouvez ensuite ouvrir un navigateur et aller sur `http://localhost:8000/docs` pour interagir avec l'interface et soumettre des flux réseaux en direct.

---

## 3. Quel est le Rôle de chaque Fichier ?

Voici un dictionnaire clair du rôle de chaque fichier de code dans le dossier `src/` :

### 🗃️ Les Données (Data)
- **`download_datasets.py`** : Télécharge automatiquement des datasets connus depuis internet vers `data/raw/`.
- **`feature_extraction.py`** : (Optionnel) Script pour convertir des fichiers de capture réseau bruts (comme les `.pcap` de Wireshark) en un tableau Excel/CSV lisible.
- **`data_prep.py`** : Transforme et nettoie votre fichier CSV brut en données mathématiques compréhensibles par l'IA (Dossier `data/preprocessed/`).
- **`data_augmentation.py`** : Contient la logique d'équilibrage SMOTE (utilisé par `data_prep.py`) pour multiplier numériquement les attaques rares afin que l'I.A. apprenne mieux.

### 🧠 L'Intelligence Artificielle (Modèles)
- **`models.py`** : C'est ici que sont dessinés les cerveaux (Les architectures des algorithmes). Ce fichier contient le code mathématique définissant ce qu'est le CNN, le LSTM, le MLP, ou l'Autoencoder.
- **`train.py`** : Le moteur d'entraînement de Deep Learning. C'est lui qui prend les données préparées, les fait passer dans les modèles de `models.py` des milliers de fois pour qu'ils apprennent, puis sauvegarde le meilleur résultat dans le dossier `models/`.
- **`train_baselines.py`** : Identique à `train.py`, mais entraîne les vieux algorithmes classiques et plus faibles (Random Forest, SVM, XGBoost) pour servir de point de comparaison.
- **`optimize.py`** : Script avancé pour chercher automatiquement les meilleurs paramètres (nombre de neurones, taux d'apprentissage) sans intervention humaine.

### 📊 L'Évaluation (Tests)
- **`evaluate.py`** : Prend un modèle déjà entraîné, lui fait deviner des données qu'il n'a jamais vues, et donne une note sur 100 (Précision, F1-Score).
- **`visualize.py`** : Dessine des graphiques (courbes, tableaux de confusion) basés sur les résultats de `evaluate.py`.

### 🌐 Le Déploiement (Production)
- **`api.py`** : Interface web connectant votre modèle Python au monde extérieur pour recevoir des alertes en temps réel.
- **`utils.py`** : Petites fonctions utilitaires pratiques réutilisées partout ailleurs.

---

## 4. Comment Tester le Modèle ?

Pour tester mathématiquement votre modèle et voir à 100% s'il détecte bien les attaques, c'est l'étape de l'évaluation avec **`src/evaluate.py`**.

**Comment faire ?**
1. Après avoir entraîné un modèle avec `train.py`, assurez-vous qu'il figure dans votre dossier `models/` (ex: `models/mlp_best.pt`).
2. Lancez le script d'évaluation en spécifiant ce modèle :
   ```powershell
   python src/evaluate.py --data-dir data/preprocessed --model-path models\mlp_best.pt --model-type mlp
   ```
> **Explication :** Le programme va lire 15% de vos données qui avaient été mises de côté lors du prétraitement (le fichier `test.npz`). Il va demander au modèle d'analyser ces connexions réseau trompeuses sans connaître la réponse en avance, puis générera les statistiques (F1-Score, Faux Positifs, etc.) sous forme de fichiers dans le dossier `results/`.
3. *(Optionnel)* Vous pouvez ensuite taper `python src/visualize.py --results-dir results --models-dir models --output-dir results --comparison` pour transformer ces statistiques brutes en superbes graphiques couleur et rapports comparatifs.
