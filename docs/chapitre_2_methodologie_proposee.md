# Chapitre 2 : Methodologie proposee

## 2.1 Description du probleme

Dans ce travail, le probleme est formule comme une **classification binaire** du trafic reseau :
- Classe 0 : trafic normal
- Classe 1 : trafic d'attaque

L'objectif est de concevoir un systeme capable de distinguer automatiquement les flux legitimes des flux malveillants dans un contexte IoT.

Les donnees reseau IoT presentent des caracteristiques particulieres :
- Heterogeneite des protocoles et des equipements
- Donnees parfois bruitees ou incompletes
- Desequilibre de classes (beaucoup plus de trafic normal que malveillant)
- Variabilite temporelle des comportements reseau

## 2.2 Jeux de donnees (Dataset)

### Presentation de datasets possibles

Plusieurs jeux de donnees reconnus en detection d'intrusion peuvent etre utilises :
- **NSL-KDD** : version amelioree de KDD'99, souvent utilisee comme benchmark classique
- **CICIDS2017** : jeu de donnees recent avec trafic realiste et differents types d'attaques
- **IoT-23** : dataset oriente objets connectes, incluant des traces de botnets et d'activites malveillantes IoT

### Pretraitement

Avant l'apprentissage, un pipeline de pretraitement est applique :
- **Nettoyage** : suppression des doublons, traitement des valeurs manquantes, correction des incoherences
- **Normalisation** : mise a l'echelle des variables numeriques pour stabiliser l'apprentissage
- **Encodage des variables** : transformation des variables categoriques en representation numerique (ex. one-hot encoding)

## 2.3 Conception du modele de Deep Learning

### Choix de l'architecture

Le choix de l'architecture est fait selon la nature des donnees :
- **DNN/MLP** pour une representation tabulaire classique des features reseau
- **LSTM** pour exploiter les dependances temporelles presentes dans les sequences de trafic

### Structure du modele

Le modele est compose de :
- **Entrees** : vecteur de features reseau (ex. duree de flux, nombre de paquets, octets, flags)
- **Couches cachees** : couches denses (DNN) et/ou recurrentes (LSTM) pour apprendre des representations discriminantes
- **Fonction d'activation** : ReLU dans les couches cachees et Sigmoid (binaire) ou Softmax (multi-classes) en sortie

## 2.4 Phase d'apprentissage

### Division des donnees

La base est divisee en trois sous-ensembles :
- **Train** : apprentissage des parametres du modele
- **Validation** : reglage des hyperparametres et prevention du surapprentissage
- **Test** : evaluation finale des performances sur des donnees non vues

### Courbes de suivi de l'apprentissage

Le point « courbes de ... » est traite par l'analyse des courbes suivantes :
- Courbe de **loss** (train/validation)
- Courbe de **accuracy** (train/validation)

Ces courbes permettent d'observer la convergence du modele et de detecter un eventuel surapprentissage.

## 2.5 Implementation

L'implementation de la solution est realisee avec :
- **Python** comme langage principal
- **PyTorch** (et/ou TensorFlow selon l'experimentation) pour la construction et l'entrainement des modeles de Deep Learning

Des bibliotheques complementaires (NumPy, Pandas, scikit-learn, Matplotlib) sont utilisees pour le pretraitement, l'evaluation et la visualisation des resultats.

## 2.6 Planification du projet en sprints

Pour ameliorer le suivi du projet, une planification Agile en sprints courts est adoptee sur la periode du **19 fevrier au 1 mai**.

### Organisation proposee

- Duree d'un sprint : 2 semaines
- Rythme de suivi : point hebdomadaire + revue de sprint
- Livrable de chaque sprint : resultat mesurable (code, modele, metriques ou document)

### Decoupage des sprints

- **Sprint 1 (19 fevrier - 4 mars)**
	- Objectifs : cadrage du probleme, choix des datasets, definition des metriques
	- Livrables : plan de travail, structure du projet, jeux de donnees cibles identifies

- **Sprint 2 (5 mars - 18 mars)**
	- Objectifs : pretraitement des donnees (nettoyage, encodage, normalisation)
	- Livrables : pipeline de pretraitement valide, jeux train/validation/test prepares

- **Sprint 3 (19 mars - 1 avril)**
	- Objectifs : implementation des modeles de base (MLP/DNN, baselines ML)
	- Livrables : premiers modeles entraines, mesures initiales (accuracy, precision, recall, F1)

- **Sprint 4 (2 avril - 15 avril)**
	- Objectifs : implementation LSTM/CNN, ajustement des hyperparametres
	- Livrables : modeles avances entraines, courbes d'apprentissage et comparaison intermediaire

- **Sprint 5 (16 avril - 1 mai)**
	- Objectifs : evaluation finale, analyse comparative, redaction du rapport
	- Livrables : resultats consolides, figures finales, chapitre methodologie finalise

### Indicateurs de suivi (KPIs)

Pour chaque sprint, les indicateurs suivants sont recommandes :
- Taux d'objectifs atteints du sprint
- Evolution du F1-score et du recall
- Stabilité du modele (ecart train/validation)
- Respect du delai et qualite des livrables