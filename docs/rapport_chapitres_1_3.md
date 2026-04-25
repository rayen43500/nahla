# Chapitre 1 : Contexte et fondements

## 1.1 Introduction a l'Internet des Objets (IoT)

L'Internet des Objets (IoT) designe un ecosysteme ou des objets physiques (capteurs, actionneurs, cameras, automates, passerelles, appareils medicaux, etc.) sont connectes a des reseaux pour collecter, echanger et exploiter des donnees.

### Definition et architecture IoT

Une architecture IoT classique peut etre decomposee en couches :
- Couche perception : capteurs et dispositifs qui capturent les mesures du monde reel.
- Couche reseau : protocoles et infrastructures de communication (Wi-Fi, 4G/5G, LPWAN, Ethernet industriel, etc.).
- Couche traitement : edge computing, cloud, stockage, analytics et intelligence artificielle.
- Couche application : services metier (supervision, maintenance predictive, telemedecine, etc.).

### Caracteristiques de l'IoT

Les environnements IoT presentent des contraintes fortes :
- Faible puissance de calcul et memoire limitee sur les objets terminaux.
- Contraintes energetiques (batterie, autonomie, duty-cycle).
- Heterogeneite materielle et logicielle elevee.
- Connectivite massive et distribuee, souvent sur des reseaux peu fiables.
- Exposition en surface d'attaque a grande echelle.

### Domaines d'application

L'IoT est present dans de nombreux secteurs :
- Smart cities : eclairage intelligent, gestion du trafic, surveillance environnementale.
- Industrie 4.0 : supervision de chaines de production, maintenance predictive, jumeaux numeriques.
- Sante : telemonitoring, dispositifs medicaux connectes, suivi patient a distance.
- Energie et reseaux intelligents : smart grids, optimisation de consommation.
- Transport et logistique : suivi d'actifs, gestion de flotte, entrepots connectes.

## 1.2 Securite dans les environnements IoT

### Vulnerabilites specifiques de l'IoT

Les systemes IoT sont particulierement vulnerables pour plusieurs raisons :
- Mots de passe par defaut et configurations faibles.
- Mises a jour de securite rares ou complexes a deployer.
- Chiffrement parfois absent ou mal configure.
- Protocoles legers privilegies au detriment de la securite.
- Multiplicite des points d'entree (objets, passerelles, API, cloud).

### Types d'attaques

Les attaques les plus frequentes dans les flux reseau IoT incluent :
- DoS / DDoS : saturation des ressources reseau ou applicatives pour rendre un service indisponible.
- Attaques de spoofing : usurpation d'identite (adresse IP/MAC, equipement, session).
- Botnets IoT (exemple Mirai) : compromission massive d'equipements afin d'orchestrer des attaques distribuees.

Dans ce projet, les donnees utilisees couvrent notamment des scenarios DoS, DDoS et attaques de scan, proches de cas reels d'infrastructure connectee.

### Limites des solutions de securite classiques

Les approches classiques (regles statiques, filtrage fixe, signatures uniquement) montrent vite leurs limites :
- Faible capacite de generalisation face a des attaques inconnues.
- Maintenance importante des signatures et regles.
- Taux de faux positifs eleve en environnement dynamique.
- Difficultes de passage a l'echelle avec l'explosion des flux IoT.

## 1.3 Systemes de detection d'intrusion (IDS)

Un IDS (Intrusion Detection System) surveille des activites reseau ou systeme pour detecter des comportements malveillants.

### IDS bases sur les signatures

Principe : comparer le trafic observe a une base de motifs connus.

Avantages :
- Efficace pour des attaques deja repertoriees.
- Interpretable et simple a operationaliser.

Limites :
- Inefficace contre zero-day et variantes non connues.
- Dependance a des mises a jour frequentes.

### IDS bases sur les anomalies

Principe : apprendre le comportement normal puis detecter les deviations.

Avantages :
- Meilleure detection potentielle des attaques nouvelles.
- Plus adapte aux environnements dynamiques et heterogenes.

Limites :
- Sensibilite au bruit et a la derive de donnees.
- Risque de faux positifs si le modele est mal calibre.

### Metriques de performance

Les metriques centrales pour evaluer un IDS sont :
- Accuracy : proportion globale de predictions correctes.
- Precision : parmi les alertes, part d'alertes correctes.
- Recall : parmi les attaques reelles, part effectivement detectee.
- F1-score : compromis entre precision et recall.
- Taux de faux positifs (FPR) : proportion de trafic normal incorrectement signale comme attaque.

Dans ce projet, l'evaluation combine accuracy, precision, recall, F1-score et matrice de confusion pour une lecture fine des erreurs.

## 1.4 Introduction au Deep Learning

### Difference Machine Learning vs Deep Learning

Machine Learning classique :
- Repose souvent sur du feature engineering manuel.
- Performant sur des problemes tabulaires bien structures.
- Modeles interpretable selon l'algorithme (ex. arbres, SVM lineaire).

Deep Learning :
- Apprend automatiquement des representations hierarchiques.
- Plus robuste pour capturer des motifs complexes/non lineaires.
- Demande generalement plus de donnees et de ressources de calcul.

### Architectures principales

Les architectures majeures mobilisables pour un IDS IoT sont :
- DNN (Deep Neural Networks) : adaptation efficace aux features tabulaires reseau.
- CNN : extraction de motifs locaux dans des sequences ou representations structurees.
- RNN / LSTM : prise en compte des dependances temporelles dans les flux reseau.

Le projet implemente plusieurs familles : MLP/DNN, CNN, LSTM et un modele hybride CNN-LSTM, avec comparaison aux baselines ML (Random Forest, SVM, XGBoost).

### Interet du DL pour la cybersecurite IoT

Le Deep Learning est pertinent en IoT car il permet :
- Detection automatique de patterns d'attaque complexes.
- Generalisation superieure face a la variabilite des flux.
- Reduction potentielle de l'ingenierie manuelle de features.
- Capacite de travailler en mode hybride (classification + detection d'anomalies).

# Chapitre 2 : Methodologie proposee

## 2.1 Description du probleme

Le probleme est formule comme une classification binaire :
- Classe 0 : trafic normal.
- Classe 1 : trafic malveillant (attaque).

Selon les besoins, une extension multi-classes est possible (DoS, DDoS, PortScan, WebAttack, etc.).

Les donnees reseau IoT sont souvent :
- Heterogenes (sources et protocoles multiples).
- Desequilibrees (beaucoup plus de normal que d'attaques rares).
- Bruitees, avec valeurs manquantes ou distributions instables.

## 2.2 Jeux de donnees (Dataset)

### Presentation de datasets possibles

Le cadre experimental s'appuie sur des datasets de reference :
- NSL-KDD : benchmark historique en detection d'intrusion.
- CICIDS2017 : scenarios recents avec grande diversite d'attaques.
- IoT-23 : corpus orientes IoT et comportements malveillants reels.

Dans ce projet, le travail est principalement aligne sur des donnees de type CICIDS2017 (fichiers Friday/Thursday presentes dans le repertoire data/raw), avec une structuration compatible pour extension vers NSL-KDD et IoT-23.

### Pretraitement

Le pipeline de pretraitement applique les etapes suivantes :
- Nettoyage : suppression/traitement des doublons, valeurs manquantes, incoherences.
- Normalisation/scaling : StandardScaler ou RobustScaler selon la sensibilite aux outliers.
- Encodage : conversion des variables categoriques (one-hot encoding).
- Gestion du desequilibre : techniques de reequilibrage (ex. SMOTE).
- Reduction dimensionnelle (optionnelle) : PCA pour accelerer l'apprentissage.
- Split des donnees : separation train/validation/test avec stratification.

## 2.3 Conception du modele de Deep Learning

### Choix de l'architecture

Le choix d'architecture suit une logique comparative :
- DNN/MLP : baseline DL solide pour donnees tabulaires.
- LSTM : pertinent pour modeliser la dynamique temporelle du trafic.
- CNN 1D : efficace pour capter des motifs locaux.
- Hybride CNN-LSTM : combine extraction locale et dependances temporelles.

### Structure du modele

Une structure type comprend :
- Entrees : vecteur de features reseau (debit, flags, duree, paquets, octets, etc.).
- Couches cachees : blocs denses et/ou convolutifs et/ou recurrents selon l'architecture.
- Fonctions d'activation : ReLU/LeakyReLU en couches intermediaires, Sigmoid/Softmax en sortie selon la tache.
- Regularisation : dropout, batch normalization, early stopping pour limiter le surapprentissage.

## 2.4 Phase d'apprentissage

### Division des donnees

Le protocole d'apprentissage suit une decomposition standard :
- Train set : apprentissage des poids du modele.
- Validation set : ajustement hyperparametres et controle du surapprentissage.
- Test set : mesure finale des performances hors entrainement.

### Courbes a analyser

Pour le point mentionne « courbes de ... », les courbes recommandees sont :
- Courbes d'apprentissage : loss train/validation et accuracy train/validation par epoch.
- Courbes Precision-Recall : pertinentes en cas de classes desequilibrees.
- Courbes ROC-AUC : evaluation de la capacite de discrimination.
- Evolution du F1-score : utile pour arbitrer entre precision et recall.

## 2.5 Implementation

Les outils utilises dans le projet sont :
- Python : langage principal de developpement.
- PyTorch : framework principal pour l'entrainement des modeles DL.
- Scikit-learn et XGBoost : baselines ML, preprocessing et metriques.
- FastAPI : exposition d'un service de prediction (mode API).

TensorFlow/Keras peut etre envisage dans des extensions futures, mais l'implementation actuelle est majoritairement centree sur PyTorch.

# Chapitre 3 : Resultats et analyse

## 3.1 Evaluation des performances

L'evaluation des modeles est basee sur :
- Accuracy globale.
- Precision, Recall et F1-score.
- Matrice de confusion pour analyser en detail les erreurs de classification.

Le projet conserve les metriques et matrices pour chaque modele (MLP, LSTM, CNN, Hybrid, Random Forest, SVM, XGBoost), ce qui permet une comparaison rigoureuse.

## 3.2 Analyse des resultats

### Comparaison avec les methodes classiques

La comparaison DL vs ML classique met en evidence :
- Les modeles DL capturent mieux des relations complexes.
- Les modeles classiques restent competitifs sur certains sous-ensembles tabulaires.
- L'analyse finale doit considerer a la fois performance, temps d'inference et cout de deploiement.

### Impact du choix du modele

Le choix du modele influe sur :
- La sensibilite a certains types d'attaques.
- La stabilite en presence de bruit et de desequilibre.
- Le compromis entre precision de detection et faux positifs.

### Robustesse face aux attaques inconnues

La robustesse zero-day se mesure via la capacite du modele a detecter des patterns jamais observes en entrainement.

Dans ce projet, cette robustesse est analysee par des protocoles de test dedies et une lecture conjointe des metriques globales et par classe.

## 3.3 Discussion

### Avantages du Deep Learning

Le Deep Learning apporte :
- Detection automatique de patterns complexes sans regles manuelles exhaustives.
- Bonne adaptation a des volumes importants de trafic.
- Possibilite d'amelioration continue par reentrainement.

### Limites

Les principales limites observees sont :
- Besoin de grandes quantites de donnees etiquetees et de qualite.
- Cout computationnel (entrainement et parfois inference).
- Complexite de calibrage et d'interpretabilite des modeles.

## 3.4 Perspectives et ameliorations

Les evolutions les plus prometteuses pour l'IDS IoT sont :
- Deep Reinforcement Learning : adaptation dynamique des politiques de detection/reponse.
- Federated Learning : entrainement distribue sans centraliser les donnees, point cle pour l'IoT.
- Detection en temps reel : integration streaming et latence faible sur flux vivants.
- Optimisation embarquee : quantization, pruning et modeles compacts pour passerelles/edge devices.

## Conclusion partielle (chapitres 1 a 3)

Ces trois chapitres etablissent une base theorique et methodologique solide pour la detection d'intrusions IoT par Deep Learning. Le cadre propose articule clairement le contexte IoT, les enjeux de securite, les choix de modelisation et les criteres d'evaluation. La suite du rapport peut approfondir les resultats quantitatifs exacts par modele et formaliser un plan de deploiement operationnel.