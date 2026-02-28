# Documentation des datasets IoT NIDS

## CICIDS2017
- **Source:** https://www.unb.ca/cic/datasets/malmem-2017.html
- **Format:** CSV, chaque ligne = un flux réseau
- **Colonnes principales:** Src IP, Src Port, Dst IP, Dst Port, Protocol, Flow Duration, Flag Count, Tot Fwd Pkts, Tot Bwd Pkts, Label
- **Types d'attaques:** DoS, DDoS, infiltration, brute force, port scan, botnet, web attacks

## NSL-KDD
- **Source:** https://github.com/defcom17/NSL_KDD
- **Format:** TXT, chaque ligne = un flux réseau
- **Colonnes principales:** src_ip, src_port, dst_ip, dst_port, protocol, duration, flag, packet_size, label
- **Types d'attaques:** DoS, Probe, R2L, U2R

## IoT-23
- **Source:** https://www.stratosphereips.org/datasets-iot23
- **Format:** CSV, chaque fichier = un scénario d'attaque ou normal
- **Colonnes principales:** src_ip, src_port, dst_ip, dst_port, protocol, duration, flag, packet_size, label
- **Types d'attaques:** DDoS, malware, port scan, botnet, brute force, reconnaissance, spoofing

---

# Guide d'utilisation

1. **Téléchargement des datasets**
   - CICIDS2017 et IoT-23: Téléchargement manuel recommandé (voir download_datasets.py)
   - NSL-KDD: Téléchargement automatique via script

2. **Extraction des features**
   - Utiliser `feature_extraction.py` pour générer des CSVs uniformisés dans `data/processed/`

3. **Structure des dossiers**
   - `data/raw/` : fichiers bruts
   - `data/processed/` : features extraites
   - `data/preprocessed/` : fichiers pour entraînement

4. **Types d'attaques**
   - Voir la colonne `label` dans chaque dataset pour la liste exacte

5. **Prétraitement**
   - Utiliser `data_prep.py` pour normalisation, split, encodage

---

# Références
- CICIDS2017: [UNB CIC](https://www.unb.ca/cic/datasets/malmem-2017.html)
- NSL-KDD: [GitHub](https://github.com/defcom17/NSL_KDD)
- IoT-23: [Stratosphere IPS](https://www.stratosphereips.org/datasets-iot23)
