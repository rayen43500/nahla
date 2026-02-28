"""
Script d'extraction de features réseau pour NIDS IoT
- Unifie les colonnes des datasets CICIDS2017, NSL-KDD, IoT-23
- Extrait IP src/dst, ports, protocoles, flags, taille paquets, durations
- Sauvegarde CSV intermédiaires dans data/processed/
"""
import os
import pandas as pd

data_raw = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
data_processed = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(data_processed, exist_ok=True)

def process_nsl_kdd():
    # NSL-KDD: colonnes fixes
    cols = [
        'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'duration', 'flag', 'packet_size', 'label'
    ]
    # Fichier train/test
    for fname in ['KDDTrain+.txt', 'KDDTest+.txt']:
        fpath = os.path.join(data_raw, 'NSL-KDD', fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, header=None)
            # Extraction simplifiée (mock)
            df.columns = cols
            df.to_csv(os.path.join(data_processed, f'nsl_kdd_{fname}.csv'), index=False)

def process_cicids2017():
    # CICIDS2017: structure complexe, à adapter selon fichiers
    cicids_dir = os.path.join(data_raw, 'CICIDS2017')
    for fname in os.listdir(cicids_dir):
        if fname.endswith('.csv'):
            fpath = os.path.join(cicids_dir, fname)
            df = pd.read_csv(fpath)
            # Sélection des colonnes principales
            keep = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Flow Duration', 'Flag Count', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Label']
            df = df[[c for c in keep if c in df.columns]]
            # Renommage pour uniformité
            df = df.rename(columns={
                'Src IP': 'src_ip', 'Src Port': 'src_port', 'Dst IP': 'dst_ip', 'Dst Port': 'dst_port',
                'Protocol': 'protocol', 'Flow Duration': 'duration', 'Flag Count': 'flag',
                'Tot Fwd Pkts': 'packet_size', 'Label': 'label'
            })
            df.to_csv(os.path.join(data_processed, f'cicids2017_{fname}'), index=False)

def process_iot23():
    # IoT-23: chaque sous-dossier = un scénario
    iot_dir = os.path.join(data_raw, 'IoT-23')
    for root, dirs, files in os.walk(iot_dir):
        for fname in files:
            if fname.endswith('.csv'):
                fpath = os.path.join(root, fname)
                df = pd.read_csv(fpath)
                # Extraction simplifiée
                keep = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'duration', 'flag', 'packet_size', 'label']
                df = df[[c for c in keep if c in df.columns]]
                df.to_csv(os.path.join(data_processed, f'iot23_{fname}'), index=False)

process_nsl_kdd()
process_cicids2017()
process_iot23()
print("Features extraites et CSVs sauvegardés dans data/processed/")
