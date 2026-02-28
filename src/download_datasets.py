"""
Script pour télécharger les datasets IoT NIDS :
- CICIDS2017
- NSL-KDD
- IoT-23
Gère le versioning et l'organisation dans data/raw/
"""
import os
import requests
from urllib.parse import urlparse

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(data_dir, exist_ok=True)

def download_file(url, dest):
    print(f"Téléchargement: {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Fichier sauvegardé: {dest}")

# CICIDS2017
cicids_url = "https://www.unb.ca/cic/datasets/malmem-2017.html" # Page, pas direct
cicids_note = "CICIDS2017 nécessite inscription et téléchargement manuel. Placez les fichiers dans data/raw/CICIDS2017/"
os.makedirs(os.path.join(data_dir, 'CICIDS2017'), exist_ok=True)

# NSL-KDD
nsl_urls = [
    "https://github.com/defcom17/NSL_KDD/blob/master/KDDTrain+.txt?raw=true",
    "https://github.com/defcom17/NSL_KDD/blob/master/KDDTest+.txt?raw=true"
]
for url in nsl_urls:
    fname = os.path.basename(urlparse(url).path)
    dest = os.path.join(data_dir, 'NSL-KDD', fname)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        download_file(url, dest)

# IoT-23
# Fichiers CSV disponibles sur le site officiel
# https://www.stratosphereips.org/datasets-iot23
# Téléchargement manuel recommandé (trop volumineux pour script direct)
iot23_note = "IoT-23: Téléchargez manuellement les fichiers depuis https://www.stratosphereips.org/datasets-iot23 et placez-les dans data/raw/IoT-23/"
os.makedirs(os.path.join(data_dir, 'IoT-23'), exist_ok=True)

# Versioning
with open(os.path.join(data_dir, 'VERSIONS.txt'), 'w') as f:
    f.write("CICIDS2017: manuel\nNSL-KDD: github defcom17\nIoT-23: manuel\n")
    f.write("Date téléchargement: 2026-02-28\n")

print("\nNotes:")
print(cicids_note)
print(iot23_note)
print("\nTous les fichiers sont dans data/raw/")
