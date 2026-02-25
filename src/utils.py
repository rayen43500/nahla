"""
Fonctions utilitaires - Projet Détection Intrusions IoT

Contient des helpers pour:
- Gestion des fichiers
- Préparation des données
- Logging
- Configuration
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


# ============================================================================
# CONFIGURATION ET LOGGING
# ============================================================================

def setup_logging(log_dir: Path = Path("logs"), log_level: str = "INFO") -> logging.Logger:
    """
    Configure le logging structuré pour le projet.
    
    Args:
        log_dir: Répertoire pour les logs
        log_level: Niveau de log (INFO, DEBUG, WARNING, ERROR)
        
    Returns:
        Logger configuré
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("IoT_IDS")
    logger.setLevel(getattr(logging, log_level))
    
    # File handler
    fh = logging.FileHandler(log_dir / "project.log")
    fh.setLevel(getattr(logging, log_level))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Charge la configuration du projet.
    
    Args:
        config_path: Chemin du fichier config.ini
        
    Returns:
        Dictionnaire de configuration
    """
    import configparser
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    result = {}
    for section in config.sections():
        result[section] = dict(config.items(section))
    
    return result


# ============================================================================
# GESTION DES CHEMINS
# ============================================================================

def create_project_dirs(base_dir: Path) -> Dict[str, Path]:
    """
    Crée les répertoires principaux du projet.
    
    Args:
        base_dir: Répertoire racine
        
    Returns:
        Dictionnaire {nom: Path} pour chaque répertoire
    """
    dirs = {
        'data_raw': base_dir / 'data' / 'raw',
        'data_processed': base_dir / 'data' / 'processed',
        'data_preprocessed': base_dir / 'data' / 'preprocessed',
        'models': base_dir / 'models',
        'results': base_dir / 'results',
        'logs': base_dir / 'logs',
        'notebooks': base_dir / 'notebooks',
    }
    
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    
    return dirs


# ============================================================================
# GESTION DONNÉES
# ============================================================================

def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """
    Sauvegarde les métriques en JSON.
    
    Args:
        metrics: Dict avec métriques
        output_path: Chemin fichier .json
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """
    Charge les métriques depuis JSON.
    
    Args:
        metrics_path: Chemin fichier .json
        
    Returns:
        Dict avec métriques
    """
    with open(metrics_path, 'r') as f:
        return json.load(f)


def get_class_weights(y: 'np.ndarray') -> Dict[int, float]:
    """
    Calcule les poids des classes pour déséquilibre.
    
    Args:
        y: Array de labels
        
    Returns:
        Dict {class_id: weight}
    """
    import numpy as np
    
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {}
    
    for cls, count in zip(unique, counts):
        weights[int(cls)] = total / (len(unique) * count)
    
    return weights


# ============================================================================
# VALIDATION
# ============================================================================

def validate_data_integrity(X: 'np.ndarray', y: 'np.ndarray') -> bool:
    """
    Valide l'intégrité des données.
    
    Args:
        X: Features
        y: Labels
        
    Returns:
        True si OK
    """
    import numpy as np
    
    if X.shape[0] != y.shape[0]:
        return False
    
    if np.isnan(X).any():
        return False
    
    if np.isinf(X).any():
        return False
    
    return True


def check_data_leakage_risk(X_train: 'np.ndarray', X_test: 'np.ndarray', 
                            tolerance: float = 1e-6) -> bool:
    """
    Détecte risk de data leakage entre train et test.
    
    Args:
        X_train: Train set
        X_test: Test set
        tolerance: Tolérance statistique
        
    Returns:
        True si risk détecté
    """
    import numpy as np
    
    overlap = np.intersect1d(X_train, X_test, return_indices=True)
    if len(overlap[0]) > 0:
        return True
    
    return False


# ============================================================================
# MODÈLES - HELPERS
# ============================================================================

def count_parameters(model: 'torch.nn.Module') -> int:
    """
    Compte le nombre de paramètres d'un modèle.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Nombre total de paramètres
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: 'torch.nn.Module', input_shape: tuple) -> None:
    """
    Affiche un résumé du modèle.
    
    Args:
        model: Modèle PyTorch
        input_shape: Shape input (batch, features)
    """
    import torch
    from torchsummary import summary
    
    try:
        summary(model, input_shape)
    except Exception as e:
        print(f"Erreur summary: {e}")
        print(f"Paramètres totaux: {count_parameters(model)}")


# ============================================================================
# PERFORMANCE
# ============================================================================

def estimate_training_time(dataset_size: int, epochs: int, 
                          batch_size: int, sample_per_sec: float = 1000) -> float:
    """
    Estime le temps d'entraînement.
    
    Args:
        dataset_size: Nombre d'exemples
        epochs: Nombre d'epochs
        batch_size: Taille batch
        sample_per_sec: Samples/sec sur le hardware
        
    Returns:
        Temps estimé en secondes
    """
    total_samples = dataset_size * epochs
    return total_samples / sample_per_sec


# ============================================================================
# VISUALISATION
# ============================================================================

def get_colormap(num_classes: int) -> list:
    """
    Retourne une colormap pour N classes.
    
    Args:
        num_classes: Nombre de classes
        
    Returns:
        Liste de couleurs
    """
    import matplotlib.pyplot as plt
    
    cmap = plt.cm.get_cmap('husl')
    colors = [cmap(i / num_classes) for i in range(num_classes)]
    
    return colors


if __name__ == "__main__":
    # Test
    logger = setup_logging()
    logger.info("Utils module loaded successfully")
