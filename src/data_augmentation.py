"""
Data Augmentation Module for IoT Network Intrusion Detection
Provides techniques for small datasets and synthetic attack generation
"""

import argparse
import pathlib
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataAugmentation:
    """Handles various data augmentation techniques for network traffic data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def gaussian_noise(self, X: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to features (jittering).
        Good for network traffic features which are continuous.
        """
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noisy = X + noise
        # Clip to reasonable bounds
        X_noisy = np.clip(X_noisy, X.min(axis=0) * 0.5, X.max(axis=0) * 2.0)
        return X_noisy
    
    def scale_perturbation(self, X: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Scale features randomly (good for packet sizes, bytes, etc.).
        Each feature is scaled independently.
        """
        scales = np.random.uniform(scale_range[0], scale_range[1], X.shape[1])
        X_scaled = X * scales
        return X_scaled
    
    def mixup(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.2, num_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation: create virtual training examples.
        Blend two samples: X_new = lambda * X_i + (1-lambda) * X_j
        """
        if num_samples is None:
            num_samples = len(X)
        
        indices_a = np.random.randint(0, len(X), num_samples)
        indices_b = np.random.randint(0, len(X), num_samples)
        
        lam = np.random.beta(alpha, alpha, num_samples)
        lam = np.maximum(lam, 1 - lam)  # Ensure lambda >= 0.5
        
        X_new = lam[:, np.newaxis] * X[indices_a] + (1 - lam[:, np.newaxis]) * X[indices_b]
        y_new = y[indices_a]  # Keep label from first sample
        
        return X_new, y_new
    
    def cutmix(self, X: np.ndarray, y: np.ndarray, num_samples: int = None, mix_prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutMix augmentation: randomly mix portions of features.
        Mimics partial feature corruption/combination.
        """
        if num_samples is None:
            num_samples = len(X)
        
        indices_a = np.random.randint(0, len(X), num_samples)
        indices_b = np.random.randint(0, len(X), num_samples)
        
        X_new = X[indices_a].copy()
        
        for i in range(num_samples):
            if np.random.rand() < mix_prob:
                # Randomly select which features to replace
                num_features = X.shape[1]
                cut_point = np.random.randint(1, num_features)
                X_new[i, cut_point:] = X[indices_b[i], cut_point:]
        
        y_new = y[indices_a]
        return X_new, y_new
    
    def rotation_perturbation(self, X: np.ndarray, angle: float = 15.0) -> np.ndarray:
        """
        Random rotation in feature space (PCA-based).
        Good for detecting rotations in feature patterns.
        """
        from sklearn.decomposition import PCA
        
        # Project to 2D PCA space, rotate, project back
        if X.shape[1] < 2:
            return X
        
        pca = PCA(n_components=min(2, X.shape[1]))
        X_pca = pca.fit_transform(X)
        
        # Random rotation angle
        theta = np.random.uniform(-angle, angle) * np.pi / 180
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        X_rotated_pca = X_pca @ rotation_matrix.T
        X_rotated = pca.inverse_transform(X_rotated_pca)
        
        return X_rotated
    
    def shift_features(self, X: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
        """
        Shift features by a small amount (temporal shift for time-series data).
        """
        shifts = np.random.uniform(-shift_range, shift_range, X.shape[1])
        X_shifted = X + shifts * np.abs(X.mean(axis=0))
        return X_shifted
    
    def smote_style_interpolation(self, X: np.ndarray, y: np.ndarray, k_neighbors: int = 5, 
                                  num_synthetic_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE-style interpolation: generate synthetic samples by interpolating neighbors.
        Good for minority classes.
        """
        if num_synthetic_samples is None:
            num_synthetic_samples = len(X)
        
        from sklearn.neighbors import NearestNeighbors
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X)
        _, indices = nbrs.kneighbors(X)
        
        X_synthetic = []
        y_synthetic = []
        
        for _ in range(num_synthetic_samples):
            idx = np.random.randint(0, len(X))
            neighbor_idx = indices[idx, np.random.randint(1, k_neighbors)]
            
            # Interpolate between point and neighbor
            alpha = np.random.rand()
            synthetic_point = X[idx] + alpha * (X[neighbor_idx] - X[idx])
            X_synthetic.append(synthetic_point)
            y_synthetic.append(y[idx])
        
        return np.array(X_synthetic), np.array(y_synthetic)


class SyntheticAttackGenerator:
    """Generates synthetic attack patterns for training data augmentation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def estimate_attack_distribution(self, X_attack: np.ndarray) -> Dict[str, Any]:
        """Estimate statistical properties of attack features."""
        return {
            'mean': X_attack.mean(axis=0),
            'std': X_attack.std(axis=0),
            'min': X_attack.min(axis=0),
            'max': X_attack.max(axis=0),
            'percentiles': {
                '25': np.percentile(X_attack, 25, axis=0),
                '50': np.percentile(X_attack, 50, axis=0),
                '75': np.percentile(X_attack, 75, axis=0),
            }
        }
    
    def generate_gaussian_attacks(self, attack_dist: Dict[str, Any], num_samples: int) -> np.ndarray:
        """
        Generate synthetic attacks using Gaussian distribution from real attack examples.
        """
        mean = attack_dist['mean']
        std = attack_dist['std']
        
        # Add slight perturbation to std to increase diversity
        std_perturbed = std * np.random.uniform(0.8, 1.2, std.shape)
        std_perturbed = np.maximum(std_perturbed, 1e-6)  # Avoid zero std
        
        synthetic = np.random.normal(mean, std_perturbed, (num_samples, len(mean)))
        
        # Clip to observed ranges with some tolerance
        for i in range(len(mean)):
            lower = attack_dist['min'][i] - std[i]
            upper = attack_dist['max'][i] + std[i]
            synthetic[:, i] = np.clip(synthetic[:, i], lower, upper)
        
        return synthetic
    
    def generate_uniform_attacks(self, attack_dist: Dict[str, Any], num_samples: int) -> np.ndarray:
        """
        Generate synthetic attacks using uniform distribution within observed ranges.
        """
        synthetic = np.zeros((num_samples, len(attack_dist['mean'])))
        
        for i in range(len(attack_dist['mean'])):
            lower = attack_dist['min'][i]
            upper = attack_dist['max'][i]
            synthetic[:, i] = np.random.uniform(lower, upper, num_samples)
        
        return synthetic
    
    def generate_perturbed_attacks(self, X_attack: np.ndarray, num_samples: int, 
                                   perturbation_factor: float = 0.15) -> np.ndarray:
        """
        Generate synthetic attacks by perturbation of real examples.
        """
        synthetic = []
        
        for _ in range(num_samples):
            idx = np.random.randint(0, len(X_attack))
            base_attack = X_attack[idx].copy()
            
            # Add random perturbation
            perturbation = np.random.normal(0, perturbation_factor, base_attack.shape)
            perturbed = base_attack * (1 + perturbation)
            
            synthetic.append(perturbed)
        
        return np.array(synthetic)
    
    def generate_combined_attacks(self, X_attack: np.ndarray, num_samples: int,
                                  num_components: int = 2) -> np.ndarray:
        """
        Generate synthetic attacks by combining features from different real attacks.
        """
        if len(X_attack) < 2:
            return self.generate_perturbed_attacks(X_attack, num_samples)
        
        synthetic = []
        max_combine = min(num_components + 1, len(X_attack))
        
        for _ in range(num_samples):
            # Select random number of attacks to combine
            n_combine = np.random.randint(2, max_combine)
            indices = np.random.choice(len(X_attack), n_combine, replace=False)
            
            # Weighted combination
            weights = np.random.dirichlet(np.ones(n_combine))
            combined = np.average(X_attack[indices], axis=0, weights=weights)
            
            synthetic.append(combined)
        
        return np.array(synthetic)
    
    def generate_interpolated_attacks(self, X_attack: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Generate synthetic attacks by interpolation between real examples.
        """
        if len(X_attack) < 2:
            return self.generate_perturbed_attacks(X_attack, num_samples)
        
        synthetic = []
        
        for _ in range(num_samples):
            idx1, idx2 = np.random.choice(len(X_attack), 2, replace=False)
            alpha = np.random.rand()
            
            interpolated = (1 - alpha) * X_attack[idx1] + alpha * X_attack[idx2]
            synthetic.append(interpolated)
        
        return np.array(synthetic)


def augment_dataset(X_train: np.ndarray, y_train: np.ndarray, 
                   augmentation_factor: float = 1.0,
                   n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main augmentation function combining multiple techniques.
    
    Args:
        X_train: Training features
        y_train: Training labels
        augmentation_factor: Multiply dataset size by this factor (e.g., 2.0 = double the data)
        n_jobs: Number of parallel jobs (currently not used, kept for compatibility)
    
    Returns:
        Augmented X and y
    """
    aug = DataAugmentation()
    n_original = len(X_train)
    n_augmented = int(n_original * augmentation_factor) - n_original
    
    if n_augmented <= 0:
        return X_train, y_train
    
    augmented_samples = []
    augmented_labels = []
    
    # Apply different augmentation techniques
    techniques = [
        ('gaussian_noise', lambda: (aug.gaussian_noise(X_train), y_train)),
        ('scale_perturbation', lambda: (aug.scale_perturbation(X_train), y_train)),
        ('mixup', lambda: aug.mixup(X_train, y_train, num_samples=max(1, n_augmented // 4))),
        ('cutmix', lambda: aug.cutmix(X_train, y_train, num_samples=max(1, n_augmented // 4))),
        ('shift_features', lambda: (aug.shift_features(X_train), y_train)),
    ]
    
    n_per_technique = max(1, n_augmented // len(techniques))
    
    for tech_name, tech_func in techniques:
        try:
            X_aug, y_aug = tech_func()
            # Safely sample from augmented data with correct bounds
            n_samples = min(n_per_technique, len(X_aug))
            if n_samples > 0:
                indices = np.random.choice(len(X_aug), n_samples, replace=False)
                augmented_samples.append(X_aug[indices])
                augmented_labels.extend(y_aug[indices])
        except Exception as e:
            print(f"Warning: {tech_name} failed - {e}")
            continue
    
    if augmented_samples:
        X_augmented = np.vstack(augmented_samples)
        y_augmented = np.array(augmented_labels)
        
        # Combine original and augmented (limit to desired augmentation)
        n_to_add = min(n_augmented, len(X_augmented))
        X_combined = np.vstack([X_train, X_augmented[:n_to_add]])
        y_combined = np.hstack([y_train, y_augmented[:n_to_add]])
        
        return X_combined, y_combined
    
    return X_train, y_train


def generate_synthetic_attacks(X_train: np.ndarray, y_train: np.ndarray,
                              target_class: int = None,
                              num_synthetic_per_class: int = 100,
                              methods: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic attack samples for minority classes or specific attack types.
    
    Args:
        X_train: Training features
        y_train: Training labels
        target_class: Specific class to generate (None = all minority classes)
        num_synthetic_per_class: Number of synthetic samples per class
        methods: List of generation methods to use
    
    Returns:
        Synthetic features and labels
    """
    if methods is None:
        methods = ['gaussian', 'uniform', 'perturbed', 'interpolated']
    
    gen = SyntheticAttackGenerator()
    synthetic_X = []
    synthetic_y = []
    
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    majority_count = class_counts.max()
    
    target_classes = [target_class] if target_class is not None else unique_classes
    
    for cls in target_classes:
        cls_mask = y_train == cls
        X_cls = X_train[cls_mask]
        cls_count = len(X_cls)
        
        # Only generate for minority classes or if target_class specified
        if target_class is None and cls_count >= majority_count * 0.8:
            continue
        
        print(f"Generating synthetic samples for class {cls} ({cls_count} original)")
        
        attack_dist = gen.estimate_attack_distribution(X_cls)
        n_samples_per_method = num_synthetic_per_class // len(methods)
        
        if 'gaussian' in methods:
            synthetic_X.append(gen.generate_gaussian_attacks(attack_dist, n_samples_per_method))
            synthetic_y.extend([cls] * n_samples_per_method)
        
        if 'uniform' in methods and n_samples_per_method > 0:
            synthetic_X.append(gen.generate_uniform_attacks(attack_dist, n_samples_per_method))
            synthetic_y.extend([cls] * n_samples_per_method)
        
        if 'perturbed' in methods and n_samples_per_method > 0:
            synthetic_X.append(gen.generate_perturbed_attacks(X_cls, n_samples_per_method))
            synthetic_y.extend([cls] * n_samples_per_method)
        
        if 'interpolated' in methods and n_samples_per_method > 0:
            synthetic_X.append(gen.generate_interpolated_attacks(X_cls, n_samples_per_method))
            synthetic_y.extend([cls] * n_samples_per_method)
    
    if synthetic_X:
        return np.vstack(synthetic_X), np.array(synthetic_y)
    
    return np.array([]), np.array([])


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Augmentation for Network Intrusion Detection")
    parser.add_argument("--input", type=pathlib.Path, help="Input npz file with X and y")
    parser.add_argument("--augmentation-factor", type=float, default=1.5, 
                       help="Multiply dataset size (1.5 = 50% more data)")
    parser.add_argument("--synthetic-method", type=str, default="gaussian,uniform,perturbed,interpolated",
                       help="Comma-separated list of synthetic generation methods")
    parser.add_argument("--synthetic-per-class", type=int, default=100,
                       help="Number of synthetic samples per minority class")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/augmented.npz"),
                       help="Output npz file for augmented data")
    parser.add_argument("--augment-only", action="store_true", help="Only augment (no synthetic generation)")
    parser.add_argument("--synthetic-only", action="store_true", help="Only generate synthetic (no augmentation)")
    
    args = parser.parse_args()
    
    if not args.input or not args.input.exists():
        print("Error: Input npz file not found")
        return
    
    print("=" * 60)
    print("DATA AUGMENTATION AND SYNTHETIC GENERATION")
    print("=" * 60)
    
    # Load data
    data = np.load(args.input)
    X, y = data['X'], data['y']
    print(f"Original data shape: X={X.shape}, y={y.shape}")
    
    # Basic statistics
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    X_final = X.copy()
    y_final = y.copy()
    
    # Apply augmentation
    if not args.synthetic_only:
        print(f"\n" + "=" * 60)
        print("AUGMENTATION PHASE")
        print("=" * 60)
        print(f"Augmentation factor: {args.augmentation_factor}x")
        X_aug, y_aug = augment_dataset(X, y, augmentation_factor=args.augmentation_factor)
        print(f"Augmented data shape: X={X_aug.shape}, y={y_aug.shape}")
        X_final, y_final = X_aug, y_aug
    
    # Generate synthetic attacks
    if not args.augment_only:
        print(f"\n" + "=" * 60)
        print("SYNTHETIC ATTACK GENERATION")
        print("=" * 60)
        methods = [m.strip() for m in args.synthetic_method.split(',')]
        print(f"Methods: {methods}")
        print(f"Samples per class: {args.synthetic_per_class}")
        
        X_syn, y_syn = generate_synthetic_attacks(
            X_final, y_final,
            num_synthetic_per_class=args.synthetic_per_class,
            methods=methods
        )
        
        if len(X_syn) > 0:
            X_final = np.vstack([X_final, X_syn])
            y_final = np.hstack([y_final, y_syn])
            print(f"Synthetic samples added: {len(X_syn)}")
    
    # Save augmented data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X_final, y=y_final)
    
    # Final statistics
    print(f"\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"Final data shape: X={X_final.shape}, y={y_final.shape}")
    print(f"Data size increased by: {(len(y_final) / len(y)) * 100:.1f}%")
    
    unique_final, counts_final = np.unique(y_final, return_counts=True)
    print(f"\nFinal class distribution:")
    for cls, count in zip(unique_final, counts_final):
        print(f"  Class {cls}: {count} samples ({count/len(y_final)*100:.1f}%)")
    
    print(f"\n✓ Saved augmented data to {args.output}")


if __name__ == "__main__":
    main()
