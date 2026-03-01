"""
Data Validation Module for IoT Network Intrusion Detection
Validates data integrity, checks for leakage, and analyzes distributions
"""

import argparse
import pathlib
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class DataLeakageDetector:
    """Detects potential data leakage between train/test/val sets."""
    
    @staticmethod
    def check_identical_samples(X_train: np.ndarray, X_test: np.ndarray, 
                               tolerance: float = 1e-5) -> Dict[str, Any]:
        """
        Check for identical samples across train and test sets.
        Could indicate data leakage.
        """
        results = {
            'identical_samples': 0,
            'similar_samples': 0,
            'leakage_risk': 'LOW'
        }
        
        if len(X_train) == 0 or len(X_test) == 0:
            return results
        
        # Check for exact duplicates
        for test_sample in X_test:
            for train_sample in X_train:
                if np.allclose(test_sample, train_sample, atol=tolerance):
                    results['identical_samples'] += 1
                    break
        
        # Check for similar samples (within 5% tolerance)
        similarity_threshold = 0.95
        for test_sample in X_test:
            distances = np.linalg.norm(X_train - test_sample, axis=1)
            min_distance = distances.min()
            max_distance = distances.max()
            
            if max_distance > 0:
                similarity = 1 - (min_distance / max_distance)
                if similarity > similarity_threshold:
                    results['similar_samples'] += 1
        
        # Assess risk
        total_test = len(X_test)
        leakage_ratio = results['identical_samples'] / total_test if total_test > 0 else 0
        
        if leakage_ratio > 0.1:
            results['leakage_risk'] = 'HIGH'
        elif leakage_ratio > 0.01:
            results['leakage_risk'] = 'MEDIUM'
        else:
            results['leakage_risk'] = 'LOW'
        
        return results
    
    @staticmethod
    def check_statistical_overlap(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Check if test labels are within training label distribution.
        """
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))
        
        unforeseen_classes = test_classes - train_classes
        seen_classes = test_classes & train_classes
        
        return {
            'train_classes': len(train_classes),
            'test_classes': len(test_classes),
            'seen_in_train': len(seen_classes),
            'unforeseen_classes': len(unforeseen_classes),
            'class_coverage': len(seen_classes) / len(test_classes) if len(test_classes) > 0 else 1.0
        }
    
    @staticmethod
    def check_feature_overlap(X_train: np.ndarray, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Check if test features are within training feature ranges.
        Could indicate extrapolation risk.
        """
        train_min = X_train.min(axis=0)
        train_max = X_train.max(axis=0)
        
        test_min = X_test.min(axis=0)
        test_max = X_test.max(axis=0)
        
        # Features outside training range
        features_below = (test_min < train_min).sum()
        features_above = (test_max > train_max).sum()
        
        total_features = X_train.shape[1]
        
        return {
            'total_features': total_features,
            'features_below_min': features_below,
            'features_above_max': features_above,
            'features_outside_range': features_below + features_above,
            'extrapolation_risk': (features_below + features_above) / total_features
        }


class DistributionAnalyzer:
    """Analyzes feature distributions before and after preprocessing."""
    
    @staticmethod
    def compute_statistics(X: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
        """Compute comprehensive statistics for all features."""
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        stats = []
        for i, name in enumerate(feature_names):
            col = X[:, i]
            stats.append({
                'Feature': name,
                'Mean': col.mean(),
                'Std': col.std(),
                'Min': col.min(),
                'Max': col.max(),
                'Median': np.median(col),
                'Q1': np.percentile(col, 25),
                'Q3': np.percentile(col, 75),
                'Skewness': _compute_skewness(col),
                'Kurtosis': _compute_kurtosis(col),
                'Range': col.max() - col.min(),
                'IQR': np.percentile(col, 75) - np.percentile(col, 25)
            })
        
        return pd.DataFrame(stats)
    
    @staticmethod
    def compare_distributions(X_before: np.ndarray, X_after: np.ndarray,
                             feature_names: List[str] = None) -> pd.DataFrame:
        """Compare statistics before and after normalization."""
        stats_before = DistributionAnalyzer.compute_statistics(X_before, feature_names)
        stats_after = DistributionAnalyzer.compute_statistics(X_after, feature_names)
        
        comparison = pd.DataFrame({
            'Feature': stats_before['Feature'],
            'Mean_Before': stats_before['Mean'],
            'Mean_After': stats_after['Mean'],
            'Std_Before': stats_before['Std'],
            'Std_After': stats_after['Std'],
            'Range_Before': stats_before['Range'],
            'Range_After': stats_after['Range'],
            'Skewness_Before': stats_before['Skewness'],
            'Skewness_After': stats_after['Skewness'],
        })
        
        return comparison
    
    @staticmethod
    def detect_outliers(X: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """Detect outliers using IQR or Z-score method."""
        outlier_counts = {}
        
        for i in range(X.shape[1]):
            col = X[:, i]
            
            if method == 'iqr':
                Q1 = np.percentile(col, 25)
                Q3 = np.percentile(col, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = ((col < lower_bound) | (col > upper_bound)).sum()
            else:  # z-score
                z_scores = np.abs((col - col.mean()) / (col.std() + 1e-8))
                outliers = (z_scores > threshold).sum()
            
            outlier_counts[f'Feature_{i}'] = {
                'count': outliers,
                'percentage': (outliers / len(col)) * 100
            }
        
        total_outliers = sum(v['count'] for v in outlier_counts.values())
        
        return {
            'method': method,
            'threshold': threshold,
            'feature_outlier_counts': outlier_counts,
            'total_outliers': total_outliers,
            'total_cells': X.size,
            'outlier_percentage': (total_outliers / X.size) * 100
        }


class StratificationValidator:
    """Validates class stratification across train/val/test sets."""
    
    @staticmethod
    def check_stratification(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Check if class distributions are well-stratified across splits.
        """
        def get_class_dist(y):
            classes, counts = np.unique(y, return_counts=True)
            total = len(y)
            return {cls: (count / total) * 100 for cls, count in zip(classes, counts)}
        
        train_dist = get_class_dist(y_train)
        val_dist = get_class_dist(y_val)
        test_dist = get_class_dist(y_test)
        
        # Compute Chi-square statistic for stratification quality
        all_classes = set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys())
        
        chi_square_train_val = _compute_chi_square_distance(train_dist, val_dist, all_classes)
        chi_square_train_test = _compute_chi_square_distance(train_dist, test_dist, all_classes)
        chi_square_val_test = _compute_chi_square_distance(val_dist, test_dist, all_classes)
        
        # Maximum chi-square for bad stratification
        stratification_quality = max(chi_square_train_val, chi_square_train_test, chi_square_val_test)
        
        if stratification_quality < 0.1:
            quality_rating = 'EXCELLENT'
        elif stratification_quality < 0.3:
            quality_rating = 'GOOD'
        elif stratification_quality < 0.6:
            quality_rating = 'ACCEPTABLE'
        else:
            quality_rating = 'POOR'
        
        return {
            'train_distribution': train_dist,
            'val_distribution': val_dist,
            'test_distribution': test_dist,
            'chi_square_train_val': chi_square_train_val,
            'chi_square_train_test': chi_square_train_test,
            'chi_square_val_test': chi_square_val_test,
            'max_chi_square': stratification_quality,
            'stratification_quality': quality_rating
        }
    
    @staticmethod
    def check_minimum_samples_per_class(y: np.ndarray, min_samples: int = 10) -> Dict[str, Any]:
        """
        Check if all classes have minimum number of samples.
        Important for training stability.
        """
        classes, counts = np.unique(y, return_counts=True)
        
        below_minimum = {}
        for cls, count in zip(classes, counts):
            if count < min_samples:
                below_minimum[cls] = {
                    'count': count,
                    'required': min_samples,
                    'deficit': min_samples - count
                }
        
        return {
            'min_samples_required': min_samples,
            'classes_below_minimum': len(below_minimum),
            'details': below_minimum,
            'all_classes_sufficient': len(below_minimum) == 0
        }


def _compute_skewness(x):
    """Compute skewness manually."""
    mean = x.mean()
    std = x.std()
    if std == 0:
        return 0
    return np.mean(((x - mean) / std) ** 3)


def _compute_kurtosis(x):
    """Compute kurtosis manually."""
    mean = x.mean()
    std = x.std()
    if std == 0:
        return 0
    return np.mean(((x - mean) / std) ** 4) - 3


def _compute_chi_square_distance(dist1: Dict, dist2: Dict, all_classes: set) -> float:
    """Compute chi-square distance between two distributions."""
    total_distance = 0
    for cls in all_classes:
        p1 = dist1.get(cls, 0) / 100
        p2 = dist2.get(cls, 0) / 100
        
        denom = p1 + p2
        if denom > 0:
            total_distance += (p1 - p2) ** 2 / denom
    
    return total_distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate preprocessed data for leakage and distribution")
    parser.add_argument("--train", type=pathlib.Path, required=True, help="Train npz file")
    parser.add_argument("--val", type=pathlib.Path, required=True, help="Validation npz file")
    parser.add_argument("--test", type=pathlib.Path, required=True, help="Test npz file")
    parser.add_argument("--original", type=pathlib.Path, help="Original unprocessed data for comparison")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/validation_report.txt"),
                       help="Output report file")
    
    args = parser.parse_args()
    
    # Load data
    print("=" * 70)
    print("DATA VALIDATION AND INTEGRITY CHECK")
    print("=" * 70)
    
    train_data = np.load(args.train)
    val_data = np.load(args.val)
    test_data = np.load(args.test)
    
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"\nLoaded data:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test: X={X_test.shape}, y={y_test.shape}")
    
    report = []
    
    # ========== DATA LEAKAGE DETECTION ==========
    print("\n" + "=" * 70)
    print("1. DATA LEAKAGE DETECTION")
    print("=" * 70)
    report.append("\n" + "=" * 70)
    report.append("1. DATA LEAKAGE DETECTION")
    report.append("=" * 70)
    
    detector = DataLeakageDetector()
    
    # Check train-test leakage
    leakage_train_test = detector.check_identical_samples(X_train, X_test)
    print(f"\nTrain-Test Identical Samples: {leakage_train_test['identical_samples']}")
    print(f"Train-Test Similar Samples: {leakage_train_test['similar_samples']}")
    print(f"Leakage Risk: {leakage_train_test['leakage_risk']}")
    
    report.append(f"\nTrain-Test Identical Samples: {leakage_train_test['identical_samples']}")
    report.append(f"Train-Test Similar Samples: {leakage_train_test['similar_samples']}")
    report.append(f"Leakage Risk: {leakage_train_test['leakage_risk']}")
    
    # Check train-val leakage
    leakage_train_val = detector.check_identical_samples(X_train, X_val)
    print(f"\nTrain-Val Identical Samples: {leakage_train_val['identical_samples']}")
    print(f"Train-Val Similar Samples: {leakage_train_val['similar_samples']}")
    print(f"Leakage Risk: {leakage_train_val['leakage_risk']}")
    
    report.append(f"\nTrain-Val Identical Samples: {leakage_train_val['identical_samples']}")
    report.append(f"Train-Val Similar Samples: {leakage_train_val['similar_samples']}")
    report.append(f"Leakage Risk: {leakage_train_val['leakage_risk']}")
    
    # Check class label consistency
    label_overlap = detector.check_statistical_overlap(y_train, y_test)
    print(f"\nClass Label Overlap (Train-Test):")
    print(f"  Train classes: {label_overlap['train_classes']}")
    print(f"  Test classes: {label_overlap['test_classes']}")
    print(f"  Seen in train: {label_overlap['seen_in_train']}")
    print(f"  Unforeseen: {label_overlap['unforeseen_classes']}")
    print(f"  Coverage: {label_overlap['class_coverage']*100:.1f}%")
    
    report.append(f"\nClass Label Overlap (Train-Test):")
    report.append(f"  Train classes: {label_overlap['train_classes']}")
    report.append(f"  Test classes: {label_overlap['test_classes']}")
    report.append(f"  Seen in train: {label_overlap['seen_in_train']}")
    report.append(f"  Unforeseen: {label_overlap['unforeseen_classes']}")
    report.append(f"  Coverage: {label_overlap['class_coverage']*100:.1f}%")
    
    # Check feature value overlap
    feature_overlap = detector.check_feature_overlap(X_train, X_test)
    print(f"\nFeature Value Overlap (Train-Test):")
    print(f"  Total features: {feature_overlap['total_features']}")
    print(f"  Features below min: {feature_overlap['features_below_min']}")
    print(f"  Features above max: {feature_overlap['features_above_max']}")
    print(f"  Extrapolation risk: {feature_overlap['extrapolation_risk']*100:.1f}%")
    
    report.append(f"\nFeature Value Overlap (Train-Test):")
    report.append(f"  Total features: {feature_overlap['total_features']}")
    report.append(f"  Features below min: {feature_overlap['features_below_min']}")
    report.append(f"  Features above max: {feature_overlap['features_above_max']}")
    report.append(f"  Extrapolation risk: {feature_overlap['extrapolation_risk']*100:.1f}%")
    
    # ========== DISTRIBUTION ANALYSIS ==========
    print("\n" + "=" * 70)
    print("2. DISTRIBUTION ANALYSIS")
    print("=" * 70)
    report.append("\n" + "=" * 70)
    report.append("2. DISTRIBUTION ANALYSIS")
    report.append("=" * 70)
    
    analyzer = DistributionAnalyzer()
    
    # Analyze training set distribution
    train_stats = analyzer.compute_statistics(X_train)
    print(f"\nTraining Set Statistics (first 10 features):")
    print(train_stats.head(10).to_string(index=False))
    
    report.append(f"\nTraining Set Statistics (first 10 features):")
    report.append(train_stats.head(10).to_string(index=False))
    
    # Check for outliers
    outliers = analyzer.detect_outliers(X_train, method='iqr', threshold=1.5)
    print(f"\nOutlier Detection (IQR method):")
    print(f"  Total outliers: {outliers['total_outliers']}")
    print(f"  Outlier percentage: {outliers['outlier_percentage']:.3f}%")
    print(f"  Features with most outliers:")
    
    sorted_features = sorted(outliers['feature_outlier_counts'].items(), 
                            key=lambda x: x[1]['count'], reverse=True)
    for feat, info in sorted_features[:5]:
        print(f"    {feat}: {info['count']} ({info['percentage']:.2f}%)")
    
    report.append(f"\nOutlier Detection (IQR method):")
    report.append(f"  Total outliers: {outliers['total_outliers']}")
    report.append(f"  Outlier percentage: {outliers['outlier_percentage']:.3f}%")
    
    # ========== STRATIFICATION VALIDATION ==========
    print("\n" + "=" * 70)
    print("3. STRATIFICATION VALIDATION")
    print("=" * 70)
    report.append("\n" + "=" * 70)
    report.append("3. STRATIFICATION VALIDATION")
    report.append("=" * 70)
    
    validator = StratificationValidator()
    
    # Check stratification
    stratification = validator.check_stratification(y_train, y_val, y_test)
    print(f"\nClass Distribution Stratification Quality: {stratification['stratification_quality']}")
    print(f"  Max Chi-Square Distance: {stratification['max_chi_square']:.4f}")
    print(f"  Train-Val Chi-Square: {stratification['chi_square_train_val']:.4f}")
    print(f"  Train-Test Chi-Square: {stratification['chi_square_train_test']:.4f}")
    print(f"  Val-Test Chi-Square: {stratification['chi_square_val_test']:.4f}")
    
    report.append(f"\nClass Distribution Stratification Quality: {stratification['stratification_quality']}")
    report.append(f"  Max Chi-Square Distance: {stratification['max_chi_square']:.4f}")
    
    print(f"\nClass Distributions by Split:")
    print(f"\nTrain Distribution:")
    for cls, pct in sorted(stratification['train_distribution'].items()):
        print(f"  Class {cls}: {pct:.2f}%")
    
    print(f"\nValidation Distribution:")
    for cls, pct in sorted(stratification['val_distribution'].items()):
        print(f"  Class {cls}: {pct:.2f}%")
    
    print(f"\nTest Distribution:")
    for cls, pct in sorted(stratification['test_distribution'].items()):
        print(f"  Class {cls}: {pct:.2f}%")
    
    report.append(f"\nClass Distributions by Split:")
    report.append(f"Train: {stratification['train_distribution']}")
    report.append(f"Val: {stratification['val_distribution']}")
    report.append(f"Test: {stratification['test_distribution']}")
    
    # Check minimum samples per class
    min_samples_train = validator.check_minimum_samples_per_class(y_train, min_samples=5)
    min_samples_val = validator.check_minimum_samples_per_class(y_val, min_samples=5)
    min_samples_test = validator.check_minimum_samples_per_class(y_test, min_samples=5)
    
    print(f"\nMinimum Samples Per Class (threshold=5):")
    print(f"  Train: {'✓ PASS' if min_samples_train['all_classes_sufficient'] else '✗ FAIL'}")
    print(f"  Val: {'✓ PASS' if min_samples_val['all_classes_sufficient'] else '✗ FAIL'}")
    print(f"  Test: {'✓ PASS' if min_samples_test['all_classes_sufficient'] else '✗ FAIL'}")
    
    report.append(f"\nMinimum Samples Per Class (threshold=5):")
    report.append(f"  Train: {'✓ PASS' if min_samples_train['all_classes_sufficient'] else '✗ FAIL'}")
    report.append(f"  Val: {'✓ PASS' if min_samples_val['all_classes_sufficient'] else '✗ FAIL'}")
    report.append(f"  Test: {'✓ PASS' if min_samples_test['all_classes_sufficient'] else '✗ FAIL'}")
    
    # ========== VALIDATION SUMMARY ==========
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    report.append("\n" + "=" * 70)
    report.append("VALIDATION SUMMARY")
    report.append("=" * 70)
    
    all_checks = {
        'Data Leakage (Train-Test)': leakage_train_test['leakage_risk'] == 'LOW',
        'Data Leakage (Train-Val)': leakage_train_val['leakage_risk'] == 'LOW',
        'Class Coverage': label_overlap['class_coverage'] > 0.95,
        'Stratification': stratification['stratification_quality'] in ['EXCELLENT', 'GOOD'],
        'Min Samples (Train)': min_samples_train['all_classes_sufficient'],
        'Min Samples (Val)': min_samples_val['all_classes_sufficient'],
        'Min Samples (Test)': min_samples_test['all_classes_sufficient'],
    }
    
    print("\nValidation Checks:")
    for check, result in all_checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {check:.<40} {status}")
        report.append(f"  {check:.<40} {status}")
    
    overall = all(all_checks.values())
    overall_status = "✓ ALL CHECKS PASSED" if overall else "✗ SOME CHECKS FAILED"
    print(f"\nOverall Status: {overall_status}")
    report.append(f"\nOverall Status: {overall_status}")
    
    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n✓ Validation report saved to {args.output}")


if __name__ == "__main__":
    main()
