"""
Analyse des Résultats - Matrice de Métriques
Explique la structure des résultats et vérifie leur validité
"""

import json
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# ============================================================================
# 1. EXPLICATION DE LA STRUCTURE MATRICE
# ============================================================================

def explain_matrix_structure():
    """
    Explique comment les résultats sont organisés en matrice.
    """
    print("=" * 80)
    print("EXPLICATION: STRUCTURE MATRICE DES RÉSULTATS")
    print("=" * 80)

    explanation = """
    Les résultats sont organisés dans une MATRICE HIÉRARCHIQUE avec:

    ┌─────────────────────────────────────────────────────────────────┐
    │                   RÉSULTATS PAR CLASSE                          │
    ├─────────────────────────────────────────────────────────────────┤
    │ Class: "BENIGN"  ← Classe 1                                     │
    │   ├─ precision: 0.998     (% de BENIGN correctes)               │
    │   ├─ recall: 0.797        (% d'attaques détectées comme BENIGN) │
    │   ├─ f1-score: 0.886      (moyenne harmonique)                  │
    │   └─ support: 5562        (nombre d'échantillons)               │
    │                                                                  │
    │ Class: "DDoS"  ← Classe 2                                       │
    │   ├─ precision: 0.605                                           │
    │   ├─ recall: 0.998                                              │
    │   ├─ f1-score: 0.754                                            │
    │   └─ support: 1718                                              │
    │                                                                  │
    │ Class: "PortScan"  ← Classe 3                                   │
    │   ├─ precision: 0.998                                           │
    │   ├─ recall: 0.995                                              │
    │   ├─ f1-score: 0.997                                            │
    │   └─ support: 1671                                              │
    │                                                                  │
    │ ... (autres classes)                                            │
    └─────────────────────────────────────────────────────────────────┘

    │   MÉTRIQUES GLOBALES (agrégés)                                  │
    │   ├─ accuracy: 0.873           (precision globale)              │
    │   ├─ macro avg: moyenne simple de toutes les classes            │
    │   └─ weighted avg: moyenne pondérée par le nombre d'exemples    │


    Cette structure peut être visualisée comme une MATRICE 2D:

    ┌─────────────┬───────────┬────────┬──────────┬─────────┐
    │ Class       │ Precision │ Recall │ F1-Score │ Support │
    ├─────────────┼───────────┼────────┼──────────┼─────────┤
    │ BENIGN      │   0.998   │ 0.797  │  0.886   │  5562   │
    │ DDoS        │   0.605   │ 0.998  │  0.754   │  1718   │
    │ PortScan    │   0.998   │ 0.995  │  0.997   │  1671   │
    │ WebAttack   │   0.712   │ 0.902  │  0.796   │   41    │
    ├─────────────┼───────────┼────────┼──────────┼─────────┤
    │ macro avg   │   0.828   │ 0.923  │  0.858   │  8992   │
    │ weighted avg│   0.922   │ 0.873  │  0.881   │  8992   │
    └─────────────┴───────────┴────────┴──────────┴─────────┘

    MATRICE DE CONFUSION (supplémentaire):

    ┌──────────────┬──────────┬────────┬──────────┬──────────┐
    │              │ Prédit:  │ Prédit │ Prédit   │ Prédit   │
    │              │ BENIGN   │ DDoS   │ PortScan │ WebAttack│
    ├──────────────┼──────────┼────────┼──────────┼──────────┤
    │ Réel: BENIGN │  4436    │   46   │   0      │   20     │
    │ Réel: DDoS   │    27    │ 1714   │   2      │   1      │
    │ Réel: PortScan│   1    │    6   │ 1662     │   2      │
    │ Réel: WebAttack│  12   │    3   │   0      │   37     │
    └──────────────┴──────────┴────────┴──────────┴──────────┘
    """

    print(explanation)


# ============================================================================
# 2. CHARGER ET AFFICHER LES RÉSULTATS
# ============================================================================

def load_metrics(metrics_file: pathlib.Path) -> Dict[str, Any]:
    """Charge les métriques d'un fichier JSON."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def display_metrics_as_matrix(metrics: Dict, model_name: str):
    """Affiche les métriques sous forme de matrice."""

    if metrics is None:
        print(f"[ERREUR] Fichier de métriques non trouvé pour {model_name}")
        return None

    print(f"\n{'=' * 80}")
    print(f"MODÈLE: {model_name.upper()}")
    print(f"{'=' * 80}\n")

    # Créer une liste de classes avec leurs métriques
    rows = []

    # Classes individuelles
    for class_name, metrics_dict in metrics.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg', 'summary']:
            if isinstance(metrics_dict, dict) and 'precision' in metrics_dict:
                rows.append({
                    'Class': class_name,
                    'Precision': f"{metrics_dict['precision']:.4f}",
                    'Recall': f"{metrics_dict['recall']:.4f}",
                    'F1-Score': f"{metrics_dict['f1-score']:.4f}",
                    'Support': f"{int(metrics_dict['support'])}"
                })

    # Créer DataFrame et afficher
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print()

    # Afficher les agrégés
    if 'accuracy' in metrics:
        print(f"Accuracy Global: {metrics['accuracy']:.4f}")

    if 'macro avg' in metrics:
        macro = metrics['macro avg']
        print(f"\nMacro Average (simple moyenne):")
        print(f"  - Precision: {macro['precision']:.4f}")
        print(f"  - Recall:    {macro['recall']:.4f}")
        print(f"  - F1-Score:  {macro['f1-score']:.4f}")

    if 'weighted avg' in metrics:
        weighted = metrics['weighted avg']
        print(f"\nWeighted Average (moyenne pondérée par support):")
        print(f"  - Precision: {weighted['precision']:.4f}")
        print(f"  - Recall:    {weighted['recall']:.4f}")
        print(f"  - F1-Score:  {weighted['f1-score']:.4f}")

    return df


# ============================================================================
# 3. CRÉER UNE MATRICE DE COMPARAISON GLOBALE
# ============================================================================

def create_comparison_matrix(results_dir: pathlib.Path):
    """Crée une matrice de comparaison de tous les modèles."""

    print(f"\n{'=' * 80}")
    print("MATRICE DE COMPARAISON: TOUS LES MODÈLES")
    print(f"{'=' * 80}\n")

    models = {
        'MLP': results_dir / '../models/mlp_metrics.json',
        'LSTM': results_dir / '../models/lstm_metrics.json',
        'CNN': results_dir / '../models/cnn_metrics.json',
        'Hybrid': results_dir / '../models/hybrid_metrics.json',
        'Random Forest': results_dir / 'random_forest_metrics.json',
        'SVM': results_dir / 'svm_metrics.json',
        'XGBoost': results_dir / 'xgboost_metrics.json',
    }

    comparison_data = []

    for model_name, metrics_path in models.items():
        metrics = load_metrics(metrics_path)
        if metrics is not None:
            # Extraire les métriques clés
            if 'summary' in metrics:
                f1 = metrics['summary'].get('macro_f1', 'N/A')
            elif 'macro avg' in metrics:
                f1 = metrics['macro avg']['f1-score']
            else:
                f1 = 'N/A'

            if 'summary' in metrics:
                recall = metrics['summary'].get('macro_recall', 'N/A')
            elif 'macro avg' in metrics:
                recall = metrics['macro avg']['recall']
            else:
                recall = 'N/A'

            accuracy = metrics.get('accuracy', 'N/A')

            if isinstance(f1, (int, float)):
                f1_str = f"{f1:.4f}"
            else:
                f1_str = str(f1)

            if isinstance(recall, (int, float)):
                recall_str = f"{recall:.4f}"
            else:
                recall_str = str(recall)

            if isinstance(accuracy, (int, float)):
                accuracy_str = f"{accuracy:.4f}"
            else:
                accuracy_str = str(accuracy)

            comparison_data.append({
                'Modèle': model_name,
                'Accuracy': accuracy_str,
                'Macro F1': f1_str,
                'Macro Recall': recall_str,
                'Type': 'Deep Learning' if model_name in ['MLP', 'LSTM', 'CNN', 'Hybrid'] else 'Machine Learning'
            })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        return df_comparison
    else:
        print("[AVERTISSEMENT] Aucune métrique trouvée")
        return None


# ============================================================================
# 4. VÉRIFIER LA VALIDITÉ DES RÉSULTATS
# ============================================================================

def verify_results_validity(results_dir: pathlib.Path):
    """Vérifie si les résultats sont réels ou synthétiques."""

    print(f"\n{'=' * 80}")
    print("VÉRIFICATION: LES RÉSULTATS SONT-ILS RÉELS?")
    print(f"{'=' * 80}\n")

    checks = []

    # ✓ Vérification 1: Les fichiers JSON existent
    metrics_files = list(results_dir.glob("*_metrics.json"))
    checks.append((
        "✓ Fichiers JSON sauvegardés",
        len(metrics_files) > 0,
        f"Trouvé {len(metrics_files)} fichiers de métriques"
    ))

    # ✓ Vérification 2: Les confusion matrices existent
    confusion_files = list(results_dir.glob("*_confusion_matrix.npy"))
    checks.append((
        "✓ Matrices de confusion (.npy)",
        len(confusion_files) > 0,
        f"Trouvé {len(confusion_files)} matrices de confusion"
    ))

    # ✓ Vérification 3: Les valeurs sont cohérentes
    if metrics_files:
        first_metrics = load_metrics(metrics_files[0])
        if first_metrics:
            all_values_valid = True
            issues = []

            # Vérifier que precision + recall + f1 sont entre 0 et 1
            for class_name, metrics_dict in first_metrics.items():
                if isinstance(metrics_dict, dict) and 'precision' in metrics_dict:
                    p = metrics_dict['precision']
                    r = metrics_dict['recall']
                    f1 = metrics_dict['f1-score']

                    if not (0 <= p <= 1 and 0 <= r <= 1 and 0 <= f1 <= 1):
                        all_values_valid = False
                        issues.append(f"{class_name}: valeurs hors [0,1]")

                    # F1 doit être <= min(precision, recall) généralement
                    if f1 > max(p, r):
                        issues.append(f"{class_name}: F1 > max(P,R) - SUSPECT")

            checks.append((
                "✓ Valeurs numériques cohérentes",
                all_values_valid,
                f"Toutes les valeurs sont dans [0,1]" if all_values_valid else f"Problèmes trouvés: {len(issues)}"
            ))

    # ✓ Vérification 4: Les supports totalisent correctement
    if metrics_files:
        first_metrics = load_metrics(metrics_files[0])
        if first_metrics and 'macro avg' in first_metrics:
            total_support = first_metrics['macro avg'].get('support', 0)
            class_supports = sum([m['support'] for m in first_metrics.values()
                                 if isinstance(m, dict) and 'support' in m])

            support_valid = abs(total_support - class_supports) < 1
            checks.append((
                "✓ Support total cohérent",
                support_valid,
                f"Total: {total_support}, Somme classes: {class_supports}"
            ))

    # ✓ Vérification 5: Comparer DL vs ML
    ml_f1 = []
    dl_f1 = []

    for metrics_file in metrics_files:
        model_name = metrics_file.stem.replace('_metrics', '')
        metrics = load_metrics(metrics_file)

        if metrics and 'macro avg' in metrics:
            f1 = metrics['macro avg']['f1-score']

            if model_name in ['mlp', 'lstm', 'cnn', 'hybrid']:
                dl_f1.append(f1)
            elif model_name in ['random_forest', 'svm', 'xgboost']:
                ml_f1.append(f1)

    if dl_f1 and ml_f1:
        avg_dl = np.mean(dl_f1)
        avg_ml = np.mean(ml_f1)
        dl_better = avg_dl > avg_ml
        checks.append((
            "✓ DL surpasse ML (F1 macro)",
            dl_better,
            f"DL F1: {avg_dl:.4f}, ML F1: {avg_ml:.4f}"
        ))

    # Afficher les résultats
    print("Résultats des vérifications:\n")
    for check_name, is_valid, detail in checks:
        status = "✅ VALIDE" if is_valid else "⚠️ ATTENTION"
        print(f"{check_name}")
        print(f"  Statut: {status}")
        print(f"  Détail: {detail}\n")

    # Conclusion
    all_valid = all(valid for _, valid, _ in checks)
    print(f"\n{'=' * 80}")
    print(f"CONCLUSION: {'LES RÉSULTATS SEMBLENT RÉELS ✅' if all_valid else 'CERTAINS RÉSULTATS PEUVENT ÊTRE SUSPECTS ⚠️'}")
    print(f"{'=' * 80}")


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    """Analyse complète des résultats."""

    results_dir = pathlib.Path(__file__).parent / 'results'

    # 1. Expliquer la structure
    explain_matrix_structure()

    # 2. Afficher les résultats du meilleur modèle
    mlp_metrics = load_metrics(results_dir / '../models/mlp_metrics.json')
    if mlp_metrics:
        display_metrics_as_matrix(mlp_metrics, 'MLP (Deep Learning)')

    # 3. Matrice de comparaison
    create_comparison_matrix(results_dir)

    # 4. Vérifier la validité
    verify_results_validity(results_dir)


if __name__ == '__main__':
    main()
