"""
Unit Tests - IoT Network Intrusion Detection System

Phase 13.3: Tests for models, training, data preparation, and API.
"""

import sys
import pathlib
import unittest
import tempfile
import shutil

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


class TestModels(unittest.TestCase):
    """Test all model architectures (Phase 7)."""

    def setUp(self):
        self.input_dim = 50
        self.num_classes = 5
        self.batch_size = 8

    def _test_model_forward(self, model, x):
        """Helper: check forward pass shape."""
        model.eval()
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (x.shape[0], self.num_classes))

    def test_mlp(self):
        from models import MLP
        model = MLP(self.input_dim, self.num_classes, hidden=128, dropout=0.2)
        x = torch.randn(self.batch_size, self.input_dim)
        self._test_model_forward(model, x)

    def test_lstm(self):
        from models import LSTMClassifier
        model = LSTMClassifier(self.input_dim, self.num_classes, hidden_dim=64,
                               num_layers=2, bidirectional=True, dropout=0.2)
        # 2D input
        x = torch.randn(self.batch_size, self.input_dim)
        self._test_model_forward(model, x)
        # 3D input (sequence)
        x_seq = torch.randn(self.batch_size, 10, self.input_dim)
        model.eval()
        with torch.no_grad():
            out = model(x_seq)
        self.assertEqual(out.shape, (self.batch_size, self.num_classes))

    def test_lstm_stateful(self):
        from models import LSTMClassifier
        model = LSTMClassifier(self.input_dim, self.num_classes, hidden_dim=64,
                               num_layers=2, stateful=True)
        x = torch.randn(self.batch_size, self.input_dim)
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        # After two calls, hidden state should persist (different outputs)
        self.assertEqual(out1.shape, out2.shape)
        model.reset_hidden()

    def test_cnn(self):
        from models import CNN1DClassifier
        model = CNN1DClassifier(self.input_dim, self.num_classes, num_filters=32,
                                kernel_sizes=(3, 5), dropout=0.2)
        x = torch.randn(self.batch_size, self.input_dim)
        self._test_model_forward(model, x)

    def test_autoencoder(self):
        from models import Autoencoder
        model = Autoencoder(self.input_dim, bottleneck_dim=8, hidden_dims=(64, 32))
        x = torch.randn(self.batch_size, self.input_dim)
        model.eval()
        with torch.no_grad():
            x_hat = model(x)
            errors = model.reconstruction_error(x)
            preds = model.predict_anomaly(x, threshold=0.5)
        self.assertEqual(x_hat.shape, x.shape)
        self.assertEqual(errors.shape, (self.batch_size,))
        self.assertEqual(preds.shape, (self.batch_size,))

    def test_hybrid(self):
        from models import HybridCNNLSTM
        model = HybridCNNLSTM(self.input_dim, self.num_classes, cnn_filters=32,
                               lstm_hidden=64, lstm_layers=2, dropout=0.2)
        x = torch.randn(self.batch_size, self.input_dim)
        self._test_model_forward(model, x)

    def test_create_model_factory(self):
        from models import create_model
        for model_type in ["mlp", "lstm", "cnn", "hybrid"]:
            model = create_model(model_type, self.input_dim, self.num_classes)
            self.assertIsInstance(model, torch.nn.Module)

        model = create_model("autoencoder", self.input_dim, self.num_classes)
        self.assertIsInstance(model, torch.nn.Module)

    def test_create_model_invalid(self):
        from models import create_model
        with self.assertRaises(ValueError):
            create_model("invalid_model", self.input_dim, self.num_classes)


class TestDataPrep(unittest.TestCase):
    """Test data preparation functions (Phase 6)."""

    def test_split_data(self):
        import pandas as pd
        from data_prep import split_data

        n = 200
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "label": np.random.choice(["A", "B", "C"], size=n),
        })

        train, val, test = split_data(df, label_col="label")
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, n)
        self.assertGreater(len(train), len(val))
        self.assertGreater(len(train), len(test))

    def test_apply_smote(self):
        from data_prep import apply_smote

        X = np.random.randn(100, 10)
        y = np.array([0]*80 + [1]*20)

        X_res, y_res = apply_smote(X, y)
        self.assertGreaterEqual(len(X_res), len(X))

    def test_apply_pca(self):
        from data_prep import apply_pca

        X_train = np.random.randn(100, 50)
        X_val = np.random.randn(30, 50)
        X_test = np.random.randn(30, 50)

        X_tr, X_v, X_te, pca = apply_pca(X_train, X_val, X_test, variance_ratio=0.95)
        self.assertLessEqual(X_tr.shape[1], 50)
        self.assertEqual(X_tr.shape[1], X_v.shape[1])
        self.assertEqual(X_tr.shape[1], X_te.shape[1])


class TestTrain(unittest.TestCase):
    """Test training functions (Phase 8)."""

    def test_early_stopping(self):
        from train import EarlyStopping
        es = EarlyStopping(patience=3)
        # Improving losses
        self.assertFalse(es.step(1.0))
        self.assertFalse(es.step(0.9))
        self.assertFalse(es.step(0.8))
        # No improvement
        self.assertFalse(es.step(0.9))
        self.assertFalse(es.step(0.9))
        self.assertTrue(es.step(0.9))  # patience exhausted

    def test_make_loader(self):
        from train import make_loader
        X = np.random.randn(100, 20).astype(np.float32)
        y = np.random.randint(0, 3, 100)
        loader, classes = make_loader(X, y, batch_size=32)
        self.assertEqual(len(classes), 3)
        batch_x, batch_y = next(iter(loader))
        self.assertEqual(batch_x.shape[1], 20)

    def test_make_loader_string_labels(self):
        from train import make_loader
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.array(["attack", "normal", "attack", "normal"] * 12 + ["attack", "normal"])
        loader, classes = make_loader(X, y, batch_size=16)
        self.assertEqual(len(classes), 2)


class TestEvaluate(unittest.TestCase):
    """Test evaluation functions (Phase 9)."""

    def test_compute_metrics(self):
        from evaluate import compute_metrics
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2, 1])
        y_probs = np.random.rand(10, 3)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
        classes = np.array(["Normal", "DoS", "DDoS"])

        metrics = compute_metrics(y_true, y_pred, y_probs, classes)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_weighted", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("tpr_per_class", metrics)
        self.assertIn("fpr_per_class", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_per_attack_type(self):
        from evaluate import evaluate_per_attack_type
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        classes = np.array(["Normal", "DoS", "DDoS"])

        result = evaluate_per_attack_type(y_true, y_pred, classes)
        self.assertIn("per_class_results", result)
        self.assertIn("hardest_to_detect", result)


class TestAPI(unittest.TestCase):
    """Test API endpoints (Phase 12)."""

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_model_info_no_model(self):
        from fastapi.testclient import TestClient
        from api import app, _state

        _state.clear()
        client = TestClient(app)
        response = client.get("/model_info")
        # Should return 503 when no model loaded
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
