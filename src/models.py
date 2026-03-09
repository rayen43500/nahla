"""
Models Module - IoT Network Intrusion Detection System

Implements:
- MLP (baseline)
- LSTM / Bidirectional LSTM
- 1D CNN (multi-kernel)
- Autoencoder (anomaly detection)
- Hybrid CNN-LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# MLP - Multi-Layer Perceptron (baseline, Phase 3)
# ============================================================================

class MLP(nn.Module):
    """Simple MLP for network traffic classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# LSTM - Long Short-Term Memory (Phase 7.1)
# ============================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM classifier for sequential network packet data.

    Supports:
    - 2+ stacked LSTM layers
    - Bidirectional mode (BLSTM)
    - Dropout + BatchNorm
    - Stateful option for continuous stream processing
    - Variable-length packet sequences
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        stateful: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.stateful = stateful
        self.num_directions = 2 if bidirectional else 1

        self.batch_norm_input = nn.BatchNorm1d(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * self.num_directions

        self.batch_norm = nn.BatchNorm1d(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Hidden state for stateful mode
        self._h = None
        self._c = None

    def reset_hidden(self):
        """Reset hidden state (for stateful mode between batches)."""
        self._h = None
        self._c = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
               If 2D, treated as single-step sequence.
        """
        if x.dim() == 2:
            # Apply batch norm on feature dim
            x = self.batch_norm_input(x)
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        else:
            # Apply batch norm across features for each timestep
            b, s, f = x.shape
            x = self.batch_norm_input(x.reshape(-1, f)).reshape(b, s, f)

        if self.stateful and self._h is not None:
            h0 = self._h.detach()
            c0 = self._c.detach()
        else:
            batch_size = x.size(0)
            h0 = torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                device=x.device,
            )
            c0 = torch.zeros_like(h0)

        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        if self.stateful:
            self._h = hn
            self._c = cn

        # Use last timestep output
        last_out = lstm_out[:, -1, :]  # (batch, hidden*directions)

        last_out = self.batch_norm(last_out)
        last_out = self.dropout(last_out)

        return self.classifier(last_out)


# ============================================================================
# CNN - 1D Convolutional Neural Network (Phase 7.2)
# ============================================================================

class CNN1DClassifier(nn.Module):
    """
    1D CNN for network traffic pattern detection.

    Supports:
    - Multiple kernel sizes for multi-scale pattern detection
    - MaxPooling + GlobalAveragePooling
    - Can reshape flat features into 2D matrices for Conv2D
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Multiple parallel conv branches with different kernel sizes
        self.conv_branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(1, num_filters, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters * 2, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(num_filters * 2),
                nn.ReLU(),
            )
            self.conv_branches.append(branch)

        total_filters = num_filters * 2 * len(kernel_sizes)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Combine avg + max pooling
        combined_dim = total_filters * 2

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) flat features or (batch, 1, input_dim) 1D signal
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        branch_outputs = []
        for branch in self.conv_branches:
            out = branch(x)  # (batch, filters*2, seq_len)
            avg_pool = self.global_avg_pool(out).squeeze(-1)
            max_pool = self.global_max_pool(out).squeeze(-1)
            branch_outputs.append(avg_pool)
            branch_outputs.append(max_pool)

        combined = torch.cat(branch_outputs, dim=1)
        return self.classifier(combined)


# ============================================================================
# Autoencoder (Phase 7.3)
# ============================================================================

class Autoencoder(nn.Module):
    """
    Symmetric Autoencoder for anomaly detection.

    Normal traffic is learned during training; high reconstruction error
    indicates anomalous (attack) traffic.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (symmetric)
        decoder_layers = []
        prev_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error (MSE)."""
        x_hat = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)

    def predict_anomaly(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Predict anomaly: 1 if reconstruction error > threshold, else 0."""
        errors = self.reconstruction_error(x)
        return (errors > threshold).long()


# ============================================================================
# Hybrid CNN-LSTM (Phase 7.4)
# ============================================================================

class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model.

    - CNN extracts local spatial features from network traffic
    - LSTM captures temporal dependencies across packet sequences
    - Fusion before final classification
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # LSTM for temporal patterns (processes CNN output as sequence)
        cnn_out_dim = cnn_filters * 2
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * self.num_directions

        self.batch_norm = nn.BatchNorm1d(lstm_out_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        # CNN: (batch, channels, seq_len) -> (batch, cnn_filters*2, seq_len//2)
        cnn_out = self.cnn(x)

        # Transpose for LSTM: (batch, seq_len//2, cnn_filters*2)
        lstm_input = cnn_out.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)
        last_out = lstm_out[:, -1, :]

        last_out = self.batch_norm(last_out)
        last_out = self.dropout_layer(last_out)

        return self.classifier(last_out)


# ============================================================================
# Factory function
# ============================================================================

def create_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_type: One of 'mlp', 'lstm', 'cnn', 'autoencoder', 'hybrid'
        input_dim: Number of input features
        num_classes: Number of output classes
        **kwargs: Model-specific parameters

    Returns:
        nn.Module instance
    """
    model_map = {
        "mlp": MLP,
        "lstm": LSTMClassifier,
        "cnn": CNN1DClassifier,
        "autoencoder": Autoencoder,
        "hybrid": HybridCNNLSTM,
    }

    model_type = model_type.lower()
    if model_type not in model_map:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(model_map.keys())}")

    cls = model_map[model_type]

    if model_type == "autoencoder":
        return cls(input_dim=input_dim, **kwargs)
    else:
        return cls(input_dim=input_dim, num_classes=num_classes, **kwargs)
