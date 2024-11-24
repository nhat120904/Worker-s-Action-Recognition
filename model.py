import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import math 

class TransformerClassifier(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 60
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Layer norm is preferred over batch norm for transformers
        
        # Improved positional encoding with register_buffer instead of Parameter
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, max_seq_length, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pe)
        
        # Add dropout after input embedding
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important for newer PyTorch versions
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add layer norm before final classification
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        
        # Input embedding
        x = self.linear(x)
        x = self.layer_norm(x)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)
        
        # Transformer layers
        x = self.transformer(x)
        
        # Global average pooling and classification
        x = x.mean(dim=1)  # [batch_size, hidden_dim]
        x = self.final_layer_norm(x)
        x = self.fc(x)
        
        return x
    
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=9):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        features = lstm_out.mean(dim=1)
        return self.fc(features)