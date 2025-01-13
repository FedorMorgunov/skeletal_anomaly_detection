import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding using sine/cosine functions.
    This helps the model understand temporal order.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding up to the current sequence length
        x = x + self.pe[:, :seq_len, :]
        return x

class SkeletonTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        """
        Args:
            input_size (int): Dimensionality of the input per frame (e.g., max_persons * 34)
            d_model (int): Dimension of the transformer embeddings
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dropout (float): Dropout rate
        """
        super(SkeletonTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model

        # 1) Linear projection to transformer embedding size
        self.embedding = nn.Linear(input_size, d_model)

        # 2) Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=256, 
                                                   dropout=dropout, 
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        # 4) Final classification head: we output a single score
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        """
        # 1) Project input to d_model
        x = self.embedding(x)  # (batch_size, seq_len, d_model)

        # 2) Add positional encoding
        x = self.pos_encoder(x)  # (batch_size, seq_len, d_model)

        # 3) Pass through Transformer encoder
        # The output shape is (batch_size, seq_len, d_model)
        encoded = self.transformer_encoder(x)

        # 4) We can use the last token's embedding or pool over all tokens
        # Option A: Use last token as summary
        last_token = encoded[:, -1, :]  # (batch_size, d_model)

        # Option B: Mean-pool over tokens
        # last_token = encoded.mean(dim=1)

        # 5) Classification head
        out = self.fc(last_token)  # (batch_size, 1)
        return out.squeeze(1)      # shape: (batch_size,)