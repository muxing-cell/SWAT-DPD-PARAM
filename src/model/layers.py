import torch
import torch.nn as nn

def get_sliding_window_mask(seq_len, window_size, device="cpu"):
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 0.0
    return mask

class SignalAdaIN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-6
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2)
        )

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, unbiased=False, keepdim=True) + self.eps
        x_norm = (x - mu) / std
        
        stats = torch.cat([mu, std], dim=-1)
        params = self.mlp(stats)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        
        return gamma * x_norm + beta

class WindowedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_out))