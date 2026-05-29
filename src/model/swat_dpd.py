import torch
import torch.nn as nn
from layers import get_sliding_window_mask, SignalAdaIN, WindowedTransformerLayer

class SWAT_DPD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.window_size = cfg.window_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=cfg.in_dim, out_channels=cfg.embed_dim, 
                      kernel_size=k, padding=k//2) 
            for k in cfg.kernel_sizes
        ])
        
        d_model = cfg.embed_dim * len(cfg.kernel_sizes)
        self.adain = SignalAdaIN(d_model)
        
        self.layers = nn.ModuleList([
            WindowedTransformerLayer(d_model, cfg.num_heads, d_model * 2, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, cfg.out_dim)
        )

    def forward(self, x):
        B, L, _ = x.size()
        
        x_p = x.permute(0, 2, 1)
        x_enc = torch.cat([conv(x_p) for conv in self.convs], dim=1).permute(0, 2, 1)
        
        x_enc = self.adain(x_enc)
        
        mask = get_sliding_window_mask(L, self.window_size, device=x.device)
        for layer in self.layers:
            x_enc = layer(x_enc, mask=mask)

        return self.regressor(x_enc)