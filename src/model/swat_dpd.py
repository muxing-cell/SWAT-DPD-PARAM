import torch
import torch.nn as nn
from layers import get_sliding_window_mask, SignalAdaIN, WindowedTransformerLayer

class SWAT_DPD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.window_size = cfg.window_size
        
        # --- 1. 带有空洞率的多尺度卷积前端 ---
   
        dilations = getattr(cfg, 'dilations', [1] * len(cfg.kernel_sizes))
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=cfg.in_dim,        # 此时 cfg.in_dim 应该是 5
                out_channels=cfg.embed_dim, 
                kernel_size=k, 
                padding=(k - 1) * d // 2,      # 动态计算 padding，确保引入空洞率后序列长度不发生突变
                dilation=d
            ) 
            for k, d in zip(cfg.kernel_sizes, dilations)
        ])
        
        d_model = cfg.embed_dim * len(cfg.kernel_sizes)
        self.adain = SignalAdaIN(d_model)
        
        # --- 2. 外部参数调制模块 (FiLM) ---
       
        param_dim = getattr(cfg, 'param_dim', 2)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(param_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(param_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # --- 3. 滑窗 Transformer 层 ---
        self.layers = nn.ModuleList([
            WindowedTransformerLayer(d_model, cfg.num_heads, d_model * 2, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        # --- 4. 平滑回归 ---
        
        self.regressor = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, cfg.out_dim, kernel_size=3, padding=1) # out_dim 为 2
        )

    def forward(self, x, params):
        """
        x shape: [B, L, 5] (real, imag, mod, square, cubic)
        params shape: [B, 2] (avg_power, papr)
        """
        B, L, _ = x.size()
        
        # --- 信号编码 ---
        x_p = x.permute(0, 2, 1) # [B, 5, L]
        x_enc = torch.cat([conv(x_p) for conv in self.convs], dim=1).permute(0, 2, 1)
        
        # --- 归一化与调制 ---
        x_enc = self.adain(x_enc)
        
        # FiLM 机制：生成缩放和平移因子，维度扩展为 [B, 1, d_model] 以适配序列长度
        gamma = self.gamma_mlp(params).unsqueeze(1) 
        beta = self.beta_mlp(params).unsqueeze(1)    
        x_enc = gamma * x_enc + beta
        
        # --- 时序注意力计算 ---
        mask = get_sliding_window_mask(L, self.window_size, device=x.device)
        for layer in self.layers:
            x_enc = layer(x_enc, mask=mask)

        # --- 回归与残差 ---
        x_enc_p = x_enc.permute(0, 2, 1)
        residual = self.regressor(x_enc_p).permute(0, 2, 1) # shape: [B, L, 2]

      
        base_signal = x[..., 0:2] 

        return base_signal + residual