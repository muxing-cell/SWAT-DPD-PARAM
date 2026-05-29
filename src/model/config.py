from dataclasses import dataclass

@dataclass
class SWATConfig:
    # --- 模型架构参数 (已提升容量) ---
    in_dim: int = 2
    embed_dim: int = 64       # 特征维度提升，增强非线性表征能力
    kernel_sizes: tuple = (3, 5, 7)
    num_heads: int = 4        # 注意力头数增加
    num_layers: int = 4       # 网络深度增加
    out_dim: int = 2
    dropout: float = 0.1
    window_size: int = 10
    lambda_f: float = 0.5
    
    # --- 训练超参数 ---
    batch_size: int = 128
    seq_len: int = 64
    lr: float = 1e-3
    epochs: int = 1000        # 延长训练周期
    device: str = "cuda"

    # --- 数据集全量加载配置 ---
    data_base_dir: str = r"C:\Users\m3xin\Desktop\SWAT-DPD\dataset\DATA_W_D_H_band\Baseband_input_output" 
    band: str = "D_band"