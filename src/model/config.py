from dataclasses import dataclass

@dataclass
class SWATConfig:

    in_dim: int = 5           
    embed_dim: int = 64       
    kernel_sizes: tuple = (3, 5, 7)
    dilations: tuple = (1, 2, 4)      
    num_heads: int = 4        
    num_layers: int = 4       
    out_dim: int = 2
    dropout: float = 0.1
    window_size: int = 10

    # 【核心修改 1】：条件维度必须从 2 升级为 3 
    # 严格对应 dataset 里的 [频段, 增益, 调制方式]
    param_dim: int = 3        
    
    batch_size: int = 128
    seq_len: int = 64
    lr: float = 1e-3
    
    # 提示：100轮用来跑通测试没问题。如果想冲击表格里 -24dB 的极限，
    # 建议测试通过后把 epochs 改回 1000 或 1400。
    epochs: int = 100        
    device: str = "cuda"
    
    T_0: int = 200            

    data_base_dir: str = r"C:\Users\m3xin\Desktop\SWAT-DPD\dataset\DATA_W_D_H_band\Baseband_input_output" 
    
    # 【核心修改 2】：彻底删除（或注释掉）单频段指定参数
    # 因为现在的 dataset 已经不需要这个参数了，它会自动遍历 D, H, W
    # band: str = "H_band"