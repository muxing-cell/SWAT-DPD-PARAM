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

   
    param_dim: int = 3        
    
    batch_size: int = 128
    seq_len: int = 64
    lr: float = 1e-3
    
   
    epochs: int = 100        
    device: str = "cuda"
    
    T_0: int = 200            

    data_base_dir: str = r"C:\Users\m3xin\Desktop\SWAT-DPD\dataset\DATA_W_D_H_band\Baseband_input_output" 
    
    