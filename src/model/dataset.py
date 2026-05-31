import os
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

class DPDDataset(Dataset):
    def __init__(self, base_dir, seq_len):
        self.seq_len = seq_len
        
        # 定义全局遍历的全集工况 (与 ARVMCTN 完全对齐)
        BANDS = ["D", "H", "W"]
        GAINS = ["1", "2", "4"]
        QAMS = ["16", "64"]

        all_x_chunks = []
        all_y_chunks = []
        all_p_chunks = [] # 用于存储 params 条件
        
        print(f"[*] 开始启动大一统全集数据加载管线 (1 对 5 增强模式)...")
        
       
        for i, band in enumerate(BANDS):
            for j, g in enumerate(GAINS):
                for k, qam in enumerate(QAMS):
                    
                    # 构建输入路径
                    input_dir = os.path.join(base_dir, "input", f"{band}_band")
                    output_dir = os.path.join(base_dir, "output", f"{band}_band")
                    
                    in_filename = f"PA_input_BS_{g}G_60G_{qam}QAM_{band}.mat"
                    in_path = os.path.join(input_dir, in_filename)
                    
                    if not os.path.exists(in_path):
                        continue
                        
                    # 加载单条输入复数序列
                    x_complex = self._load_mat_data(in_path)
                    
                    # 【核心突破】：1 个 Input 对应 5 个 Output，数据量直接翻 5 倍
                    for idx in range(1, 6):
                        out_filename = f"PA_baseband_{g}G_{qam}QAM_{band}_{idx}.mat"
                        out_path = os.path.join(output_dir, out_filename)
                        
                        if not os.path.exists(out_path):
                            continue
                            
                        y_complex = self._load_mat_data(out_path)
                        
                        # 序列截断分块
                        num_chunks = min(len(x_complex), len(y_complex)) // seq_len
                        if num_chunks == 0: continue
                        
                        x_chopped = x_complex[:num_chunks * seq_len].reshape(num_chunks, seq_len)
                        y_chopped = y_complex[:num_chunks * seq_len].reshape(num_chunks, seq_len)
                        
                        all_x_chunks.append(x_chopped)
                        all_y_chunks.append(y_chopped)
                        
                        # === 生成全局条件 Param ===
                        # 将 i, j, k 归一化到 [0, 1] 区间，这是给神经网络 MLP 最友好的数值尺度
                        p_vec = np.array([
                            i / (len(BANDS) - 1),  # Band: 0, 0.5, 1
                            j / (len(GAINS) - 1),  # Gain: 0, 0.5, 1
                            k / (len(QAMS) - 1)    # QAM:  0, 1
                        ], dtype=np.float32)
                        
                        # 沿样本数量维度复制
                        p_chopped = np.tile(p_vec, (num_chunks, 1))
                        all_p_chunks.append(p_chopped)

        if not all_x_chunks:
            raise RuntimeError("未能成功加载任何数据，请检查 data_base_dir 路径配置。")
            
        # 数据拼接
        x_all = np.concatenate(all_x_chunks, axis=0)
        y_all = np.concatenate(all_y_chunks, axis=0)
        p_all = np.concatenate(all_p_chunks, axis=0)
        
        x_iq = np.stack((np.real(x_all), np.imag(x_all)), axis=-1)
        y_iq = np.stack((np.real(y_all), np.imag(y_all)), axis=-1)
        
        self.x_data = torch.from_numpy(x_iq).float()
        self.y_data = torch.from_numpy(y_iq).float()
        self.params_data = torch.from_numpy(p_all).float() # [N, 3]

        # === 特征工程：构建 5 维物理先验特征 ===
        real = self.x_data[..., 0:1]  
        imag = self.x_data[..., 1:2]
        
        mod_square = real**2 + imag**2                 # 瞬时功率
        mod = torch.sqrt(mod_square + 1e-8)            # 幅值包络
        mod_cubic = mod_square * mod                   # 高阶包络
        
        # 拼接后 x_data 的 shape 变为 [N, seq_len, 5]
        self.x_data = torch.cat([real, imag, mod, mod_square, mod_cubic], dim=-1)

        # === 全局 Z-Score 标准化 ===
        self.x_mean = torch.mean(self.x_data, dim=(0, 1), keepdim=True)
        self.x_std = torch.std(self.x_data, dim=(0, 1), keepdim=True)
        self.x_data = (self.x_data - self.x_mean) / (self.x_std + 1e-8)

        self.y_mean = torch.mean(self.y_data, dim=(0, 1), keepdim=True)
        self.y_std = torch.std(self.y_data, dim=(0, 1), keepdim=True)
        self.y_data = (self.y_data - self.y_mean) / (self.y_std + 1e-8)
        
        # === 防御：硬截断异常值 ===
        self.x_data = torch.clamp(self.x_data, min=-5.0, max=5.0)
        self.y_data = torch.clamp(self.y_data, min=-5.0, max=5.0)
        
        print(f"[*] 全局加载完毕！| 成功加载工况组合: 18 (1对5增强) | 总样本池: {len(self.x_data)}")
        print(f"[*] 特征维度: X:[..., 5], Y:[..., 2], Params:[..., 3]")

    def _load_mat_data(self, path):
        """辅助方法：读取 MAT 文件并展平为复数一维数组"""
        mat_dict = sio.loadmat(path)
        data_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        data = mat_dict[data_keys[0]]
        return data.flatten()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.params_data[idx]

def get_dataloader(cfg):
    
    dataset = DPDDataset(
        base_dir=cfg.data_base_dir,
        seq_len=cfg.seq_len
    )
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)