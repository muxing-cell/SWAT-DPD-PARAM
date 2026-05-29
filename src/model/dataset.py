import os
import re
import glob
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

class DPDDataset(Dataset):
    def __init__(self, base_dir, band, seq_len):
        self.seq_len = seq_len
        
        input_dir = os.path.join(base_dir, "input", band)
        output_dir = os.path.join(base_dir, "output", band)
        
        input_files = glob.glob(os.path.join(input_dir, "PA_input_*.mat"))
        if not input_files:
            raise FileNotFoundError(f"在 {input_dir} 文件夹下没有找到任何输入数据。")

        all_x_chunks = []
        all_y_chunks = []
        
        print(f"[*] 开始全量加载 {band} 数据...")
        
        for in_path in input_files:
            filename = os.path.basename(in_path)
            
            match = re.search(r'PA_input_BS_(.*?)_\d+G_(.*?)_(D|W|H)\.mat', filename)
            if not match:
                continue
                
            bw, qam, b_str = match.groups()
            out_filename = f"PA_baseband_{bw}_{qam}_{b_str}_1.mat"
            out_path = os.path.join(output_dir, out_filename)
            
            if not os.path.exists(out_path):
                continue
                
            x_complex = self._load_mat_data(in_path)
            y_complex = self._load_mat_data(out_path)
            
            num_chunks = min(len(x_complex), len(y_complex)) // seq_len
            if num_chunks == 0: continue
            
            x_chopped = x_complex[:num_chunks * seq_len].reshape(num_chunks, seq_len)
            y_chopped = y_complex[:num_chunks * seq_len].reshape(num_chunks, seq_len)
            
            all_x_chunks.append(x_chopped)
            all_y_chunks.append(y_chopped)

        if not all_x_chunks:
            raise RuntimeError("未能成功加载并配对任何数据，请检查数据目录。")
            
        x_all = np.concatenate(all_x_chunks, axis=0)
        y_all = np.concatenate(all_y_chunks, axis=0)
        
        x_iq = np.stack((np.real(x_all), np.imag(x_all)), axis=-1)
        y_iq = np.stack((np.real(y_all), np.imag(y_all)), axis=-1)
        
        self.x_data = torch.from_numpy(x_iq).float()
        self.y_data = torch.from_numpy(y_iq).float()

        # --- 新增：复数信号幅度全局归一化 ---
        x_complex_abs = torch.sqrt(self.x_data[..., 0]**2 + self.x_data[..., 1]**2)
        y_complex_abs = torch.sqrt(self.y_data[..., 0]**2 + self.y_data[..., 1]**2)
        
        self.x_max = torch.max(x_complex_abs)
        self.y_max = torch.max(y_complex_abs)
        
        self.x_data = self.x_data / self.x_max
        self.y_data = self.y_data / self.y_max
        # ------------------------------------
        
        print(f"[*] 全频段加载完毕 | 总工况数: {len(all_x_chunks)} | 总样本池: {len(self.x_data)}")
        print(f"[*] 归一化参数 | X_max: {self.x_max.item():.4e}, Y_max: {self.y_max.item():.4e}")

    def _load_mat_data(self, path):
        mat_dict = sio.loadmat(path)
        data_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        data = mat_dict[data_keys[0]]
        return data.flatten()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

def get_dataloader(cfg):
    dataset = DPDDataset(
        base_dir=cfg.data_base_dir,
        band=cfg.band,
        seq_len=cfg.seq_len
    )
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)