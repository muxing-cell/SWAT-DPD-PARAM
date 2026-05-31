import os
import torch
import torch.nn as nn
import numpy as np

from model.swat_dpd import SWAT_DPD
from dataset import get_dataloader
from trainer import Trainer
from output import OutputManager
from utils import calculate_nmse_np, plot_residual_analysis

class TrainPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataloader = get_dataloader(cfg)
        self.model = SWAT_DPD(cfg)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=cfg.lr, 
            weight_decay=1e-4
        )
        
        self.output_mgr = OutputManager(save_dir="checkpoints")
        self.trainer = Trainer(
            self.model, self.dataloader, self.optimizer, 
            self.criterion, self.cfg, self.output_mgr
        )

    def execute(self):
        print("\n" + "="*40)
        print(">>> Phase 1: 启动 SWAT-DPD 全局大一统训练管线 <<<")
        print("="*40)
        
        best_ckpt_name = self.trainer.fit()
        self.output_mgr.export_history()
        
        return best_ckpt_name


class EvalPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.dataloader = get_dataloader(cfg)
        self.model = SWAT_DPD(cfg).to(self.device)
        
        self.y_mean = self.dataloader.dataset.y_mean.cpu().numpy()
        self.y_std = self.dataloader.dataset.y_std.cpu().numpy()

    def execute(self, checkpoint_name):
        print("\n" + "="*40)
        print(f">>> Phase 2: 启动自动化评估管线 [{checkpoint_name}] <<<")
        print("="*40)

        checkpoint_path = os.path.join("checkpoints", checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f" 找不到权重文件: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()

       
        all_y_true, all_y_pred, all_params = [], [], []
        
        with torch.no_grad():
            for x_batch, y_batch, params_batch in self.dataloader:
                x_batch = x_batch.to(self.device)
                params_batch = params_batch.to(self.device)
                
                y_pred = self.model(x_batch, params_batch)
                
                all_y_true.append(y_batch.cpu().numpy())
                all_y_pred.append(y_pred.cpu().numpy())
                all_params.append(params_batch.cpu().numpy())

    
        y_true_norm = np.concatenate(all_y_true, axis=0)
        y_pred_norm = np.concatenate(all_y_pred, axis=0)
        p_all = np.concatenate(all_params, axis=0) # Shape: [N_total, 3]

        
        y_true_real_scale = y_true_norm * self.y_std + self.y_mean
        y_pred_real_scale = y_pred_norm * self.y_std + self.y_mean

        
        def calc_subset_nmse(mask, subset_name):
            if not np.any(mask):
                return None
            
            y_t = y_true_real_scale[mask].reshape(-1, 2)
            y_p = y_pred_real_scale[mask].reshape(-1, 2)
            nmse = calculate_nmse_np(y_p, y_t)
            print(f"[*] {subset_name:<8} NMSE: {nmse:.3f} dB")
            return nmse

        
        print("\n" + "-"*45)
        print(" [基线对比成绩单: SWAT-DPD (Param 模式)]")
        print("-" * 45)
        
        
        mask_all = np.ones(len(p_all), dtype=bool)
        calc_subset_nmse(mask_all, "All")
        
        
        mask_w = np.isclose(p_all[:, 0], 1.0)
        calc_subset_nmse(mask_w, "W-Band")
        
        #
        mask_d = np.isclose(p_all[:, 0], 0.0)
        calc_subset_nmse(mask_d, "D-Band")
        
        mask_h = np.isclose(p_all[:, 0], 0.5)
        calc_subset_nmse(mask_h, "H-Band")
        
        print("-" * 45 + "\n")

      
        y_true_flat = y_true_real_scale.reshape(-1, 2)
        y_pred_flat = y_pred_real_scale.reshape(-1, 2)

        os.makedirs("results", exist_ok=True)
       
        save_path = os.path.join("results", f"eval_residual_{checkpoint_name.replace('.pth', '')}_All.png")
        plot_residual_analysis(
            y_true_flat[:, 0], y_true_flat[:, 1], 
            y_pred_flat[:, 0], y_pred_flat[:, 1], 
            save_path
        )
        print(f"[*] 全局残差分析高清图已保存至: {save_path}")