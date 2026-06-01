import torch
import torch.nn as nn
import os 
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from utils import calculate_nmse_db  

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, cfg, output_mgr):
        self.model = model.to(cfg.device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.cfg = cfg
        self.output_mgr = output_mgr
        
        # 1. 恢复纯时域 MSE 损失
        self.criterion = nn.MSELoss()
        
        # 2. 引入带热重启的余弦退火
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=getattr(self.cfg, 'T_0', 200), 
            T_mult=2,         
            eta_min=1e-6
        )

        # 3. 初始化 EMA 影子模型
        self.ema_model = AveragedModel(
            self.model, 
            multi_avg_fn=get_ema_multi_avg_fn(0.999)
        ).to(cfg.device)
        
        self.y_mean = self.dataloader.dataset.y_mean.to(self.cfg.device)
        self.y_std = self.dataloader.dataset.y_std.to(self.cfg.device)

        # 4. 梯度累加步数
        self.accumulation_steps = 4  

    def train_epoch(self):
        
        self.model.train()
        epoch_loss = 0.0
        epoch_nmse = 0.0

        for i, (x_batch, y_batch, params_batch) in enumerate(self.dataloader):
            x_batch = x_batch.to(self.cfg.device)
            y_batch = y_batch.to(self.cfg.device)
            params_batch = params_batch.to(self.cfg.device)

            y_pred = self.model(x_batch, params_batch)
            
            # --- 恢复干净利落的 MSE 计算 ---
            loss = self.criterion(y_pred, y_batch)
            
            # 损失除以累加步数，保证物理步长安全
            loss = loss / self.accumulation_steps
            loss.backward()
            
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.dataloader):
                
                # 梯度防爆
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                self.optimizer.zero_grad() 
                
                # EMA 吸收
                self.ema_model.update_parameters(self.model)

            epoch_loss += loss.item() * self.accumulation_steps
            
            with torch.no_grad():
                y_pred_smooth = self.ema_model(x_batch, params_batch)
                
                y_pred_real = y_pred_smooth * self.y_std + self.y_mean
                y_batch_real = y_batch * self.y_std + self.y_mean
                
                epoch_nmse += calculate_nmse_db(y_pred_real, y_batch_real)

        steps = len(self.dataloader)
        return epoch_loss / steps, epoch_nmse / steps

    def fit(self):
        best_nmse = float('inf')
        best_epoch = 0
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            loss, nmse = self.train_epoch()
            
            self.output_mgr.log_metrics(epoch, loss, nmse)
            self.output_mgr.save_checkpoint(self.ema_model.module, self.optimizer, epoch)
            self.scheduler.step()
            
            # --- 记分牌逻辑 ---
            if nmse < best_nmse:
                best_nmse = nmse
                best_epoch = epoch
                # 注意：下面这两行必须和上面的 best_epoch 严格垂直对齐
                best_path = os.path.join(save_dir, "best_model_All.pth")
                torch.save(self.ema_model.module.state_dict(), best_path)
                
        print("\n" + "*"*50)
        print(f"[*] 训练完结！")
        print(f"[*] 历史最好参数出现于 Epoch {best_epoch}，NMSE: {best_nmse:.3f} dB")
        print(f"[*] 最好权重已独立封存至: checkpoints/best_model_All.pth")
        print("*"*50)
        
        return "best_model_All.pth"