import torch
from utils import calculate_nmse_db

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, cfg, output_mgr):
        self.model = model.to(cfg.device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.output_mgr = output_mgr
        
        # 新增：余弦退火学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.cfg.epochs, 
            eta_min=1e-6
        )

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        epoch_nmse = 0.0

        for x_batch, y_batch in self.dataloader:
            x_batch = x_batch.to(self.cfg.device)
            y_batch = y_batch.to(self.cfg.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_nmse += calculate_nmse_db(y_pred, y_batch)

        steps = len(self.dataloader)
        return epoch_loss / steps, epoch_nmse / steps

    def fit(self):
        for epoch in range(1, self.cfg.epochs + 1):
            loss, nmse = self.train_epoch()
            self.output_mgr.log_metrics(epoch, loss, nmse)
            self.output_mgr.save_checkpoint(self.model, self.optimizer, epoch)
            
            # 新增：每个 epoch 结束后更新学习率
            self.scheduler.step()