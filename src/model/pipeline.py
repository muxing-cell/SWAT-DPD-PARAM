import torch
from config import SWATConfig
from model.swat_dpd import SWAT_DPD
from utils import JointTimeFreqLoss
from dataset import get_dataloader
from trainer import Trainer
from output import OutputManager

def main():
    cfg = SWATConfig()
    
    dataloader = get_dataloader(cfg)
    model = SWAT_DPD(cfg)
    criterion = JointTimeFreqLoss(lambda_f=cfg.lambda_f)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    output_mgr = OutputManager()
    
    trainer = Trainer(model, dataloader, optimizer, criterion, cfg, output_mgr)
    trainer.fit()
    
    output_mgr.export_history()

if __name__ == "__main__":
    main()