import sys
import os

# 1. 获取当前 main.py 所在的绝对路径 (也就是 src 文件夹)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 拼接出 model 文件夹的路径
model_dir = os.path.join(current_dir, 'model')

# 3. 将 model 文件夹强行插入到 Python 模块搜索路径的最前面
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)
import argparse
import torch
from config import SWATConfig
from model.swat_dpd import SWAT_DPD
from utils import JointTimeFreqLoss
from dataset import get_dataloader
from trainer import Trainer
from output import OutputManager

def parse_args():
    parser = argparse.ArgumentParser(description="SWAT-DPD Training Pipeline")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Target device (cuda or cpu)')
    return parser.parse_args()

def main():
    # 1. 直接读取你的全局配置，不需要 parse_args() 来捣乱了
    cfg = SWATConfig()

    # 2. 保留原作者一个很好的设计：GPU 可用性检查
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到可用的 GPU，强制降级为 CPU 运行。")
        cfg.device = 'cpu'

    print(f"[*] Starting SWAT-DPD Training on {cfg.device.upper()}...")

    # 构建数据管道与模型组件 (下面完全保持原样)
    dataloader = get_dataloader(cfg)
    model = SWAT_DPD(cfg)
    criterion = JointTimeFreqLoss(lambda_f=cfg.lambda_f)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 实例化输出管理器与训练器
    output_mgr = OutputManager(save_dir="checkpoints")
    trainer = Trainer(model, dataloader, optimizer, criterion, cfg, output_mgr)
    
    # 启动训练闭环
    trainer.fit()
    
    # 导出训练历史记录
    output_mgr.export_history()
    print("[*] Training pipeline completed successfully.")
if __name__ == "__main__":
    main()