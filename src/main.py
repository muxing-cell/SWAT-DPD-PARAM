import sys
import os
import argparse
import torch

# 确保能找到内部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

from config import SWATConfig
from pipeline import TrainPipeline, EvalPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="SWAT-DPD Golden Entry Point")
    parser.add_argument('--epochs', type=int, help='覆盖 Config 中的轮数')
    parser.add_argument('--batch_size', type=int, help='覆盖 Config 中的 Batch Size')
    parser.add_argument('--lr', type=float, help='覆盖 Config 中的学习率')
   
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'eval', 'all'], 
                        help='运行模式: train(仅训练), eval(仅评估), all(训练并评估)')
    parser.add_argument('--eval_ckpt', type=str, default=None, 
                        help='仅在 eval 模式下有效，指定要评估的权重名称')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = SWATConfig()
    
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr

    if cfg.device == 'cuda' and not torch.cuda.is_available():
        print(" 警告: 未检测到可用 GPU，强制降级为 CPU 运行。")
        cfg.device = 'cpu'

    print(f"[*] SWAT-DPD 系统启动 | 设备: {cfg.device.upper()} | 模式: {args.mode.upper()}")

    try:
        final_ckpt = args.eval_ckpt

        
        if args.mode in ['train', 'all']:
            train_pipe = TrainPipeline(cfg)
          
            final_ckpt = train_pipe.execute()
            
        
        if args.mode in ['eval', 'all']:
            if not final_ckpt:
                final_ckpt = "best_model.pth" # 默认寻找最佳模型
                
            eval_pipe = EvalPipeline(cfg)
            eval_pipe.execute(checkpoint_name=final_ckpt)
            
    except KeyboardInterrupt:
        print("\n[*] 任务被用户手动中断。")
    except Exception as e:
        print(f"\n  运行发生严重错误: {e}")

if __name__ == "__main__":
    main()