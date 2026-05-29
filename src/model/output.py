import os
import torch
import json

class OutputManager:
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = {'loss': [], 'nmse': []}

    def log_metrics(self, epoch, loss, nmse):
        self.history['loss'].append(loss)
        self.history['nmse'].append(nmse)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | NMSE: {nmse:.2f} dB")

    def save_checkpoint(self, model, optimizer, epoch, filename="latest.pth"):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def export_history(self):
        with open(os.path.join(self.save_dir, "history.json"), "w") as f:
            json.dump(self.history, f)