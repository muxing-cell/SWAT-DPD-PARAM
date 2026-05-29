import torch
import torch.nn as nn

class JointTimeFreqLoss(nn.Module):
    def __init__(self, lambda_f=0.5):
        super().__init__()
        self.lambda_f = lambda_f
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        l_time = self.mse(y_pred, y_true)
        
        c_pred = torch.complex(y_pred[..., 0], y_pred[..., 1])
        c_true = torch.complex(y_true[..., 0], y_true[..., 1])
        
        f_pred = torch.fft.fft(c_pred, dim=-1)
        f_true = torch.fft.fft(c_true, dim=-1)
        
        l_freq = torch.mean(torch.abs(f_pred - f_true) ** 2)
        return l_time + self.lambda_f * l_freq

def calculate_nmse_db(y_pred, y_true):
    with torch.no_grad():
        error_sq = (y_pred[..., 0] - y_true[..., 0])**2 + (y_pred[..., 1] - y_true[..., 1])**2
        true_power = y_true[..., 0]**2 + y_true[..., 1]**2
        nmse_linear = torch.sum(error_sq) / (torch.sum(true_power) + 1e-12)
        nmse_db = 10 * torch.log10(nmse_linear)
    return nmse_db.item()