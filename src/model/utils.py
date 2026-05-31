import torch
import torch.nn as nn
import numpy as np

# 强制唤醒独立图形窗口
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

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

def calculate_nmse_np(pred, true):
    mse = np.mean((pred - true) ** 2)
    power = np.mean(true ** 2)
    nmse = mse / (power + 1e-10)
    return 10 * np.log10(nmse + 1e-10)

def plot_residual_analysis(y_true_real, y_true_imag, y_pred_real, y_pred_imag, save_path, display_len=500):
    plt.figure(figsize=(15, 8))
    t = np.arange(display_len)
    
    plt.subplot(2, 2, 1)
    plt.plot(t, y_true_real[:display_len], label='True Real', color='black', linestyle='-', linewidth=1.5)
    plt.plot(t, y_pred_real[:display_len], label='Pred Real', color='red', linestyle='--', linewidth=1.5)
    plt.title("Real Part Prediction")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(2, 2, 2)
    plt.plot(t, y_true_imag[:display_len], label='True Imag', color='black', linestyle='-', linewidth=1.5)
    plt.plot(t, y_pred_imag[:display_len], label='Pred Imag', color='blue', linestyle='--', linewidth=1.5)
    plt.title("Imaginary Part Prediction")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    error_real = y_true_real[:display_len] - y_pred_real[:display_len]
    plt.subplot(2, 2, 3)
    plt.plot(t, error_real, color='gray', linewidth=1.0)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals (True - Pred) - Real Part")
    plt.xlabel("Sample Index")
    plt.ylabel("Error Amplitude")
    plt.grid(True, alpha=0.3)

    error_imag = y_true_imag[:display_len] - y_pred_imag[:display_len]
    plt.subplot(2, 2, 4)
    plt.plot(t, error_imag, color='gray', linewidth=1.0)
    plt.axhline(0, color='blue', linestyle='--')
    plt.title("Residuals (True - Pred) - Imag Part")
    plt.xlabel("Sample Index")
    plt.ylabel("Error Amplitude")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    # 强制弹出图形交互窗口
    plt.show() 
    plt.close()