import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def dataloader(
    dataset_input_list: list,
    dataset_output_list_list: list,
    param_list: list,
    window_size: int = 11,
    batch_size: int = 32,
    device: str = "cuda",
):

    X_train, y_train, X_test, y_test = [], [], [], []
    param_train, param_test = [], []

    # ===================== Data Loading =====================
    for idx, input_path in enumerate(dataset_input_list):
        output_paths = dataset_output_list_list[idx]
        param = param_list[idx]

        # --- 修改 1: 绝对保留复数形态，拒绝拆分解耦 ---
        # 展平为一维复数数组，假设原始形状为 [N, 1]
        input_data = scipy.io.loadmat(input_path)["x"].flatten()

        # --- Load multiple output signals ---
        outputs = []
        for output_path in output_paths:
            data = scipy.io.loadmat(output_path)["PA_baseband"].flatten()
            outputs.append(data)

        # Split first 4 outputs for training, last one for testing
        train_outputs = outputs[:4]
        test_output = outputs[-1]

        # --- Construct sliding windows ---
        num_samples = len(input_data) - window_size
        for output in train_outputs:
            for i in range(num_samples):
                X_train.append(input_data[i:i + window_size])
                y_train.append(output[i + window_size - 1])

        for i in range(num_samples):
            X_test.append(input_data[i:i + window_size])
            y_test.append(test_output[i + window_size - 1])

        # --- Expand parameters ---
        param_train.extend(np.tile(np.array(param), (num_samples * 4, 1)))
        param_test.extend(np.tile(np.array(param), (num_samples, 1)))

    # ===================== Convert to NumPy Arrays =====================
    # 此时生成的 array 类型将自动识别为 np.complex128 或 np.complex64
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    param_train, param_test = np.array(param_train), np.array(param_test)

    # ===================== Data Augmentation =====================
    # --- 修改 2: 暂时注释掉实值域的特征增强 ---
    # X_train = augmented_features(X_train)
    # X_test = augmented_features(X_test)

    # ===================== Normalization =====================
    # --- 修改 3: 实现严格的复数域中心化与缩放 ---
    # 计算复数均值和基于模长的标准差
    x_mean = np.mean(X_train)
    x_std = np.std(X_train)
    X_train_norm = (X_train - x_mean) / x_std
    X_test_norm = (X_test - x_mean) / x_std

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    # ===================== Convert to PyTorch Tensors =====================
    # --- 修改 4: 数据类型必须声明为复数张量 (torch.cfloat) ---
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.cfloat, device=device)
    y_train_tensor = torch.tensor(y_train_norm, dtype=torch.cfloat, device=device)
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.cfloat, device=device)
    y_test_tensor = torch.tensor(y_test_norm, dtype=torch.cfloat, device=device)
    
    # 参数/标签网络依然保持实数
    param_train_tensor = torch.tensor(param_train, dtype=torch.float32, device=device)
    param_test_tensor = torch.tensor(param_test, dtype=torch.float32, device=device)

    # ===================== Create Datasets and Loaders =====================
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, param_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, param_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 返回 y_mean 和 y_std 构成的元组，用于后续反归一化计算真实 NMSE
    return train_loader, test_loader, (y_mean, y_std)