import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import random
import pickle
import matplotlib.pyplot as plt
import os

def set_seed(seed):
    """
    设置所有可能的随机种子以确保实验的可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")

# ========================= 1. 流式数据集定义 =========================
class TimeSeriesIterableDataset(IterableDataset):
    def __init__(self, file_path, look_back, mean, std, start_row=0, nrows=None, batch_rows=10000):
        """
        :param file_path: CSV 文件路径
        :param look_back: 滑动窗口长度
        :param mean, std: 标准化参数
        :param start_row: 从第几行开始读（0-based，不含 header）
        :param nrows: 总共读多少行（None 表示到文件末尾）
        :param batch_rows: 每次从 CSV 读多少行
        """
        self.file_path = file_path
        self.look_back = look_back
        self.mean = mean
        self.std = std
        self.start_row = start_row
        self.nrows = nrows
        self.batch_rows = batch_rows

    def __iter__(self):
        leftover = None
        total_read = 0

        skiprows = 1 + self.start_row

        for chunk in pd.read_csv(
                self.file_path,
                header=None,
                skiprows=skiprows,
                chunksize=self.batch_rows,
                nrows=self.nrows
        ):
            chunk = chunk.values.astype(np.float32)

            if leftover is not None:
                chunk = np.vstack([leftover, chunk])

            leftover = chunk[-self.look_back:].copy()

            chunk = (chunk - self.mean) / self.std

            for i in range(len(chunk) - self.look_back):
                X = chunk[i:i+self.look_back]
                y = chunk[i+self.look_back]
                yield torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

            total_read += len(chunk) - (0 if leftover is None else self.look_back)
            if self.nrows is not None and total_read >= self.nrows:
                break

# ========================= 2. 模型定义 (修改部分) =========================
class LstmTransformerModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size=64, lstm_layers=1, dropout=0.1,
                 transformer_nhead=8, transformer_dim_feedforward=256, transformer_layers=1, output_size=None):
        super().__init__()

        # 定义 Transformer 的嵌入维度，确保它能被 nhead 整除
        self.transformer_embed_dim = transformer_dim_feedforward // 2
        # 添加一个线性层来匹配维度
        self.transformer_input_projection = nn.Linear(input_size, self.transformer_embed_dim)

        # LSTM 分支
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        # Transformer 分支
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_embed_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,
                                                         num_layers=transformer_layers)

        # 最终全连接层，输入是 LSTM 和 Transformer 输出的拼接
        combined_output_size = lstm_hidden_size * 2 + self.transformer_embed_dim
        self.fc_out = nn.Linear(combined_output_size, output_size if output_size is not None else input_size)

    def forward(self, x):
        # x shape: (batch_size, look_back, input_size)

        # LSTM 分支
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # Transformer 分支
        transformer_input = self.transformer_input_projection(x) # 投影到新的维度
        transformer_out = self.transformer_encoder(transformer_input)
        transformer_out = transformer_out[:, -1, :] # 取最后一个时间步的输出

        # 拼接两个分支的输出
        combined_out = torch.cat((lstm_out, transformer_out), dim=1)

        # 最终预测
        out = self.fc_out(combined_out)
        return out


# ========================= 3. 评价指标 =========================
def calculate_metrics(y_true, y_pred):
    y_true_np = y_true.flatten()
    y_pred_np = y_pred.flatten()
    mae = mean_absolute_error(y_true_np, y_pred_np)
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_np, y_pred_np)
    r, _ = pearsonr(y_true_np, y_pred_np)
    mean_y_true = np.mean(y_true_np)
    mean_y_pred = np.mean(y_pred_np)
    std_y_true = np.std(y_true_np)
    std_y_pred = np.std(y_pred_np)
    ccc = (2 * r * std_y_true * std_y_pred) / \
          (std_y_true**2 + std_y_pred**2 + (mean_y_true - mean_y_pred)**2)
    return mae, rmse, mse, ccc, r2, r


# ========================= 4. 绘图函数 =========================
def plot_metrics(train_loss, val_loss, r2, r, rmse, mse, mae, ccc, save_path='training_metrics_lstm_transformer.png'):
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 4, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss,label='Validation Loss')
    plt.title("Loss")
    plt.legend()
    metrics = {
        "R2": r2, "Pearson R": r, "RMSE": rmse, "MSE": mse, "MAE": mae, "CCC": ccc
    }
    for i, (name, values) in enumerate(metrics.items(), 2):
        plt.subplot(2, 4, i)
        plt.plot(values)
        plt.title(name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ========================= 5. 主程序 =========================
if __name__ == '__main__':
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = './datasets/complex_interference_dataset.csv'
    model_save_path = './best_model_lstm_transformer.pth'
    norm_params_path = './norm_params_lstm_transformer.pkl'

    max_training_samples = None

    look_back = 10
    batch_size = 64

    # LSTM 模型超参数
    lstm_hidden_size = 128
    lstm_layers = 3
    dropout = 0.3

    # Transformer 模型超参数
    transformer_nhead = 4
    transformer_dim_feedforward = 512
    transformer_layers = 1

    num_epochs = 100
    learning_rate = 0.001
    patience = 100
    epochs_no_improve = 0

    print(f"Loading data to determine total rows...")
    if max_training_samples is None:
        try:
            total_rows = pd.read_csv(data_path, header=None, skiprows=1).shape[0]
            print(f"Total rows found: {total_rows}")
            data_to_read = total_rows
        except FileNotFoundError:
            print(f"Error: The file {data_path} was not found.")
            exit()
    else:
        data_to_read = max_training_samples

    train_size_rows = int(data_to_read * 0.8)
    val_size_rows = data_to_read - train_size_rows

    if os.path.exists(norm_params_path):
        print(f"✅ Normalization parameters file found. Loading from {norm_params_path}...")
        with open(norm_params_path, 'rb') as f:
            norm_params = pickle.load(f)
        mean_per_feature = norm_params['mean']
        std_per_feature = norm_params['std']
        first_row = pd.read_csv(data_path, header=None, skiprows=1, nrows=1).values.astype(np.float32)
        input_size = first_row.shape[1]
    else:
        print(f"🚨 Normalization parameters file not found. Reading training data to compute stats...")
        train_data = pd.read_csv(data_path, header=None, skiprows=1, nrows=train_size_rows).values.astype(np.float32)
        mean_per_feature = np.mean(train_data, axis=0, keepdims=True)
        std_per_feature = np.std(train_data, axis=0, keepdims=True)
        std_per_feature[std_per_feature == 0] = 1e-8
        input_size = train_data.shape[1]
        print(f"Saving normalization parameters to {norm_params_path}...")
        with open(norm_params_path, 'wb') as f:
            pickle.dump({'mean': mean_per_feature, 'std': std_per_feature}, f)
        print("✅ Normalization parameters saved.")

    output_size = input_size

    train_dataset = TimeSeriesIterableDataset(
        file_path=data_path,
        look_back=look_back,
        mean=mean_per_feature,
        std=std_per_feature,
        start_row=0,
        nrows=train_size_rows,
        batch_rows=5000
    )
    val_dataset = TimeSeriesIterableDataset(
        file_path=data_path,
        look_back=look_back,
        mean=mean_per_feature,
        std=std_per_feature,
        start_row=train_size_rows,
        nrows=val_size_rows,
        batch_rows=5000
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LstmTransformerModel(
        input_size=input_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        transformer_nhead=transformer_nhead,
        transformer_dim_feedforward=transformer_dim_feedforward,
        transformer_layers=transformer_layers,
        output_size=output_size
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    best_val_loss = np.inf
    best_epoch_metrics = {}

    train_loss_list, val_loss_list, mae_list, rmse_list, mse_list, ccc_list, r2_list, r_list = [], [], [], [], [], [], [], []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", unit="batch")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': loss.item()})
        avg_train_loss = np.mean(train_losses)

        # 验证阶段
        model.eval()
        val_losses = []
        all_preds, all_labels = [], []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", unit="batch")
        with torch.no_grad():
            for inputs_val, labels_val in val_pbar:
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                val_losses.append(loss_val.item())
                all_preds.append(outputs_val.cpu().numpy())
                all_labels.append(labels_val.cpu().numpy())
                val_pbar.set_postfix({'loss': loss_val.item()})

        avg_val_loss = np.mean(val_losses)
        y_pred_normalized = np.concatenate(all_preds, axis=0)
        y_true_normalized = np.concatenate(all_labels, axis=0)
        y_pred = y_pred_normalized * std_per_feature + mean_per_feature
        y_true = y_true_normalized * std_per_feature + mean_per_feature
        mae, rmse, mse, ccc, r2, r = calculate_metrics(y_true, y_pred)

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mse_list.append(mse)
        ccc_list.append(ccc)
        r2_list.append(r2)
        r_list.append(r)

        # 打印当前 epoch 的指标
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"Current metrics: Val Loss: {avg_val_loss:.6f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, CCC: {ccc:.4f}, R2: {r2:.4f} , R: {r:.4f}")

        # 打印历史最佳指标
        if best_epoch_metrics:
            print(f"🏆 Best metrics so far (Epoch {best_epoch_metrics['epoch']}): Val Loss: {best_epoch_metrics['val_loss']:.6f}, MAE: {best_epoch_metrics['mae']:.4f}, RMSE: {best_epoch_metrics['rmse']:.4f}, CCC: {best_epoch_metrics['ccc']:.4f}, R2: {best_epoch_metrics['r2']:.4f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            best_epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'mae': mae,
                'rmse': rmse,
                'mse': mse,
                'ccc': ccc,
                'r2': r2,
                'r': r
            }
            print(f"🎉 **已保存最佳模型** -> Val Loss = {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("🚨 Early stopping triggered.")
                break

    print("\n" + "="*50)
    print("✨ Training Finished! Final Best Model Metrics:")
    if best_epoch_metrics:
        print(f"Best Epoch: {best_epoch_metrics['epoch']}")
        print(f"Train Loss: {best_epoch_metrics['train_loss']:.6f}, Val Loss: {best_epoch_metrics['val_loss']:.6f}")
        print(f"MAE: {best_epoch_metrics['mae']:.4f}, RMSE: {best_epoch_metrics['rmse']:.4f}, MSE: {best_epoch_metrics['mse']:.4f}")
        print(f"CCC: {best_epoch_metrics['ccc']:.4f}, R2: {best_epoch_metrics['r2']:.4f}, R: {best_epoch_metrics['r']:.4f}")
    else:
        print("No best model was found during training.")
    print("="*50)

    plot_metrics(train_loss_list, val_loss_list, r2_list, r_list, rmse_list, mse_list, mae_list, ccc_list, save_path='training_metrics_lstm_transformer.png')