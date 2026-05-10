import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from src.ml.plotting_utils import plot_scatter, plot_residuals, plot_r2_vs_k

# Configuration
DATA_DIR = "C:/Users/UserSK/Desktop/goodagent/data"
SHARD_PREFIX = "data_shard"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KapitzaDataset(Dataset):
    def __init__(self, rho, sigma, labels, k_vals, x_sensors, readings):
        self.rho = torch.FloatTensor(rho)
        self.sigma = torch.FloatTensor(sigma)
        self.labels = torch.FloatTensor(labels)
        self.k_vals = torch.LongTensor(k_vals)
        self.x_sensors = torch.FloatTensor(x_sensors)
        self.readings = torch.FloatTensor(readings)

    def __len__(self):
        return len(self.rho)

    def __getitem__(self, idx):
        # Relative normalization: (h - h_mean) / (h_mean + eps)
        h_raw = self.readings[idx]
        k = self.k_vals[idx]
        h_active = h_raw[:k]
        h_mean = torch.mean(h_active)
        h_norm = (h_raw - h_mean) / (h_mean + 1e-7)
        
        return {
            'rho': self.rho[idx],
            'sigma': self.sigma[idx],
            'k': k,
            'x': self.x_sensors[idx],
            'h': h_norm,
            'y': self.labels[idx]
        }

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=1, d_model=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)

class SpatialEncoder(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, v, x, k_mask):
        combined = torch.cat([v, x.unsqueeze(-1)], dim=-1)
        z = self.transformer(self.embedding(combined), src_key_padding_mask=~k_mask)
        return (z * k_mask.unsqueeze(-1)).sum(dim=1) / k_mask.sum(dim=1, keepdim=True).clamp(min=1)

class Variant3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp_dim = 32
        self.spat_dim = 32
        self.temporal_encoder = TemporalEncoder(d_model=self.temp_dim)
        self.spatial_encoder = SpatialEncoder(input_dim=self.temp_dim + 1, d_model=self.spat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.spat_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, rho, sigma, k, x, h, k_eval=None):
        batch_size, k_max = h.shape[0], h.shape[1]
        v = self.temporal_encoder(h.view(batch_size * k_max, 100)).view(batch_size, k_max, -1)
        limit = torch.min(k, torch.tensor(k_eval if k_eval else 10).to(DEVICE))
        k_mask = torch.arange(k_max).to(DEVICE).unsqueeze(0) < limit.unsqueeze(1)
        v_global = self.spatial_encoder(v, x, k_mask)
        return self.mlp(torch.cat([v_global, rho.unsqueeze(1), sigma.unsqueeze(1)], dim=-1))

def load_data():
    shards = sorted([f for f in os.listdir(DATA_DIR) if f.startswith(SHARD_PREFIX) and f.endswith(".h5")])
    rho, sigma, labels, k_vals, x_sensors, readings = [], [], [], [], [], []
    for shard in tqdm(shards, desc="Loading"):
        with h5py.File(os.path.join(DATA_DIR, shard), 'r') as f:
            rho.extend(f['rho'][:])
            sigma.extend(f['sigma'][:])
            labels.extend(f['labels'][:])
            k_vals.extend(f['k'][:])
            x_sensors.extend(f['x_sensors'][:])
            readings.extend(f['readings'][:])
    return map(np.array, (rho, sigma, labels, k_vals, x_sensors, readings))

def main():
    rho, sigma, labels, k_vals, x_sensors, readings = load_data()
    y = np.log10(labels[:, 0])
    
    indices = np.arange(len(rho))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=VAL_SPLIT/(TRAIN_SPLIT+VAL_SPLIT), random_state=42)
    
    scaler_rho = StandardScaler().fit(rho[train_idx].reshape(-1, 1))
    scaler_sigma = StandardScaler().fit(sigma[train_idx].reshape(-1, 1))
    rho_norm = scaler_rho.transform(rho.reshape(-1, 1)).flatten()
    sigma_norm = scaler_sigma.transform(sigma.reshape(-1, 1)).flatten()
    
    def create_ds(idx):
        return KapitzaDataset(rho_norm[idx], sigma_norm[idx], y[idx], k_vals[idx], x_sensors[idx], readings[idx])
    
    train_loader = DataLoader(create_ds(train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(create_ds(val_idx), batch_size=BATCH_SIZE)
    test_loader = DataLoader(create_ds(test_idx), batch_size=BATCH_SIZE)
    
    model = Variant3Model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for b in train_loader:
            optimizer.zero_grad()
            preds = model(b['rho'].to(DEVICE), b['sigma'].to(DEVICE), b['k'].to(DEVICE), b['x'].to(DEVICE), b['h'].to(DEVICE))
            loss = criterion(preds.squeeze(), b['y'].to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b in val_loader:
                preds = model(b['rho'].to(DEVICE), b['sigma'].to(DEVICE), b['k'].to(DEVICE), b['x'].to(DEVICE), b['h'].to(DEVICE))
                val_loss += criterion(preds.squeeze(), b['y'].to(DEVICE)).item()
        
        avg_train, avg_val = train_loss / len(train_loader), val_loss / len(val_loader)
        train_losses.append(avg_train); val_losses.append(avg_val)
        print(f"Epoch {epoch+1} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")
            
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Потери при тренировке'); plt.plot(val_losses, label='Потери при валидации')
    plt.xlabel('Эпоха'); plt.ylabel('Потери (MSE)'); plt.legend(); plt.grid(True)
    plt.savefig('loss_curve_final.png')

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for b in test_loader:
            preds = model(b['rho'].to(DEVICE), b['sigma'].to(DEVICE), b['k'].to(DEVICE), b['x'].to(DEVICE), b['h'].to(DEVICE))
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(b['y'].cpu().numpy().flatten())
            
    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)
    
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    print("\n" + "="*40)
    print(f"Variant 3 Final Evaluation:")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - R2 Score: {r2:.4f}")
    print("="*40)
    
    plot_scatter(all_targets, all_preds, "Scatter Plot: Variant 3", "scatter_v3.png")
    plot_residuals(all_targets, all_preds, "Residuals: Variant 3", "residuals_v3.png")
    
    # R2 vs k calculation
    k_range = np.arange(1, 11)
    r2_vs_k = []
    with torch.no_grad():
        for k_eval in k_range:
            temp_preds, temp_targets = [], []
            for b in test_loader:
                preds = model(b['rho'].to(DEVICE), b['sigma'].to(DEVICE), b['k'].to(DEVICE), b['x'].to(DEVICE), b['h'].to(DEVICE), k_eval=k_eval)
                temp_preds.append(preds.cpu().numpy().flatten())
                temp_targets.append(b['y'].cpu().numpy().flatten())
            r2_vs_k.append(r2_score(np.concatenate(temp_targets), np.concatenate(temp_preds)))
            
    plot_r2_vs_k(k_range, np.array(r2_vs_k), "R2 vs k: Variant 3", "r2_vs_k_v3.png")

if __name__ == "__main__":
    main()
