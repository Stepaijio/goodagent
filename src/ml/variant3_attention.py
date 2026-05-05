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

# Configuration
DATA_DIR = "C:/Users/UserSK/Desktop/goodagent/data"
SHARD_PREFIX = "data_shard"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3 
EPOCHS = 50
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KapitzaDataset(Dataset):
    """Custom Dataset to load sensor data from HDF5 shards with relative normalization."""
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
        h_raw = self.readings[idx]
        k = self.k_vals[idx]
        
        # Relative normalization: (h - h_mean) / (h_mean + eps)
        h_active = h_raw[:k]
        h_mean = torch.mean(h_active)
        h_norm = (h_raw - h_mean) / (h_mean + 1e-7)
        
        # Shape: [k_max, 100]
        return {
            'rho': self.rho[idx],
            'sigma': self.sigma[idx],
            'k': k,
            'x': self.x_sensors[idx],
            'h': h_norm,
            'y': self.labels[idx]
        }

class TemporalEncoder(nn.Module):
    """Processes a single sensor time series using a 1D-CNN for stability."""
    def __init__(self, input_dim=1, d_model=32):
        super(TemporalEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # 100 -> 50
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 50 -> 1
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) # [batch * k, 1, 100]
        x = self.net(x) # [batch * k, 32, 1]
        return x.squeeze(-1) # [batch * k, 32]

class SpatialEncoder(nn.Module):
    """Processes set of sensor features using a lightweight Spatial Attention."""
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=1):
        super(SpatialEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, v, x, k_mask):
        x = x.unsqueeze(-1) # [batch, k_max, 1]
        combined = torch.cat([v, x], dim=-1) # [batch, k_max, d_model_temporal + 1]
        
        z = self.embedding(combined) # [batch, k_max, d_model]
        mask = ~k_mask 
        z = self.transformer(z, src_key_padding_mask=mask) # [batch, k_max, d_model]
        
        z_masked = z * k_mask.unsqueeze(-1)
        global_v = z_masked.sum(dim=1) / (k_mask.sum(dim=1, keepdim=True) + 1e-7)
        return global_v # [batch, d_model]

class Variant3Model(nn.Module):
    """
    Variant 3 (Hybrid): Temporal CNN -> Spatial Attention -> MLP
    Predicts only log10(mu).
    """
    def __init__(self):
        super(Variant3Model, self).__init__()
        self.temp_dim = 32
        self.spat_dim = 32
        
        self.temporal_encoder = TemporalEncoder(d_model=self.temp_dim)
        self.spatial_encoder = SpatialEncoder(input_dim=self.temp_dim + 1, d_model=self.spat_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.spat_dim + 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, rho, sigma, k, x, h):
        batch_size = h.shape[0]
        k_max = h.shape[1]
        
        h_reshaped = h.view(batch_size * k_max, 100)
        v = self.temporal_encoder(h_reshaped) # [batch * k_max, temp_dim]
        v = v.view(batch_size, k_max, -1) # [batch, k_max, temp_dim]
        
        k_mask = torch.arange(k_max).to(DEVICE).unsqueeze(0) < k.unsqueeze(1)
        v_global = self.spatial_encoder(v, x, k_mask) # [batch, spat_dim]
        
        constants = torch.stack([rho, sigma], dim=-1) # [batch, 2]
        final_input = torch.cat([v_global, constants], dim=-1) # [batch, spat_dim + 2]
        
        return self.mlp(final_input)

def load_data():
    shards = sorted([f for f in os.listdir(DATA_DIR) if f.startswith(SHARD_PREFIX) and f.endswith(".h5")])
    rho, sigma, labels, k_vals, x_sensors, readings = [], [], [], [], [], []
    for shard in tqdm(shards, desc="Loading shards"):
        with h5py.File(os.path.join(DATA_DIR, shard), 'r') as f:
            rho.extend(f['rho'][:])
            sigma.extend(f['sigma'][:])
            labels.extend(f['labels'][:])
            k_vals.extend(f['k'][:])
            x_sensors.extend(f['x_sensors'][:])
            readings.extend(f['readings'][:])
    return (np.array(rho), np.array(sigma), np.array(labels), 
            np.array(k_vals), np.array(x_sensors), np.array(readings))

def main():
    print(f"Using device: {DEVICE}")
    rho, sigma, labels, k_vals, x_sensors, readings = load_data()
    y_mu = np.log10(labels[:, 0])
    
    scaler_rho = StandardScaler().fit(rho.reshape(-1, 1))
    scaler_sigma = StandardScaler().fit(sigma.reshape(-1, 1))
    rho_norm = scaler_rho.transform(rho.reshape(-1, 1)).flatten()
    sigma_norm = scaler_sigma.transform(sigma.reshape(-1, 1)).flatten()
    
    indices = np.arange(len(rho))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=VAL_SPLIT/(TRAIN_SPLIT+VAL_SPLIT), random_state=42)
    
    def create_ds(idx):
        return KapitzaDataset(rho_norm[idx], sigma_norm[idx], y_mu[idx], k_vals[idx], x_sensors[idx], readings[idx])

    train_loader = DataLoader(create_ds(train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(create_ds(val_idx), batch_size=BATCH_SIZE)
    test_loader = DataLoader(create_ds(test_idx), batch_size=BATCH_SIZE)
    
    model = Variant3Model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            rho, sigma, k, x, h, y = batch['rho'].to(DEVICE), batch['sigma'].to(DEVICE), batch['k'].to(DEVICE), batch['x'].to(DEVICE), batch['h'].to(DEVICE), batch['y'].to(DEVICE).unsqueeze(-1)
            optimizer.zero_grad()
            preds = model(rho, sigma, k, x, h)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                rho, sigma, k, x, h, y = batch['rho'].to(DEVICE), batch['sigma'].to(DEVICE), batch['k'].to(DEVICE), batch['x'].to(DEVICE), batch['h'].to(DEVICE), batch['y'].to(DEVICE).unsqueeze(-1)
                preds = model(rho, sigma, k, x, h)
                val_loss += criterion(preds, y).item()
        
        avg_train, avg_val = train_loss / len(train_loader), val_loss / len(val_loader)
        train_losses.append(avg_train); val_losses.append(avg_val)
        
        scheduler.step(avg_val)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model_v3.pth")
            
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.title('Variant 3 Loss Curve'); plt.legend(); plt.grid(True)
    plt.savefig('loss_curve_v3.png')
    
    model.load_state_dict(torch.load("best_model_v3.pth", weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            rho, sigma, k, x, h, y = batch['rho'].to(DEVICE), batch['sigma'].to(DEVICE), batch['k'].to(DEVICE), batch['x'].to(DEVICE), batch['h'].to(DEVICE), batch['y'].to(DEVICE)
            preds = model(rho, sigma, k, x, h)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(y.cpu().numpy().flatten())
            
    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)
    print("\n" + "="*40)
    print(f"Variant 3 (Attention) Evaluation:")
    print(f"  - Target: Log10(Viscosity)")
    print(f"  - MAE: {mean_absolute_error(all_targets, all_preds):.6f}")
    print(f"  - R2 Score: {r2_score(all_targets, all_preds):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
