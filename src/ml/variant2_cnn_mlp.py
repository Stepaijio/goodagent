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
EPOCHS = 30
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KapitzaDataset(Dataset):
    """Custom Dataset to load sensor data from HDF5 shards."""
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
        
        # Only process active sensors for normalization
        h_active = h_raw[:k]
        h_mean = torch.mean(h_active)
        
        # Relative normalization: (h - h_mean) / (h_mean + eps)
        h_norm = (h_raw - h_mean) / (h_mean + 1e-7)
        
        # Shape: [k_max, 1, 100]
        h_combined = h_norm.unsqueeze(1)
        
        return {
            'rho': self.rho[idx],
            'sigma': self.sigma[idx],
            'k': k,
            'x': self.x_sensors[idx],
            'h': h_combined,
            'y': self.labels[idx]
        }

class Variant2Model(nn.Module):
    """
    Variant 2: 1D-CNN Encoder -> Mean Pooling across sensors -> MLP
    Predicts only log10(mu).
    """
    def __init__(self):
        super(Variant2Model, self).__init__()
        
        # 1D-CNN Encoder to extract local features from a single sensor signal
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # 100 -> 50
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # 50 -> 25
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 25 -> 1
        )
        
        self.feature_dim = 64
        self.mlp_input_dim = self.feature_dim + 2
        
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Predicts log10(mu)
        )

    def forward(self, rho, sigma, k, x, h):
        batch_size = h.shape[0]
        k_max = h.shape[1]
        
        # 1. Process all sensors through the shared encoder
        h_reshaped = h.view(batch_size * k_max, 1, 100)
        encoded_features = self.encoder(h_reshaped) # [batch * k_max, 64, 1]
        encoded_features = encoded_features.squeeze(-1) # [batch * k_max, 64]
        
        # 2. Reshape back to [batch, k_max, 64]
        encoded_features = encoded_features.view(batch_size, k_max, -1)
        
        # 3. Mean Pooling across active sensors only
        mask = torch.arange(k_max).to(DEVICE).unsqueeze(0) < k.unsqueeze(1)
        mask = mask.unsqueeze(-1).float() # [batch, k_max, 1]
        
        sum_features = torch.sum(encoded_features * mask, dim=1) # [batch, 64]
        global_features = sum_features / k.unsqueeze(-1).float() # [batch, 64]
        
        # 4. Add global constants [rho, sigma]
        constants = torch.stack([rho, sigma], dim=-1) # [batch, 2]
        final_input = torch.cat([global_features, constants], dim=-1) # [batch, 66]
        
        return self.mlp(final_input)

def load_data():
    """Loads all examples from HDF5 shards."""
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
        return KapitzaDataset(
            rho_norm[idx], sigma_norm[idx], y_mu[idx], 
            k_vals[idx], x_sensors[idx], readings[idx]
        )

    train_ds = create_ds(train_idx)
    val_ds = create_ds(val_idx)
    test_ds = create_ds(test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = Variant2Model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            rho = batch['rho'].to(DEVICE)
            sigma = batch['sigma'].to(DEVICE)
            k = batch['k'].to(DEVICE)
            x = batch['x'].to(DEVICE)
            h = batch['h'].to(DEVICE)
            y = batch['y'].to(DEVICE).unsqueeze(-1)
            
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
                rho = batch['rho'].to(DEVICE)
                sigma = batch['sigma'].to(DEVICE)
                k = batch['k'].to(DEVICE)
                x = batch['x'].to(DEVICE)
                h = batch['h'].to(DEVICE)
                y = batch['y'].to(DEVICE).unsqueeze(-1)
                
                preds = model(rho, sigma, k, x, h)
                val_loss += criterion(preds, y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
            
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model_v2.pth")
            
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("\nLoss curve saved as loss_curve.png")
            
    model.load_state_dict(torch.load("best_model_v2.pth", weights_only=True))
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            rho = batch['rho'].to(DEVICE)
            sigma = batch['sigma'].to(DEVICE)
            k = batch['k'].to(DEVICE)
            x = batch['x'].to(DEVICE)
            h = batch['h'].to(DEVICE)
            y = batch['y'].to(DEVICE)
            
            preds = model(rho, sigma, k, x, h)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(y.cpu().numpy().flatten())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    print("\n" + "="*40)
    print(f"Variant 2 (CNN-MLP) Evaluation:")
    print(f"  - Target: Log10(Viscosity)")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - R2 Score: {r2:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
