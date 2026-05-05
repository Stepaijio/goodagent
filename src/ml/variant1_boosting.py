import os
import h5py
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Configuration
DATA_DIR = "C:/Users/UserSK/Desktop/goodagent/data"
SHARD_PREFIX = "data_shard"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

def load_all_data():
    """
    Loads all examples from HDF5 shards and transforms them into a format
    suitable for 'Independent Boosting'.
    Each sensor in each example becomes a separate training row.
    """
    shards = sorted([f for f in os.listdir(DATA_DIR) if f.startswith(SHARD_PREFIX) and f.endswith(".h5")])
    
    all_features = []
    all_mu_targets = []
    all_h_targets = []
    
    print(f"Loading data from {len(shards)} shards...")
    for shard in tqdm(shards):
        with h5py.File(os.path.join(DATA_DIR, shard), 'r') as f:
            rho = f['rho'][:]
            sigma = f['sigma'][:]
            labels = f['labels'][:] # [mu, h_final]
            k_vals = f['k'][:]
            x_sensors = f['x_sensors'][:]
            readings = f['readings'][:]
            
            num_examples = rho.shape[0]
            for i in range(num_examples):
                k = k_vals[i]
                # Each sensor is an independent training example
                for s in range(k):
                    # 1. Extract raw readings
                    h_series = readings[i, s, :]
                    h_mean = np.mean(h_series)
                    
                    # 2. Relative normalization: (h - h_mean) / (h_mean + eps)
                    # This forces the model to focus on wave dynamics rather than absolute thickness
                    h_norm = (h_series - h_mean) / (h_mean + 1e-7)
                    
                    # Feature vector: [rho, sigma, h_norm_1, ..., h_norm_100]
                    feature_vector = np.concatenate([
                        [rho[i], sigma[i]], 
                        h_norm
                    ])
                    all_features.append(feature_vector)
                    
                    # Target mu is handled in log-space for better distribution
                    all_mu_targets.append(np.log10(labels[i, 0]))
                    all_h_targets.append(labels[i, 1])
                    
    return np.array(all_features), np.array(all_mu_targets), np.array(all_h_targets)

def train_and_evaluate(X, y, target_name):
    """Helper to train CatBoost and print metrics."""
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SPLIT/(TRAIN_SPLIT+VAL_SPLIT), random_state=42, shuffle=True
    )

    # Normalize features (especially rho and sigma)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"\nTraining model for {target_name}...")
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        verbose=False,
        random_seed=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    # Predict and evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Results for {target_name}:")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - R2 Score: {r2:.4f}")
    
    return model

def main():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: No data found in {DATA_DIR}. Please run the generator first.")
        return

    # 1. Load and prepare data
    X, y_mu, y_h = load_all_data()
    print(f"\nDataset loaded. Total sensor-samples: {X.shape[0]}")
    print(f"Feature vector size: {X.shape[1]}")
    
    # 2. Train Model for Viscosity (log-space)
    model_mu = train_and_evaluate(X, y_mu, "Log-Viscosity (log10(mu))")
    
    # 3. Train Model for Final Thickness
    model_h = train_and_evaluate(X, y_h, "Final Thickness (h_final)")
    
    print("\n" + "="*40)
    print("Variant 1 (Independent Boosting) Complete")
    print("="*40)

if __name__ == "__main__":
    main()
