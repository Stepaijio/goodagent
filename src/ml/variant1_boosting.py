import os
import h5py
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from src.ml.plotting_utils import plot_scatter, plot_residuals, plot_r2_vs_k

# Configuration
DATA_DIR = "C:/Users/UserSK/Desktop/goodagent/data"
SHARD_PREFIX = "data_shard"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

def load_all_data():
    """
    Loads all examples from HDF5 shards.
    Returns:
    - features: [num_examples, k_max, feature_dim]
    - mu_targets: [num_examples]
    - h_targets: [num_examples]
    - k_vals: [num_examples]
    """
    shards = sorted([f for f in os.listdir(DATA_DIR) if f.startswith(SHARD_PREFIX) and f.endswith(".h5")])
    
    all_features = []
    all_mu_targets = []
    all_h_targets = []
    all_k_vals = []
    
    print(f"Loading data from {len(shards)} shards...")
    for shard in tqdm(shards):
        with h5py.File(os.path.join(DATA_DIR, shard), 'r') as f:
            rho = f['rho'][:]
            sigma = f['sigma'][:]
            labels = f['labels'][:]
            k_vals = f['k'][:]
            readings = f['readings'][:]
            
            num_examples = rho.shape[0]
            for i in range(num_examples):
                k = k_vals[i]
                example_features = []
                for s in range(10): # k_max = 10
                    if s < k:
                        h_series = readings[i, s, :]
                        h_mean = np.mean(h_series)
                        h_norm = (h_series - h_mean) / (h_mean + 1e-7)
                        feat = np.concatenate([[rho[i], sigma[i]], h_norm])
                    else:
                        feat = np.zeros(102)
                    example_features.append(feat)
                
                all_features.append(example_features)
                all_mu_targets.append(np.log10(labels[i, 0]))
                all_h_targets.append(labels[i, 1])
                all_k_vals.append(k)
                
    return np.array(all_features), np.array(all_mu_targets), np.array(all_h_targets), np.array(all_k_vals)

def train_and_evaluate(X_grouped, y, k_vals, target_name):
    """
    X_grouped: [num_examples, k_max, feat_dim]
    y: [num_examples]
    k_vals: [num_examples]
    """
    # Split by example to avoid data leakage
    indices = np.arange(len(X_grouped))
    train_val_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=VAL_SPLIT/(TRAIN_SPLIT+VAL_SPLIT), random_state=42)
    
    # Prepare flat training data (only active sensors)
    def flatten_active(idx_list):
        feat_flat, target_flat = [], []
        for idx in idx_list:
            k = k_vals[idx]
            feat_flat.append(X_grouped[idx, :k])
            target_flat.append([y[idx]] * k)
        return np.vstack([np.vstack(x) for x in feat_flat]), np.concatenate(target_flat)

    X_train_raw, y_train = flatten_active(train_idx)
    X_val_raw, y_val = flatten_active(val_idx)
    X_test_raw, y_test_flat = flatten_active(test_idx)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"\nTraining model for {target_name}...")
    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6, loss_function='RMSE', verbose=False, random_seed=42
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    # Evaluate using the grouping logic for the test set
    y_test_grouped = y[test_idx]
    all_preds_flat = model.predict(X_test)
    
    test_k_vals = k_vals[test_idx]
    example_preds = []
    start_idx = 0
    for k in test_k_vals:
        example_preds.append(all_preds_flat[start_idx : start_idx + k])
        start_idx += k
        
    # Calculate R2 for k=1..10
    k_range = np.arange(1, 11)
    r2_vs_k = []
    
    for k_eval in k_range:
        k_avg_preds = []
        for i in range(len(example_preds)):
            # Use only the first k_eval sensors, or all available if fewer than k_eval
            sensors_to_use = example_preds[i][:k_eval]
            if len(sensors_to_use) > 0:
                k_avg_preds.append(np.mean(sensors_to_use))
            else:
                # Fallback for examples with 0 sensors (though k >= 1 in dataset)
                k_avg_preds.append(np.nan)
        
        # Remove NaNs before calculating R2
        y_true_filtered = np.array([y for y, p in zip(y_test_grouped, k_avg_preds) if not np.isnan(p)])
        p_filtered = np.array([p for p in k_avg_preds if not np.isnan(p)])
        
        if len(p_filtered) > 0:
            r2_vs_k.append(r2_score(y_true_filtered, p_filtered))
        else:
            r2_vs_k.append(0.0)
        
    final_preds = [np.mean(p) for p in example_preds]
    final_r2 = r2_score(y_test_grouped, np.array(final_preds))
    mae = mean_absolute_error(y_test_grouped, np.array(final_preds))
    
    print(f"Results for {target_name}:")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - R2 Score: {final_r2:.4f}")
    
    if "Log-Viscosity" in target_name:
        plot_scatter(y_test_grouped, np.array(final_preds), f"Scatter Plot: {target_name} (V1)", "scatter_v1.png")
        plot_residuals(y_test_grouped, np.array(final_preds), f"Residuals: {target_name} (V1)", "residuals_v1.png")
        plot_r2_vs_k(k_range, np.array(r2_vs_k), f"R2 vs k: {target_name} (V1)", "r2_vs_k_v1.png")

    
    return model

def main():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: No data found in {DATA_DIR}. Please run the generator first.")
        return
    
    # 1. Load and prepare data
    X_grouped, y_mu, y_h, k_vals = load_all_data()
    print(f"\nDataset loaded. Total examples: {X_grouped.shape[0]}")
    print(f"Feature vector size: {X_grouped.shape[-1]}")
    
    # 2. Train Model for Viscosity (log-space)
    model_mu = train_and_evaluate(X_grouped, y_mu, k_vals, "Log-Viscosity (log10(mu))")
    
    # 3. Train Model for Final Thickness
    model_h = train_and_evaluate(X_grouped, y_h, k_vals, "Final Thickness (h_final)")
    
    print("\n" + "="*40)
    print("Variant 1 (Independent Boosting) Complete")
    print("="*40)


if __name__ == "__main__":
    main()
