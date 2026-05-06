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

def extract_enriched_features(h_series, rho, sigma, prev_h_series=None):
    """
    Extracts enriched features from a single sensor time series.
    Phase 1: FFT + Basic Statistics.
    Phase 2: Dynamic Features (Derivatives).
    Phase 3: Spatial Context (Correlation with previous sensor).
    """
    # 1. Relative Normalization
    h_mean = np.mean(h_series)
    h_norm = (h_series - h_mean) / (h_mean + 1e-7)
    
    # 2. Frequency Domain Features (FFT)
    fft_vals = np.abs(np.fft.rfft(h_norm))
    if len(fft_vals) > 1:
        dom_freq_idx = np.argmax(fft_vals[1:]) + 1
        dom_freq = dom_freq_idx / len(h_series)
        spectral_energy = np.sum(fft_vals**2)
        peak_amp = np.max(fft_vals)
    else:
        dom_freq, spectral_energy, peak_amp = 0, 0, 0

    # 3. Time Domain Statistics
    amplitude = np.max(h_norm) - np.min(h_norm)
    variance = np.var(h_norm)
    std_dev = np.std(h_norm)
    
    # 4. Dynamic Features (Phase 2)
    h_diff = np.diff(h_norm)
    mean_abs_vel = np.mean(np.abs(h_diff))
    max_abs_vel = np.max(np.abs(h_diff))
    diff_var = np.var(h_diff)
    
    # 5. Spatial Context (Phase 3)
    if prev_h_series is not None:
        # Normalize previous series too
        prev_mean = np.mean(prev_h_series)
        prev_norm = (prev_h_series - prev_mean) / (prev_mean + 1e-7)
        
        # Cross-correlation to find lag and similarity
        # We use a simple normalized cross-correlation
        corr = np.correlate(h_norm, prev_norm, mode='full')
        # Normalize correlation by magnitude to get coefficient-like value
        norm_factor = np.sqrt(np.sum(h_norm**2) * np.sum(prev_norm**2)) + 1e-7
        corr_norm = corr / norm_factor
        
        max_corr = np.max(corr_norm)
        # Lag at which max correlation occurs (relative to center)
        # Center of 'full' correlate is at index len(h_norm) - 1
        lag = np.argmax(corr_norm) - (len(h_norm) - 1)
        std_diff = abs(np.std(h_norm) - np.std(prev_norm))
        
        spatial_features = [max_corr, lag, std_diff]
    else:
        # No previous sensor for the first one in the sequence
        spatial_features = [0, 0, 0]
    
    # 6. Combine all features
    # [rho, sigma] + [raw_norm_series (100)] + [FFT (3)] + [Stats (3)] + [Dynamics (3)] + [Spatial (3)]
    features = np.concatenate([
        [rho, sigma],
        h_norm,
        [dom_freq, spectral_energy, peak_amp, amplitude, variance, std_dev],
        [mean_abs_vel, max_abs_vel, diff_var],
        spatial_features
    ])
    
    return features

def load_all_data():
    """
    Loads all examples from HDF5 shards and transforms them into 
    an enriched feature set for Independent Boosting.
    """
    shards = sorted([f for f in os.listdir(DATA_DIR) if f.startswith(SHARD_PREFIX) and f.endswith(".h5")])
    
    all_features = []
    all_mu_targets = []
    
    print(f"Loading and enriching data from {len(shards)} shards...")
    for shard in tqdm(shards):
        with h5py.File(os.path.join(DATA_DIR, shard), 'r') as f:
            rho = f['rho'][:]
            sigma = f['sigma'][:]
            labels = f['labels'][:] # [mu, h_final]
            k_vals = f['k'][:]
            readings = f['readings'][:]
            
            num_examples = rho.shape[0]
            for i in range(num_examples):
                k = k_vals[i]
                # Each sensor is an independent training example, 
                # but we now consider the sequence to extract spatial features.
                prev_h = None
                for s in range(k):
                    current_h = readings[i, s, :]
                    feat = extract_enriched_features(current_h, rho[i], sigma[i], prev_h_series=prev_h)
                    all_features.append(feat)
                    
                    # Target mu in log-space
                    all_mu_targets.append(np.log10(labels[i, 0]))
                    
                    # Store current for next sensor's spatial analysis
                    prev_h = current_h
                    
    return np.array(all_features), np.array(all_mu_targets)

def train_and_evaluate(X, y, target_name):
    """Helper to train CatBoost and print metrics."""
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SPLIT/(TRAIN_SPLIT+VAL_SPLIT), random_state=42, shuffle=True
    )
    
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
    
    # 1. Load and Enrich Data
    X, y_mu = load_all_data()
    print(f"\nDataset loaded. Total sensor-samples: {X.shape[0]}")
    print(f"Enriched feature vector size: {X.shape[1]}")
    
    # 2. Train Model for Viscosity (log-space)
    model_mu = train_and_evaluate(X, y_mu, "Log-Viscosity (log10(mu))")
    
    # Save the best model for ensemble use
    model_mu.save_model("best_model_v4.pth")
    print("Model saved to best_model_v4.pth")
    
    print("\n" + "="*40)
    print("Variant 4 (Enriched Boosting - Phase 1) Complete")
    print("="*40)

if __name__ == "__main__":
    main()
