import os
import h5py
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
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
    Consistent feature extraction from Variant 4.
    Includes: Normalization, FFT, Stats, Dynamics, and Spatial Context.
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
    
    # 4. Dynamic Features
    h_diff = np.diff(h_norm)
    mean_abs_vel = np.mean(np.abs(h_diff))
    max_abs_vel = np.max(np.abs(h_diff))
    diff_var = np.var(h_diff)
    
    # 5. Spatial Context
    if prev_h_series is not None:
        p_mean = np.mean(prev_h_series)
        p_norm = (prev_h_series - p_mean) / (p_mean + 1e-7)
        corr = np.correlate(h_norm, p_norm, mode='full')
        norm_factor = np.sqrt(np.sum(h_norm**2) * np.sum(p_norm**2)) + 1e-7
        corr_norm = corr / norm_factor
        spatial_features = [np.max(corr_norm), np.argmax(corr_norm) - (len(h_norm)-1), abs(np.std(h_norm) - np.std(p_norm))]
    else:
        spatial_features = [0, 0, 0]
    
    return np.concatenate([
        [rho, sigma],
        h_norm,
        [dom_freq, spectral_energy, peak_amp, amplitude, variance, std_dev],
        [mean_abs_vel, max_abs_vel, diff_var],
        spatial_features
    ])

def load_and_enrich_data():
    """Loads data and applies Variant 4 enrichment for all samples."""
    shards = sorted([f for f in os.listdir(DATA_DIR) if f.startswith(SHARD_PREFIX) and f.endswith(".h5")])
    
    all_features = []
    all_mu_targets = []
    
    print(f"Loading and enriching data from {len(shards)} shards...")
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
                prev_h = None
                for s in range(k):
                    current_h = readings[i, s, :]
                    feat = extract_enriched_features(current_h, rho[i], sigma[i], prev_h)
                    all_features.append(feat)
                    all_mu_targets.append(np.log10(labels[i, 0]))
                    prev_h = current_h
                    
    return np.array(all_features), np.array(all_mu_targets)

def main():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("Error: No data found.")
        return
    
    # 1. Prepare Data
    X, y = load_and_enrich_data()
    print(f"\nDataset loaded. Features shape: {X.shape}")
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SPLIT/(TRAIN_SPLIT+VAL_SPLIT), random_state=42, shuffle=True
    )
    
    # 2. Initialize Models
    print("\nTraining Ensemble Models...")
    
    # Model A: CatBoost
    m_cat = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=False, random_seed=42)
    m_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    # Model B: XGBoost
    m_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, early_stopping_rounds=50)
    m_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Model C: Random Forest (to reduce variance)
    m_rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    m_rf.fit(X_train, y_train)
    
    # 3. Evaluation on Test Set
    print("\nGenerating Ensemble Predictions...")
    p_cat = m_cat.predict(X_test)
    p_xgb = m_xgb.predict(X_test)
    p_rf = m_rf.predict(X_test)
    
    # Simple Averaging Ensemble
    final_preds = (p_cat + p_xgb + p_rf) / 3.0
    
    # Metrics for individual models and ensemble
    results = {
        "CatBoost": (mean_absolute_error(y_test, p_cat), r2_score(y_test, p_cat)),
        "XGBoost": (mean_absolute_error(y_test, p_xgb), r2_score(y_test, p_xgb)),
        "RandomForest": (mean_absolute_error(y_test, p_rf), r2_score(y_test, p_rf)),
        "Ensemble": (mean_absolute_error(y_test, final_preds), r2_score(y_test, final_preds))
    }
    
    print("\n" + "="*50)
    print(f"{'Model':<20} | {'MAE':<12} | {'R2 Score':<12}")
    print("-" * 50)
    for name, (mae, r2) in results.items():
        print(f"{name:<20} | {mae:<12.6f} | {r2:<12.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
