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
from src.ml.plotting_utils import plot_scatter, plot_residuals, plot_r2_vs_k

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
    """Loads data and applies Variant 4 enrichment for all samples. Returns grouped data."""
    shards = sorted([f for f in os.listdir(DATA_DIR) if f.startswith(SHARD_PREFIX) and f.endswith(".h5")])
    
    all_features = []
    all_mu_targets = []
    all_k_vals = []
    
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
                example_features = []
                prev_h = None
                for s in range(10):
                    if s < k:
                        current_h = readings[i, s, :]
                        feat = extract_enriched_features(current_h, rho[i], sigma[i], prev_h)
                        prev_h = current_h
                    else:
                        feat = np.zeros(114)
                    example_features.append(feat)
                
                all_features.append(example_features)
                all_mu_targets.append(np.log10(labels[i, 0]))
                all_k_vals.append(k)
                
    return np.array(all_features), np.array(all_mu_targets), np.array(all_k_vals)

def main():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("Error: No data found.")
        return
    
    # 1. Prepare Data (Grouped)
    X_grouped, y, k_vals = load_and_enrich_data()
    print(f"\nDataset loaded. Examples: {X_grouped.shape[0]}")
    
    indices = np.arange(len(X_grouped))
    train_val_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=VAL_SPLIT/(TRAIN_SPLIT+VAL_SPLIT), random_state=42)
    
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
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 2. Initialize Models
    print("\nTraining Ensemble Models...")
    m_cat = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=False, random_seed=42)
    m_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    m_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, early_stopping_rounds=50)
    m_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    m_rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    m_rf.fit(X_train, y_train)
    
    # 3. Evaluation and Plots
    y_test_grouped = y[test_idx]
    test_k_vals = k_vals[test_idx]
    
    # Predictions for all active sensors
    p_cat_flat = m_cat.predict(X_test)
    p_xgb_flat = m_xgb.predict(X_test)
    p_rf_flat = m_rf.predict(X_test)
    
    def get_example_preds(flat_preds, k_vals):
        example_preds = []
        start_idx = 0
        for k in k_vals:
            example_preds.append(flat_preds[start_idx : start_idx + k])
            start_idx += k
        return example_preds

    ep_cat = get_example_preds(p_cat_flat, test_k_vals)
    ep_xgb = get_example_preds(p_xgb_flat, test_k_vals)
    ep_rf = get_example_preds(p_rf_flat, test_k_vals)
    
    # Ensemble Prediction: Average across models, then average across sensors
    final_preds = []
    for i in range(len(test_k_vals)):
        combined_sensors = (np.array(ep_cat[i]) + np.array(ep_xgb[i]) + np.array(ep_rf[i])) / 3.0
        final_preds.append(np.mean(combined_sensors))
    
    final_preds = np.array(final_preds)
    
    mae_ens = mean_absolute_error(y_test_grouped, final_preds)
    r2_ens = r2_score(y_test_grouped, final_preds)

    # Calculate metrics for individual models
    def calc_metrics(ep):
        preds = np.array([np.mean(p) for p in ep])
        return mean_absolute_error(y_test_grouped, preds), r2_score(y_test_grouped, preds)

    mae_cat, r2_cat = calc_metrics(ep_cat)
    mae_xgb, r2_xgb = calc_metrics(ep_xgb)
    mae_rf, r2_rf = calc_metrics(ep_rf)

    print("\n" + "="*50)
    print(f"{'Model':<20} | {'MAE':<12} | {'R2 Score':<12}")
    print("-" * 50)
    print(f"{'CatBoost':<20} | {mae_cat:<12.6f} | {r2_cat:<12.4f}")
    print(f"{'XGBoost':<20} | {mae_xgb:<12.6f} | {r2_xgb:<12.4f}")
    print(f"{'RandomForest':<20} | {mae_rf:<12.6f} | {r2_rf:<12.4f}")
    print(f"{'Ensemble':<20} | {mae_ens:<12.6f} | {r2_ens:<12.4f}")
    print("="*50)
    
    # Plotting ONLY for XGBoost as it is the best performing component
    xgb_final_preds = [np.mean(p) for p in ep_xgb]
    xgb_final_preds = np.array(xgb_final_preds)
    xgb_r2 = r2_score(y_test_grouped, xgb_final_preds)
    
    plot_scatter(y_test_grouped, xgb_final_preds, "Scatter Plot: XGBoost (Variant 5)", "scatter_v5.png")
    plot_residuals(y_test_grouped, xgb_final_preds, "Residuals: XGBoost (Variant 5)", "residuals_v5.png")
    
    k_range = np.arange(1, 11)
    xgb_r2_vs_k = []
    for k_eval in k_range:
        k_avg_preds = [np.mean(p[:k_eval]) for p in ep_xgb]
        xgb_r2_vs_k.append(r2_score(y_test_grouped, np.array(k_avg_preds)))
        
    plot_r2_vs_k(k_range, np.array(xgb_r2_vs_k), "R2 vs k: XGBoost (Variant 5)", "r2_vs_k_v5.png")

if __name__ == "__main__":
    main()
