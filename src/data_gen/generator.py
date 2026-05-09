import os
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.physics.model import KapitzaModel
from src.utils.constants import G

# Configuration
TOTAL_EXAMPLES = 100_000
EXAMPLES_PER_SHARD = 10_000
BATCH_SIZE = 10  # Number of examples a worker generates per task
DATA_DIR = "data"
SHARD_PREFIX = "data_shard"

# Physical Parameter Ranges
T_RANGE = (20, 80)
C_RANGE = (0.3, 0.8)
H0_RANGE = (0.001, 0.01)
EPSILON_RANGE = (0.001, 0.05)
L = 0.5
SENSOR_X_RANGE = (0.05, 0.45)
SENSOR_COUNT_RANGE = (1, 10)
T_SIM_RANGE = (0.2, 0.3)
T_WINDOW = 0.1
SAMPLES_PER_WINDOW = 100
DT_SENS = 0.001

def sample_log_uniform(low, high):
    """Sample from a log-uniform distribution."""
    return 10**np.random.uniform(np.log10(low), np.log10(high))

def get_physical_properties(T, C):
    """Calculate rho, mu, sigma based on Temperature and Concentration of Glycerin."""
    # Density: linear interpolation between water (1000) and glycerin (1260)
    rho = 1000 + 260 * C
    
    # Surface tension: linear interpolation between water (0.072) and glycerin (0.063)
    sigma = 0.072 - 0.009 * C
    
    # Viscosity: empirical approximation
    # mu_20 is viscosity at 20C: log10(mu) = -3 + 3*C
    mu_20 = 10**(-3 + 3 * C)
    # Temperature correction: mu(T) = mu_20 * exp(-0.02 * (T - 20))
    mu = mu_20 * np.exp(-0.02 * (T - 20))
    
    return rho, mu, sigma

def generate_batch(batch_size):
    """Generates a batch of valid examples."""
    batch_features = []
    batch_labels = []
    
    count = 0
    while count < batch_size:
        # --- Sampling ---
        T = np.random.uniform(*T_RANGE)
        C = np.random.uniform(*C_RANGE)
        rho, mu, sigma = get_physical_properties(T, C)
        
        h0 = sample_log_uniform(*H0_RANGE)
        epsilon = np.random.uniform(*EPSILON_RANGE)
        delta = epsilon * h0
        t_sim = np.random.uniform(*T_SIM_RANGE)
        
        # Re check: Re = (rho^2 * g * h0^3) / (3 * mu^2)
        # G is imported from src.utils.constants
        re = (rho**2 * G * h0**3) / (3 * mu**2)
        if re > 50:
            continue
            
        k = np.random.randint(SENSOR_COUNT_RANGE[0], SENSOR_COUNT_RANGE[1] + 1)
        x_sensors = np.sort(np.random.uniform(SENSOR_X_RANGE[0], SENSOR_X_RANGE[1], k))
        
        # --- Simulation ---
        nx = 100
        dt = 1e-6
        model = KapitzaModel(rho, mu, sigma, h0, L, nx, delta)
        
        # We need to record data during the last T_WINDOW.
        start_recording_step = int((t_sim - T_WINDOW) / dt)
        total_steps = int(t_sim / dt)
        
        readings = np.zeros((k, SAMPLES_PER_WINDOW))
        
        is_stable = True
        for step in range(total_steps):
            model.step(dt)
            h = model.get_h()
            
            if np.any(np.isnan(h)) or np.any(np.isinf(h)) or np.max(h) > 10 * h0:
                is_stable = False
                break
            
            if step >= start_recording_step:
                relative_step = step - start_recording_step
                sample_idx = relative_step // int(DT_SENS / dt)
                if sample_idx < SAMPLES_PER_WINDOW:
                    readings[:, sample_idx] = np.interp(x_sensors, model.x, h)
        
        if is_stable:
            features = {
                'rho': rho,
                'sigma': sigma,
                'x_sensors': x_sensors,
                'readings': readings
            }
            labels = np.array([mu, np.mean(model.get_h())])
            
            batch_features.append(features)
            batch_labels.append(labels)
            count += 1
            
    return batch_features, batch_labels


def generate_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Resume logic: check how many shards already exist
    existing_shards = [f for f in os.listdir(DATA_DIR) if f.startswith(SHARD_PREFIX) and f.endswith(".h5")]
    num_existing_shards = len(existing_shards)
    start_shard = num_existing_shards
    
    examples_generated = num_existing_shards * EXAMPLES_PER_SHARD
    print(f"Found {num_existing_shards} shards. Resuming from example {examples_generated}...")

    with ProcessPoolExecutor() as executor:
        # Calculate how many shards are left
        remaining_examples = TOTAL_EXAMPLES - examples_generated
        if remaining_examples <= 0:
            print("Dataset already complete!")
            return

        num_batches = (remaining_examples + BATCH_SIZE - 1) // BATCH_SIZE
        futures = [executor.submit(generate_batch, BATCH_SIZE) for _ in range(num_batches)]
        
        pbar = tqdm(total=remaining_examples, initial=examples_generated, desc="Generating Dataset")
        
        current_shard_idx = start_shard
        current_shard_count = 0
        
        # Current shard file
        shard_path = os.path.join(DATA_DIR, f"{SHARD_PREFIX}_{current_shard_idx}.h5")
        # We use a dictionary-like structure for HDF5. 
        # Because k varies, we'll store:
        # - 'rho': (N,)
        # - 'sigma': (N,)
        # - 'labels': (N, 2)
        # - 'sensor_data': Group where each dataset is a sensor record? No, that's messy.
        # Better: Store sensor_data as a ragged array or padded array.
        # Let's use a padded array [N, 10, 100] and a 'k' array [N].
        
        # Pre-open HDF5 for the current shard
        f = h5py.File(shard_path, 'w')
        dset_rho = f.create_dataset("rho", (EXAMPLES_PER_SHARD,), dtype='f4')
        dset_sigma = f.create_dataset("sigma", (EXAMPLES_PER_SHARD,), dtype='f4')
        dset_labels = f.create_dataset("labels", (EXAMPLES_PER_SHARD, 2), dtype='f4')
        dset_k = f.create_dataset("k", (EXAMPLES_PER_SHARD,), dtype='i4')
        dset_x = f.create_dataset("x_sensors", (EXAMPLES_PER_SHARD, 10), dtype='f4')
        dset_h = f.create_dataset("readings", (EXAMPLES_PER_SHARD, 10, 100), dtype='f4')

        try:
            for future in as_completed(futures):
                batch_feat, batch_lab = future.result()
                
                for feat, lab in zip(batch_feat, batch_lab):
                    # Write to current shard
                    idx = current_shard_count
                    dset_rho[idx] = feat['rho']
                    dset_sigma[idx] = feat['sigma']
                    dset_labels[idx] = lab
                    
                    k = len(feat['x_sensors'])
                    dset_k[idx] = k
                    dset_x[idx, :k] = feat['x_sensors']
                    dset_h[idx, :k, :] = feat['readings']
                    
                    current_shard_count += 1
                    pbar.update(1)
                    
                    if current_shard_count >= EXAMPLES_PER_SHARD:
                        f.close()
                        current_shard_idx += 1
                        current_shard_count = 0
                        shard_path = os.path.join(DATA_DIR, f"{SHARD_PREFIX}_{current_shard_idx}.h5")
                        f = h5py.File(shard_path, 'w')
                        dset_rho = f.create_dataset("rho", (EXAMPLES_PER_SHARD,), dtype='f4')
                        dset_sigma = f.create_dataset("sigma", (EXAMPLES_PER_SHARD,), dtype='f4')
                        dset_labels = f.create_dataset("labels", (EXAMPLES_PER_SHARD, 2), dtype='f4')
                        dset_k = f.create_dataset("k", (EXAMPLES_PER_SHARD,), dtype='i4')
                        dset_x = f.create_dataset("x_sensors", (EXAMPLES_PER_SHARD, 10), dtype='f4')
                        dset_h = f.create_dataset("readings", (EXAMPLES_PER_SHARD, 10, 100), dtype='f4')
        finally:
            f.close()
            pbar.close()

if __name__ == "__main__":
    generate_dataset()
