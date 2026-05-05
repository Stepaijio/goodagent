import os
import h5py
import numpy as np
import pandas as pd

def list_shards(data_dir="data"):
    """Lists all available HDF5 shards in the data directory."""
    shards = sorted([f for f in os.listdir(data_dir) if f.startswith("data_shard") and f.endswith(".h5")])
    return shards

def inspect_example(shard_file, example_idx, data_dir="data"):
    """Loads a specific example from a shard and returns it as a readable format."""
    file_path = os.path.join(data_dir, shard_file)
    
    with h5py.File(file_path, 'r') as f:
        # Basic parameters
        rho = f['rho'][example_idx]
        sigma = f['sigma'][example_idx]
        labels = f['labels'][example_idx]
        k = f['k'][example_idx]
        
        # Sensor data (slicing only active sensors)
        x_sensors = f['x_sensors'][example_idx, :k]
        readings = f['readings'][example_idx, :k, :]
        
    # Prepare a summary table for the parameters
    params_data = {
        "Parameter": ["Density (rho)", "Surface Tension (sigma)", "Viscosity (mu) [Target]", "Final Thickness (h_final) [Target]"],
        "Value": [f"{rho:.4f}", f"{sigma:.4f}", f"{labels[0]:.6f}", f"{labels[1]:.6f}"]
    }
    df_params = pd.DataFrame(params_data)
    
    # Prepare a table for the sensors
    sensor_rows = []
    for i in range(k):
        h_series = readings[i, :]
        sensor_rows.append({
            "Sensor ID": i + 1,
            "Position (x)": f"{x_sensors[i]:.4f}",
            "Mean h": f"{np.mean(h_series):.6f}",
            "Max h": f"{np.max(h_series):.6f}",
            "Min h": f"{np.min(h_series):.6f}",
            "Std Dev": f"{np.std(h_series):.6f}"
        })
    df_sensors = pd.DataFrame(sensor_rows)
    
    return df_params, df_sensors, x_sensors, readings

def main():
    data_dir = "C:/Users/UserSK/Desktop/goodagent/data"
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Error: No data found in {data_dir}. Please run the generator first.")
        return

    shards = list_shards(data_dir)
    print("\nAvailable Shards:")
    for i, shard in enumerate(shards):
        print(f"{i}: {shard}")
    
    try:
        shard_idx = int(input("\nSelect shard index: "))
        if shard_idx < 0 or shard_idx >= len(shards):
            print("Invalid shard index.")
            return
        
        selected_shard = shards[shard_idx]
        
        # Get total examples in this shard
        with h5py.File(os.path.join(data_dir, selected_shard), 'r') as f:
            total_examples = f['rho'].shape[0]
            
        print(f"Shard {selected_shard} contains {total_examples} examples.")
        example_idx = int(input(f"Select example index (0 to {total_examples-1}): "))
        
        if example_idx < 0 or example_idx >= total_examples:
            print("Invalid example index.")
            return
            
        df_params, df_sensors, x_sensors, readings = inspect_example(selected_shard, example_idx, data_dir)
        
        print("\n" + "="*40)
        print(" GLOBAL PARAMETERS")
        print("="*40)
        print(df_params.to_string(index=False))
        
        print("\n" + "="*60)
        print(" SENSOR DATA SUMMARY")
        print("="*60)
        print(df_sensors.to_string(index=False))
        
        print("\n" + "="*60)
        print(" TIME SERIES DATA")
        print("="*60)
        for i in range(len(x_sensors)):
            print(f"Sensor {i+1} at x={x_sensors[i]:.4f}:")
            print(readings[i, :])
            print("-" * 30)
        print("="*60)
        
    except ValueError:
        print("Error: Please enter a valid number.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
