# How to Use the Output of main_led_nba.py

## Overview

When you run `python main_led_nba.py --cfg led_augment --gpu 0 --train 0 --info reproduce`, the script evaluates the model and produces several types of outputs:

## 1. Performance Metrics (Log File)

### Location
- **Path**: `results/led_augment/{info}/log/log.txt`
- **For your command**: `results/led_augment/reproduce/log/log.txt`

### Content
The log file contains:
- Model architecture information (number of parameters)
- Processing progress (batches processed)
- **Performance metrics**:
  - **ADE (Average Displacement Error)**: Average error over the entire predicted trajectory
  - **FDE (Final Displacement Error)**: Error at the final predicted position
  - Metrics are reported for 1s, 2s, 3s, and 4s prediction horizons

### Example Output
```
--ADE(1s): 0.1764	--FDE(1s): 0.2691
--ADE(2s): 0.3691	--FDE(2s): 0.5642
--ADE(3s): 0.5818	--FDE(3s): 0.8367
--ADE(4s): 0.8095	--FDE(4s): 1.0959
```

### How to Use
1. **Read the log file**:
   ```python
   with open('results/led_augment/reproduce/log/log.txt', 'r') as f:
       content = f.read()
       print(content)
   ```

2. **Parse metrics programmatically**:
   ```python
   import re
   
   log_path = 'results/led_augment/reproduce/log/log.txt'
   with open(log_path, 'r') as f:
       lines = f.readlines()
       
   for line in lines:
       if 'ADE' in line and 'FDE' in line:
           # Extract ADE and FDE values
           ade_match = re.search(r'ADE\((\d+)s\):\s+([\d.]+)', line)
           fde_match = re.search(r'FDE\((\d+)s\):\s+([\d.]+)', line)
           if ade_match and fde_match:
               time_horizon = ade_match.group(1)
               ade = float(ade_match.group(2))
               fde = float(fde_match.group(2))
               print(f"At {time_horizon}s: ADE={ade:.4f}, FDE={fde:.4f}")
   ```

## 2. Console Output

The script prints the same metrics to the console in real-time. You can:
- Redirect to a file: `python main_led_nba.py ... > output.txt 2>&1`
- Use in scripts: Parse the stdout stream

## 3. Saving Predictions for Visualization

Currently, the `test_single_model()` function only evaluates and logs metrics. To save predictions for visualization, you have two options:

### Option A: Modify test_single_model() to Save Predictions

Add code to save predictions at the end of the test loop:

```python
# In test_single_model() function, after the loop:
save_dir = os.path.join(self.cfg.log_dir, 'predictions')
os.makedirs(save_dir, exist_ok=True)

# Save predictions (you'll need to collect them during the loop)
torch.save({
    'predictions': all_predictions,
    'ground_truth': all_ground_truth,
    'past_trajectories': all_past_traj
}, os.path.join(save_dir, 'test_predictions.pt'))
```

### Option B: Use the Existing save_data() Function

The trainer has a `save_data()` function that saves visualization data. To use it:

1. **Uncomment the save_data() call** in `main_led_nba.py`:
   ```python
   if config.train==1:
       t.fit()
   else:
       t.save_data()  # Uncomment this
       # t.test_single_model()
   ```

2. **Run the script**: It will save visualization data to `visualization/data/`:
   - `past.pt`: Past trajectories
   - `future.pt`: Ground truth future trajectories
   - `prediction.pt`: Model predictions
   - `p_mean.pt`: Mean estimation
   - `p_sigma.pt`: Variance estimation
   - `p_mean_denoise.pt`: Denoised mean predictions

3. **Visualize using the notebook**: Use `visualization/draw_mean_variance.ipynb` to plot the results

## 4. Using Saved Predictions

### Load and Analyze Predictions

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load saved predictions (if you modified test_single_model to save them)
data = torch.load('results/led_augment/reproduce/log/predictions/test_predictions.pt')
predictions = data['predictions']  # Shape: [batch_size, num_samples, time_steps, 2]
ground_truth = data['ground_truth']

# Calculate custom metrics
def custom_metric(pred, gt):
    # Your custom evaluation
    pass

# Visualize trajectories
def plot_trajectories(past, pred, gt):
    plt.figure(figsize=(10, 10))
    # Plot past trajectory
    plt.plot(past[:, 0], past[:, 1], 'b-', label='Past')
    # Plot predicted trajectories
    for i in range(pred.shape[0]):
        plt.plot(pred[i, :, 0], pred[i, :, 1], 'r--', alpha=0.3)
    # Plot ground truth
    plt.plot(gt[:, 0], gt[:, 1], 'g-', label='Ground Truth')
    plt.legend()
    plt.show()
```

## 5. Integration into Other Scripts

### Use the Trained Model in Your Code

```python
from trainer.train_led_trajectory_augment_input import Trainer
import argparse

# Create config (minimal)
class Config:
    cfg = 'led_augment'
    info = 'reproduce'
    gpu = 0
    cuda = True
    learning_rate = 0.002

config = Config()
trainer = Trainer(config)

# Load trained model
model_path = './results/checkpoints/led_new.p'
model_dict = torch.load(model_path, map_location='cpu')['model_initializer_dict']
trainer.model_initializer.load_state_dict(model_dict)

# Make predictions on new data
trainer.model_initializer.eval()
with torch.no_grad():
    # Prepare your data (same format as in dataloader)
    # past_traj shape: [num_agents, past_frames, 2]
    # traj_mask shape: [num_agents, num_agents]
    
    sample_prediction, mean_estimation, variance_estimation = trainer.model_initializer(past_traj, traj_mask)
    # Process and use predictions...
```

## 6. Comparing Different Models

To compare results from different runs:

```python
import os
import glob

def extract_metrics(log_path):
    """Extract ADE and FDE from log file"""
    metrics = {}
    with open(log_path, 'r') as f:
        for line in f:
            if 'ADE' in line and 'FDE' in line:
                # Parse line...
                pass
    return metrics

# Find all log files
log_files = glob.glob('results/led_augment/*/log/log.txt')

# Compare metrics
comparison = {}
for log_file in log_files:
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(log_file)))
    comparison[exp_name] = extract_metrics(log_file)

# Print comparison
for exp_name, metrics in comparison.items():
    print(f"{exp_name}: ADE(4s)={metrics.get('ADE_4s', 'N/A')}")
```

## 7. Export Results to CSV/JSON

```python
import json
import csv
import re

def export_metrics_to_csv(log_path, output_csv):
    """Export metrics from log file to CSV"""
    metrics = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'ADE' in line and 'FDE' in line:
                # Extract metrics
                matches = re.findall(r'(ADE|FDE)\((\d+)s\):\s+([\d.]+)', line)
                row = {}
                for metric_type, time, value in matches:
                    row[f'{metric_type}_{time}s'] = float(value)
                if row:
                    metrics.append(row)
    
    if metrics:
        # Write to CSV
        fieldnames = metrics[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

# Usage
export_metrics_to_csv(
    'results/led_augment/reproduce/log/log.txt',
    'results/led_augment/reproduce/metrics.csv'
)
```

## Quick Reference

| Output Type | Location | Format | Usage |
|------------|----------|--------|-------|
| Performance Metrics | `results/{cfg}/{info}/log/log.txt` | Text | Read directly or parse programmatically |
| Console Output | stdout | Text | Redirect to file or parse |
| Model Checkpoints | `results/checkpoints/*.p` | PyTorch | Load for inference |
| Visualization Data | `visualization/data/*.pt` | PyTorch tensors | Use with visualization notebook |

## Tips

1. **Monitor Progress**: The log file includes progress updates every 10 batches when running with `--train 0`
2. **Multiple Experiments**: Use different `--info` values to track multiple experiments
3. **Custom Metrics**: Modify `test_single_model()` to calculate and save custom evaluation metrics
4. **Batch Processing**: If you need to process predictions, collect them in a list during the test loop

