import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Function to extract data from TensorBoard logs
def extract_tensorboard_data(log_dir='stargan/logs'):
    # Dictionary to store all the extracted data
    data = {}
    
    # Get all event files in the log directory
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    # Extract data from each event file
    for event_file in event_files:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                # Create a list for this tag if it doesn't exist
                if v.tag not in data:
                    data[v.tag] = {'step': [], 'value': []}
                
                # Append the step and value
                data[v.tag]['step'].append(e.step)
                data[v.tag]['value'].append(tf.make_ndarray(v.tensor).item())
    
    return data

# Function to plot loss curves
def plot_losses(data, save_dir='loss_plots'):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Group similar losses
    loss_groups = {
        'Discriminator_Losses': ['D/loss_real', 'D/loss_fake', 'D/loss_cls', 'D/loss_gp'],
        'Generator_Losses': ['G/loss_fake', 'G/loss_rec', 'G/loss_cls']
    }
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    # Plot each group
    for group_name, loss_names in loss_groups.items():
        # Check which losses are actually available
        available_losses = [name for name in loss_names if name in data]
        
        if not available_losses:
            continue
            
        plt.figure(figsize=(12, 6))
        
        for i, loss_name in enumerate(available_losses):
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data[loss_name])
            
            # Apply some smoothing for better visualization
            window_size = 50
            if len(df) > window_size:
                df['smooth_value'] = df['value'].rolling(window=window_size).mean()
                # Plot the smoothed curve
                plt.plot(df['step'], df['smooth_value'], label=loss_name, linewidth=2, color=colors[i])
                # Plot the raw data with transparency for reference
                plt.plot(df['step'], df['value'], alpha=0.2, color=colors[i])
            else:
                plt.plot(df['step'], df['value'], label=loss_name, linewidth=2, color=colors[i])
        
        plt.title(f'{group_name} During Training', fontsize=16, fontweight='bold')
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Loss Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f'{group_name}.png'), dpi=300, bbox_inches='tight')
        print(f"Saved {group_name} plot")
        
        # Also create individual plots for each loss
        for i, loss_name in enumerate(available_losses):
            plt.figure(figsize=(10, 5))
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data[loss_name])
            
            # Apply some smoothing for better visualization
            window_size = 50
            if len(df) > window_size:
                df['smooth_value'] = df['value'].rolling(window=window_size).mean()
                # Plot the smoothed curve
                plt.plot(df['step'], df['smooth_value'], label='Smoothed', linewidth=2, color=colors[i])
                # Plot the raw data with transparency
                plt.plot(df['step'], df['value'], alpha=0.2, label='Raw', color=colors[i])
            else:
                plt.plot(df['step'], df['value'], label=loss_name, linewidth=2, color=colors[i])
                
            plt.title(f'{loss_name} During Training', fontsize=16, fontweight='bold')
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('Loss Value', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the figure - replace slashes with underscores for filenames
            safe_filename = loss_name.replace('/', '_')
            plt.savefig(os.path.join(save_dir, f'{safe_filename}.png'), dpi=300, bbox_inches='tight')
            print(f"Saved {loss_name} plot")

# Main execution
if __name__ == "__main__":
    print("Extracting data from TensorBoard logs...")
    data = extract_tensorboard_data()
    print(f"Found {len(data)} different metrics")
    
    print("Generating loss plots...")
    plot_losses(data)
    
    print("Done!")