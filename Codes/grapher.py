import matplotlib.pyplot as plt
import pandas as pd

def save_tuning(results, model_name, param_grid):
    df = pd.DataFrame(results)
    
    # Dynamically extract parameter keys from param_grid
    param_keys = list(param_grid.keys())
    
    # Create short display names for each parameter
    short_names = {
        'hidden_size': 'H',
        'num_layers': 'L', 
        'dropout': 'D',
        'lr': 'LR',
        'batch_size': 'B',
        'n_heads': 'NH'  # For Transformer
    }
    
    # Map indices back to actual values dynamically for all parameters
    display_cols = {}
    for key in param_keys:
        col_name = short_names.get(key, key[:3].upper())  # Use short name or first 3 chars
        display_cols[key] = col_name
        df[col_name] = df[key].apply(lambda x, k=key: param_grid[k][x] if isinstance(x, int) else x)
    
    df['acc'] = df['val_accuracy'].apply(lambda x: f"{x:.4f}")

    # --- FIGURE 1: BAR CHART (Stability across folds) ---
    plt.figure(figsize=(10, 6))
    top_3 = df.nlargest(3, 'val_accuracy')
    
    # Build x_labels dynamically based on available parameters
    def build_label(row, idx):
        parts = [f"#{idx+1}:"]
        for key in param_keys:
            col_name = display_cols[key]
            parts.append(f"{col_name}={row[col_name]}")
        return ", ".join(parts[:3]) + ",\n" + ", ".join(parts[3:])  # Split for readability
    
    x_labels = [build_label(row, i) for i, (_, row) in enumerate(top_3.iterrows())]
    
    fold_data = list(top_3['per_fold_accuracies'])
    num_folds = len(fold_data[0])
    bar_width = 0.15
    
    for f_idx in range(num_folds):
        scores = [config[f_idx] for config in fold_data]
        # Capture the bars in a variable called 'container'
        container = plt.bar([i + (f_idx * bar_width) for i in range(len(top_3))], 
                            scores, 
                            width=bar_width, 
                            label=f'Fold {f_idx+1}')
        
        # Add labels on top of each bar in this fold
        plt.bar_label(container, fmt='%.3f', padding=3, fontsize=8, rotation=90)

    # Adjusting ylim slightly higher to make room for the labels
    plt.ylim(0, 1.15) 

    plt.title(f'{model_name} Cross-Validation Stability (Top 3)')
    plt.ylabel('Accuracy')
    plt.xticks([i + bar_width * (num_folds/2 - 0.5) for i in range(len(top_3))], x_labels)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{model_name}_hypertuner_barChart.png")
    plt.close()

    # --- FIGURE 2: SUMMARY TABLE ---
    # Sorting results by accuracy for the final report summary
    # Use dynamic column names from display_cols
    table_cols = [display_cols[key] for key in param_keys] + ['acc']
    full_table_df = df.sort_values('val_accuracy', ascending=False)[table_cols]
    table_data = full_table_df.values
    columns = [display_cols[key] for key in param_keys] + ['Mean Acc']
    
    # Dynamic height adjustment to prevent row overlap in long tuning sessions
    fig_height = max(4, len(table_data) * 0.4) 
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')
    
    the_table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1.2, 1.8) # Adjust scaling for professional presentation
    
    plt.title(f'{model_name} Full Hyperparameter Summary', pad=20)
    
    plt.savefig(f"{model_name}_hypertuner_table.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visuals saved: {model_name}_hypertuner_barChart.png and {model_name}_hypertuner_table.png")

def plot_training_curves(model_name, train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Training Results: {model_name}', fontsize=16)

    # 1. Loss Curve
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 2. Accuracy Curve
    ax2.plot(epochs, train_accs, 'g-s', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'm-s', label='Validation Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{model_name}_learning_curves.png")
    plt.show()
    print(f"Graph saved as {model_name}_learning_curves.png")


import matplotlib.pyplot as plt

def plot_guess_breakdown(model_name, stats_summary):
    emotions = [s['emotion'] for s in stats_summary]
    
    # Define fixed colors for each emotion
    emotion_colors = {
        "ANG": "red",
        "DIS": "green",
        "FEA": "purple",
        "HAP": "yellow",
        "NEU": "gray",
        "SAD": "blue"
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, pred_emotion in enumerate(emotions):
        actual_composition = stats_summary[i]['confusion_col']
        
        bottom = 0
        for j, actual_emotion_name in enumerate(emotions):
            count = actual_composition[j]
            if count > 0:
                # Use the mapped color for the actual emotion
                color = emotion_colors.get(actual_emotion_name, "black")
                
                bar = ax.bar(pred_emotion, count, bottom=bottom, color=color,
                             label=actual_emotion_name if i == 0 else "")
                
                # Add text labels for segments that represent > 5% of the total bar height
                if count > (sum(actual_composition) * 0.05):
                    # Set text color based on bar color for readability
                    text_color = "black" if color == "yellow" else "white"
                    ax.text(pred_emotion, bottom + count/2, str(int(count)), 
                            ha='center', va='center', color=text_color, fontweight='bold')
                
                bottom += count

    ax.set_title(f'Composition of Model Predictions: {model_name}', fontsize=15)
    ax.set_ylabel('Number of Guesses')
    ax.set_xlabel('What the Model Guessed')
    
    # Clean up legend handles to avoid duplicates
    handles, labels = ax.get_legend_handles_labels()
    # Ensure legend entries match the order of defined colors
    ax.legend(handles, labels, title="Actual Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_guess_breakdown.png")
    plt.show()

def plot_success_rates(model_name, stats_summary):
    emotions = [s['emotion'] for s in stats_summary]
    rates = [s['success_rate'] for s in stats_summary]
    
    # Extract correct and total from the confusion data stored in the summary
    # Index 'i' is the emotion's position in the 0-5 list
    correct_per_class = [s['confusion_col'][i] for i, s in enumerate(stats_summary)]
    total_per_class = [sum(s['confusion_row']) for s in stats_summary]
    
    total_correct = sum(correct_per_class)
    total_samples = sum(total_per_class)
    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0

    plt.figure(figsize=(10, 6))
    
    # Emotion color mapping you requested
    emotion_colors = ["red", "green", "purple", "yellow", "gray", "blue"]
    bars = plt.bar(emotions, rates, color=emotion_colors, edgecolor='black', alpha=0.8)
    
    # Horizontal line for the simple average of the success rates
    avg_rate = sum(rates) / len(rates)
    plt.axhline(y=avg_rate, color='r', linestyle='--', label=f'Mean Recall: {avg_rate:.1f}%')
    
    # Horizontal line for the TOTAL accuracy across the whole dataset
    plt.axhline(y=overall_accuracy, color='black', linestyle='-', linewidth=2, 
                label=f'Total Success Rate: {overall_accuracy:.1f}%')
    
    plt.title(f'Success Rate (Recall) Per Emotion: {model_name}', fontsize=14)
    plt.ylabel('Percentage Correct (%)')
    plt.ylim(0, 115) 
    
    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', 
                 ha='center', va='bottom', fontweight='bold')

    # Add text box for data totals
    stats_text = f"Total Samples: {total_samples}\nTotal Correct: {total_correct}"
    plt.text(len(emotions)-0.5, 110, stats_text, ha='right', va='top', 
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))

    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{model_name}_success_rates.png")
    plt.show()