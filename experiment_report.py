import os
import json
import csv
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import glob
import sys
import os
import time
# Use LaTeX and IEEE-style font settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

def create_full_table_sorted_by_f1():
    # Create a copy of the dataframe with algorithm names mapped
    display_df = df.copy()
    display_df['algorithm'] = display_df['algorithm'].replace({'nsgaii': 'NSA-NSGA-II', 'ga': 'NSA-GA'})
    
    # Round detectors count to nearest integer
    display_df['detectors_count_avg'] = display_df['detectors_count_avg'].round().astype(int)
    
    # Sort by F1 score descending
    sorted_df = display_df.sort_values('f1_avg', ascending=False)
    
    # Select only the relevant columns
    cols = ['algorithm', 'dataset', 'dimension', 'f1_avg', 'precision_avg', 'recall_avg', 'accuracy_avg', 'detectors_count_avg']
    sorted_df = sorted_df[cols]
    
    # Rename columns for better readability
    sorted_df.columns = ['Algorithm', 'Embedding', 'Dim', 'F1-score', 'Precision', 'Recall', 'Accuracy', 'Detectors']
    
    # Save to CSV
    sorted_df.to_csv('report/results/full_results_by_f1.csv', index=False)
    # Save table to LaTeX format
    with open('report/results/full_results_by_f1.tex', 'w') as tf:
        latex_str = sorted_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}")
        # Replace the LaTeX table structure with the specified format
        latex_str = latex_str.replace('\\begin{tabular}', '\\begin{table*}[h]\n    \\centering\n    \\tiny\n    \\resizebox{\\textwidth}{!}{\n    \\begin{tabular}')
        latex_str = latex_str.replace('\\end{tabular}', '\\end{tabular}\n    }\n\\end{table*}')
        # Remove the original table environment since we're creating our own
        latex_str = latex_str.replace('\\begin{table}', '')
        latex_str = latex_str.replace('\\end{table}', '')
        # Write to file
        tf.write(latex_str)
    print("\nAll Results Sorted by F1-score:")
    print(sorted_df.to_string(index=False))
    
    return sorted_df

def create_negative_space_coverage_plot():
    """
    Plots negative space coverage grouped by dimension with algorithms shown side by side.
    Creates a grouped bar chart comparing negative space coverage across algorithms and dimensions.
    """
    # Create directory for results if it doesn't exist
    os.makedirs('report/results', exist_ok=True)
    
    # Load the data
    csv_data = pd.read_csv('report/results/averaged_results.csv')
    
    # Prepare data for plotting
    csv_data['algorithm'] = csv_data['algorithm'].replace({
        'nsgaii': 'NSA-NSGA-II',
        'ga': 'NSA-GA'
    })
    
    # Sort by dimension
    dim_order = {'1D': 1, '2D': 2, '3D': 3, '4D': 4}
    csv_data['dim_order'] = csv_data['dimension'].map(dim_order)
    csv_data = csv_data.sort_values(['dim_order', 'algorithm'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set width of bars and positions
    bar_width = 0.35
    dimensions = sorted(csv_data['dimension'].unique(), key=lambda x: dim_order[x])
    algorithms = csv_data['algorithm'].unique()
    
    # Set up x positions for grouped bars
    x = np.arange(len(dimensions))
    
    # Colors for algorithms
    colors = {'NSA-NSGA-II': 'steelblue', 'NSA-GA': 'darkred'}
    
    # Plot bars for each algorithm side by side
    for i, algo in enumerate(algorithms):
        algo_data = csv_data[csv_data['algorithm'] == algo]
        values = []
        
        # Get values in dimension order
        for dim in dimensions:
            val = algo_data[algo_data['dimension'] == dim]['negative_space_coverage_avg'].values
            values.append(val[0] if len(val) > 0 else 0)
        
        # Plot bars with offset
        offset = (i - 0.5*(len(algorithms)-1)) * bar_width
        bars = ax.bar(x + offset, values, bar_width * 0.9, label=algo, color=colors[algo])
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height * 1.01,
                   f'{int(height):,}', 
                   ha='center', va='bottom', 
                   fontsize=8, rotation=0)
    
    # Add labels and title
    ax.set_xlabel(r'\textbf{Dimension}', fontsize=12)
    ax.set_ylabel(r'\textbf{Negative Space Coverage}', fontsize=12)
    ax.set_title(r'\textbf{Negative Space Coverage by Dimension and Algorithm}', fontsize=14)
    
    # Set x-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    
    # Add grid and legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_yscale('log')  # Use log scale for better visibility
    ax.legend(loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('report/results/negative_space_coverage_by_dim_algo.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('report/results/negative_space_coverage_by_dim_algo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created negative space coverage plot grouped by dimension with algorithms side by side.")

# Function to create best overall metrics table
def create_best_overall_table():
    # Create a copy of the dataframe with algorithm names mapped
    display_df = df.copy()
    display_df['algorithm'] = display_df['algorithm'].replace({'nsgaii': 'NSA-NSGA-II', 'ga': 'NSA-GA'})
    
    # Round detectors count to nearest integer
    display_df['detectors_count_avg'] = display_df['detectors_count_avg'].round().astype(int)
    
    # Get the top 5 by F1-score
    top_f1 = display_df.sort_values('f1_avg', ascending=False).head(5)
    
    # Select and reorder columns
    cols = ['algorithm', 'dataset', 'dimension', 'f1_avg', 'precision_avg', 'recall_avg', 'accuracy_avg', 'detectors_count_avg']
    best_metrics = top_f1[cols]
    
    # Rename columns for better readability (using "Dim" instead of "Dimension")
    best_metrics.columns = ['Algorithm', 'Embedding', 'Dim', 'F1-score', 'Precision', 'Recall', 'Accuracy', 'Detectors']
    
    # Save to CSV
    best_metrics.to_csv('report/results/top_5_f1.csv', index=False)
    
    # Save to LaTeX format with the specified structure
    with open('report/results/top_5_f1.tex', 'w') as tf:
        latex_str = best_metrics.to_latex(index=False, float_format=lambda x: f"{x:.3f}")
        # Replace the LaTeX table structure with the specified format
        latex_str = latex_str.replace('\\begin{tabular}', '\\begin{table*}[h]\n    \\centering\n    \\tiny\n    \\resizebox{\\textwidth}{!}{\n    \\begin{tabular}')
        latex_str = latex_str.replace('\\end{tabular}', '\\end{tabular}\n    }\n    \\caption{Top 5 configurations by F1-score}\n\\end{table*}')
        # Remove the original table environment since we're creating our own
        latex_str = latex_str.replace('\\begin{table}', '')
        latex_str = latex_str.replace('\\end{table}', '')
        # Write to file
        tf.write(latex_str)
    
    print("\nTop 5 by F1-score:")
    print(best_metrics.to_string(index=False))
    
    return best_metrics

# Function to create tables per embedding
def create_per_embedding_tables():
    datasets = df['dataset'].unique()
    results = {}
    
    # Create a custom dimension order
    dimension_order = {'1D': 0, '2D': 1, '3D': 2, '4D': 3}
    
    for dataset in datasets:
        # Filter for current dataset
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # Map algorithm names
        dataset_df['algorithm'] = dataset_df['algorithm'].replace({'nsgaii': 'NSA-NSGA-II', 'ga': 'NSA-GA'})
        
        # Round detectors count to nearest integer
        dataset_df['detectors_count_avg'] = dataset_df['detectors_count_avg'].round().astype(int)
        
        # First sort by algorithm, then by dimension in custom order
        dataset_df['dim_order'] = dataset_df['dimension'].map(dimension_order)
        dataset_df = dataset_df.sort_values(['algorithm', 'dim_order'], ascending=[True, True])
        dataset_df = dataset_df.drop(columns=['dim_order'])
        
        # Select only the relevant columns
        cols = ['algorithm', 'dimension', 'f1_avg', 'precision_avg', 'recall_avg', 'accuracy_avg', 'detectors_count_avg']
        dataset_df = dataset_df[cols]
        
        # Rename columns for better readability
        dataset_df.columns = ['Algorithm', 'Dim', 'F1-score', 'Precision', 'Recall', 'Accuracy', 'Detectors']
        
        # Save to CSV
        dataset_df.to_csv(f'report/results/{dataset}_results.csv', index=False)
        
        # Save table to LaTeX format
        with open(f'report/results/{dataset}_results.tex', 'w') as tf:
            latex_str = dataset_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}")
            # Replace the LaTeX table structure with the specified format
            latex_str = latex_str.replace('\\begin{tabular}', '\\begin{table*}[h]\n    \\centering\n    \\tiny\n    \\resizebox{\\textwidth}{!}{\n    \\begin{tabular}')
            latex_str = latex_str.replace('\\end{tabular}', '\\end{tabular}\n    }\n    \\caption{' + dataset + ' Caption}\n\\end{table*}')
            # Remove the original table environment since we're creating our own
            latex_str = latex_str.replace('\\begin{table}', '')
            latex_str = latex_str.replace('\\end{table}', '')
            # Write to file
            tf.write(latex_str)
        
        print(f"\nResults for {dataset} Embedding:")
        print(dataset_df.to_string(index=False))
        
        results[dataset] = dataset_df
    
    return results

def create_boxplot_by_dimension(column, naming):
    """
    Generates a LaTeX-formatted boxplot grouped by dimensionality (1D to 4D),
    comparing all algorithm-embedding combinations. Outputs high-resolution PNG and PDF.
    """

    # Prepare DataFrame
    display_df = df.copy()
    display_df['algorithm'] = display_df['algorithm'].replace({
        'nsgaii': 'NSA-NSGA-II',
        'ga': 'NSA-GA'
    })
    display_df['algorithm_embedding'] = display_df['algorithm'] + ' - ' + display_df['dataset']

    # Collect data per dimension
    dimensions = ['1D', '2D', '3D', '4D']
    boxplot_data = [display_df[display_df['dimension'] == dim][column] for dim in dimensions]

    # Create figure
    plt.figure(figsize=(6, 6))
    bp = plt.boxplot(boxplot_data, patch_artist=True, notch=False, widths=0.6)

    # Style the boxplot
    for box in bp['boxes']:
        box.set(facecolor='lightgray', edgecolor='black', linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=4, markerfacecolor='black',
                    markeredgecolor='black', alpha=0.5)

    # Add labels and layout
    plt.xlabel(r'\textbf{Dimension}')
    plt.ylabel(r'\textbf{' + naming + '}')
    plt.title(r'\textbf{Impact of Dimensionality on ' + naming + '}', pad=10)
    plt.xticks(ticks=range(1, len(dimensions) + 1), labels=dimensions)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save figure
    plt.savefig(f'report/results/{column}_boxplot_by_dimension.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'report/results/{column}_boxplot_by_dimension.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Created LaTeX-styled detector count boxplot by dimension.")

def create_boxplot_by_embedding(column, naming):
    """
    Generates a LaTeX-formatted boxplot grouped by embedding models (RoBERTa, BERT, DistilBERT, FastText),
    comparing the algorithms. Outputs high-resolution PNG and PDF.
    """

    # Prepare DataFrame
    display_df = df.copy()
    display_df['algorithm'] = display_df['algorithm'].replace({
        'nsgaii': 'NSA-NSGA-II',
        'ga': 'NSA-GA'
    })

    # Collect data per embedding model
    embedding_models = sorted(display_df['dataset'].unique())
    boxplot_data = [display_df[display_df['dataset'] == model][column] for model in embedding_models]

    # Create figure
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(boxplot_data, patch_artist=True, notch=False, widths=0.6)

    # Style the boxplot
    for box in bp['boxes']:
        box.set(facecolor='lightgray', edgecolor='black', linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=4, markerfacecolor='black',
                    markeredgecolor='black', alpha=0.5)

    # Add labels and layout
    plt.xlabel(r'\textbf{Embedding Model}')
    plt.ylabel(r'\textbf{' + naming + '}')
    plt.title(r'\textbf{Impact of Embedding Model on ' + naming + '}', pad=10)
    plt.xticks(ticks=range(1, len(embedding_models) + 1), labels=embedding_models, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save figure
    plt.savefig(f'report/results/{column}_boxplot_by_embedding.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'report/results/{column}_boxplot_by_embedding.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created LaTeX-styled {naming} boxplot by embedding model.")

def create_boxplot_by_algorithm(column, naming):
    """
    Generates a LaTeX-formatted boxplot comparing the algorithms (NSA-GA and NSA-NSGA-II).
    Shows the impact of algorithm selection on performance metrics.
    Outputs high-resolution PNG and PDF.
    """

    # Prepare DataFrame
    display_df = df.copy()
    display_df['algorithm'] = display_df['algorithm'].replace({
        'nsgaii': 'NSA-NSGA-II',
        'ga': 'NSA-GA'
    })

    # Collect data per algorithm
    algorithms = sorted(display_df['algorithm'].unique())
    boxplot_data = [display_df[display_df['algorithm'] == algo][column] for algo in algorithms]

    # Create figure
    plt.figure(figsize=(6, 6))
    bp = plt.boxplot(boxplot_data, patch_artist=True, notch=False, widths=0.6)

    # Style the boxplot
    for box in bp['boxes']:
        box.set(facecolor='lightgray', edgecolor='black', linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=4, markerfacecolor='black',
                  markeredgecolor='black', alpha=0.5)

    # Add labels and layout
    plt.xlabel(r'\textbf{Algorithm}')
    plt.ylabel(r'\textbf{' + naming + '}')
    plt.title(r'\textbf{Impact of Algorithm on ' + naming + '}', pad=10)
    plt.xticks(ticks=range(1, len(algorithms) + 1), labels=algorithms)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save figure
    plt.savefig(f'report/results/{column}_boxplot_by_algorithm.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'report/results/{column}_boxplot_by_algorithm.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created LaTeX-styled {naming} boxplot comparing algorithms.")

def plot_f1_precision_recall_negative_space_per_detector_amount():
    """
    Plots the F1, precision, recall evolution and negative space coverage evolution
    for each 25th detector, for each unique combination of algorithm, embedding model, and dimension.
    Uses dual y-axis to show negative space coverage on the right axis.
    Creates separate LaTeX files for each embedding model.
    """
    # Directory with experiment results
    path = 'model/detector'
    
    # Get all experiment result files
    files = [f for f in os.listdir(path) if 'experiment_result' in f and '-1' not in f]
    
    # Create directory for plots
    os.makedirs('report/results/evolution_plots', exist_ok=True)
    
    # Dictionary to store data for each combination
    evolution_data = {}
    
    # Process each file
    for file_name in files:
        # Parse file name to get algorithm, dataset, dimension
        if 'nsgaii' in file_name:
            algo = 'NSA-NSGA-II'
        else:
            algo = 'NSA-GA'
        
        f_split = file_name.split('_')
        dataset = f_split[1]
        dim = f'{f_split[2][0]}D'
        
        # Create unique identifier for this configuration
        config = f"{algo}_{dataset}_{dim}"
        
        # Read the data
        with open(os.path.join(path, file_name), 'r') as file:
            data = json.load(file)
            #print('Processing', file_name)
            # Check if the validation lists exist and are not empty
            if ('validation_precision_list' in data and
                'validation_recall_list' in data and
                'validation_negative_space_coverage_list' in data and
                data['validation_precision_list'] and
                data['validation_recall_list'] and
                data['validation_negative_space_coverage_list']):
                
                # Get validation data
                precision_list = data['validation_precision_list']
                recall_list = data['validation_recall_list']
                negative_space_list = data['validation_negative_space_coverage_list']
                
                # Calculate F1 scores
                f1_list = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                            for p, r in zip(precision_list, recall_list)]
                
                # Create detector counts (every 25th detector)
                detector_counts = [(i+1)*25 for i in range(len(precision_list))]
                # Store the data
                # Check if this configuration already exists
                if config not in evolution_data:
                    # First occurrence - store the data directly
                    evolution_data[config] = {
                        'dataset': dataset,
                        'algo': algo,
                        'dim': dim,
                        'detector_counts': detector_counts,
                        'precision': precision_list,
                        'recall': recall_list,
                        'f1': f1_list,
                        'negative_space': negative_space_list,
                        'test_detector_count': data['test_detectors_count'],
                        'count': 1  # Track number of samples for averaging
                    }
    
    # Create plots for each configuration with dual y-axis
    for config, data in evolution_data.items():
        algo = data['algo']
        dataset = data['dataset']
        dim = data['dim']
        
        # Create figure with dual y-axis
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()
        # Mark the final detector count used for testing with a vertical line
        test_detector_count = data['test_detector_count']
        ax1.axvline(x=test_detector_count, color='black', linestyle='-.', linewidth=1.5, 
                    label=f'Convergence ({test_detector_count})')
        # Plot F1, Precision, Recall on left axis
        ax1.plot(data['detector_counts'], data['precision'], 'b-', label='Precision', linewidth=1.5)
        ax1.plot(data['detector_counts'], data['recall'], 'r-', label='Recall', linewidth=1.5)
        ax1.plot(data['detector_counts'], data['f1'], 'g-', label='F1-score', linewidth=1.5)
        
        # Plot Negative Space Coverage on right axis
        ax2.plot(data['detector_counts'], data['negative_space'], 'k--', label='Negative Space', linewidth=1.5)
        
        # Set y-axis limits to start from the minimum value of metrics
        min_metric = min(min(data['precision']), min(data['recall']), min(data['f1']))
        # Add a small padding (5% of the range) to avoid cutting off plot lines
        padding = (1.0 - min_metric) * 0.05
        ax1.set_ylim([max(0, min_metric - padding), 1.05])
        
        # Find min and max for negative space to set appropriate scale
        min_ns = min(data['negative_space'])
        max_ns = max(data['negative_space'])
        # Add padding to avoid overlap with the axis
        padding = (max_ns - min_ns) * 0.1
        ax2.set_ylim([max(0, min_ns - padding), max_ns + padding])
        # Fix left y-axis max to 1.0
        ax1.set_ylim([max(0, min_metric - padding), 1.0])
        
        # Labels and title for left axis
        ax1.set_xlabel(r'\textbf{Number of Detectors}')
        ax1.set_ylabel(r'\textbf{Score}')
        
        # Label for right axis
        ax2.set_ylabel(r'\textbf{Negative Space Coverage}')
        
        # Title and grid
        plt.title(f"\\textbf{{Performance Evolution - {algo}, {dataset}, {dim}}}")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Legend combining both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'report/results/evolution_plots/combined_{algo}_{dataset}_{dim}.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(f'report/results/evolution_plots/combined_{algo}_{dataset}_{dim}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Group plots by embedding model
    embedding_plots = {}
    for config, data in evolution_data.items():
        dataset = data['dataset']
        if dataset not in embedding_plots:
            embedding_plots[dataset] = []
        embedding_plots[dataset].append(config)
    
    # Ensure the output directories exist
    os.makedirs('report/results', exist_ok=True)
    os.makedirs('report/results/evolution_plots', exist_ok=True)
    
    # Generate separate LaTeX code for each embedding model
    for embedding, configs in embedding_plots.items():
        plots = [f"combined_{config}.png" for config in configs]
        
        # Calculate number of rows needed (2 plots per row)
        num_plots = len(plots)
        num_rows = (num_plots + 1) // 2  # Ceiling division
        
        # For a one-column thesis page, we need more compact figure layout
        latex_code = f"""
        \\begin{{figure}}[htbp]
        \\centering"""
        
        # Add each subfigure with reduced spacing
        for i, plot in enumerate(plots):
            config = plot.replace('combined_', '').replace('.png', '')
            algo, _, dim = config.split('_')
            row = i // 2
            col = i % 2
            
            # Add less vertical space between rows
            if col == 0 and i > 0:
                latex_code += "\n    \\vspace{0.2cm}\n"
            
            # Make subfigures slightly smaller
            width = "0.40"
            
            latex_code += f"""    \\begin{{subfigure}}[b]{{{width}\\textwidth}}
        \\includegraphics[width=\\textwidth]{{images/plots/evolution_plots/{plot}}}
        \\caption{{\\scriptsize {algo}, {dim}}}
        \\label{{subfig:{embedding}_{algo}_{dim}}}
        \\end{{subfigure}}"""
            
            # Less horizontal space between columns
            if col < 1 and i < num_plots - 1:
                latex_code += "\\hspace{0.05\\textwidth}\n"
            else:
                latex_code += "\n"
            
        # More compact caption
        latex_code += f"""    \\caption{{\\small Example runs showing performance metrics and negative space coverage for NSA-GA and NSGA-II for {embedding}.}}
        \\label{{fig:evolution_plots_{embedding}}}
        \\end{{figure}}
        """
        # Ensure the directory exists
        os.makedirs('report/results', exist_ok=True)
        
        # Save LaTeX code to file
        with open(f'report/results/evolution_plots/evolution_plots_{embedding}_figure.tex', 'w') as f:
            f.write(latex_code)
            
    print(f"Created combined evolution plots for {len(evolution_data)} configurations.")
    print(f"Created separate LaTeX files for {len(embedding_plots)} embedding models.")


def plot_detector_models():
    """
    Plots all existing detector models from auto.json files.
    Executes the algorithm for each detector file without creating new detectors.
    Covers all algorithms, embedding models, and dimensions.
    """
    
    # Find all auto.json detector files
    detector_files = glob.glob('model/detector/*_auto_0.json')

    print(f"Found {len(detector_files)} detector files to plot")
    
    for detector_file in detector_files:
        
        # Parse filename to extract parameters
        filename = os.path.basename(detector_file)
        parts = filename.split('_')
        detector_file = detector_file.replace('\\', '/')
        # Remove "_0" from the detector filename if it exists
        if detector_file.endswith('_0.json'):
            detector_file = detector_file[:-7] + '.json'
        # Extract information from filename
        if 'fasttext' in filename:
            model = 'fasttext'
        elif 'roberta' in filename:
            model = 'roberta-base'
        elif 'distilbert' in filename:
            model = 'distilbert-base-cased'
        elif 'bert' in filename:
            model = 'bert-base-cased'
        else:
            print(f"Unknown model type in {filename}, skipping")
            continue
            
        # Extract dimension
        for part in parts:
            if 'dim' in part:
                dim = part.replace('dim', '')
                break
        else:
            print(f"Could not determine dimension from {filename}, skipping")
            continue
            
        # Determine algorithm
        if 'nsgaii' in filename:
            algorithm = 'nsgaii'
        else:
            algorithm = 'ga'
            
        # Construct dataset path
        dimensions = parts[2]  # e.g., "2dim"
        suffix = '_'.join(parts[3:-2])  # e.g., "15_25700_21417"
        dataset_path = f"dataset/ISOT/True_Fake_{model}_umap_{dimensions}_15_25700_21417.h5"
        
        # Construct command
        command = [
            "python",
            f"./{algorithm}_nsa.py",
            f"--dim={dim}",
            f"--dataset={dataset_path}",
            f"--detectorset={detector_file}",
            "--amount=0",  # Don't create new detectors
            "--convergence_every=25",
            "--self_region=-1",
            "--coverage=0.0005",
            "--sample=2500",
            "--experiment=0",
            f"--model={model}"
        ]
        # Execute command
        print(f"Executing: {' '.join(command)}")
        try:
            # Use sys.executable to ensure the same Python interpreter is used
            subprocess.run([sys.executable] + command[1:], check=True)
            time.sleep(10)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for {detector_file}: {e}")
        except FileNotFoundError as e:
            print(f"File not found error: {e}")
            print(f"File not found error: {e}")
            
    print("Finished plotting all detector models")

# algorithm, dataset, dim, experiment no.
'''{
    "test_precision": 0.9263899765074393,
    "test_recall": 0.5522875816993464,
    "test_f1": 0.6920152091254753,
    "test_true_detected": 188,
    "test_true_total": 4284,
    "test_fake_detected": 2366,
    "test_fake_total": 4284,
    "test_negative_space_coverage": 375225.6265239304,
    "test_time_to_build": 19655.375818600005,
    "test_detectors_count": 3000,
    "test_time_to_infer": 71.42128960002447,
    "validation_precision_list": [

    ],
    "validation_recall_list": [

    ],
    "validation_true_detected_list": [

    ],
    "validation_fake_detected_list": [

    ],
    "validation_negative_space_coverage_list": [
        
    ],
    "self_region": 0.07325451668655299,
    "stagnation": 0
}'''
results = {}

file_identification = 'experiment_result'
path = 'model/detector'

files = [f for f in os.listdir(path) if (file_identification in f and '-1' not in f)]
for f in files:
    if 'nsgaii' in f:
        algo = 'nsgaii'
    else:
        algo = 'ga'
    f_split = f.split('_')

    with open(os.path.join(path, f), 'r') as file:
        data = json.load(file)
        dataset = f_split[1]
        dim = f'{f_split[2][0]}D'
        experiment_no = f_split[3].split('.')[0]

        if algo not in results:
            results[algo] = {}
        if dataset not in results[algo]:
            results[algo][dataset] = {}
        if dim not in results[algo][dataset]:
            results[algo][dataset][dim] = []

        results[algo][dataset][dim].append({
            "precision": data["test_precision"],
            "recall": data["test_recall"],
            "true_detected": data["test_true_detected"],
            "true_total": data["test_true_total"],
            "fake_detected": data["test_fake_detected"],
            "fake_total": data["test_fake_total"],
            "negative_space_coverage": data["test_negative_space_coverage"],
            "time_to_build": data["test_time_to_build"],
            "detectors_count": data["test_detectors_count"],
            "time_to_infer": data["test_time_to_infer"],
            "self_region": data["self_region"],
            "stagnation": data["stagnation"]
        })
        # TODO: remove this when I have more than 1 experiment!
        results[algo][dataset][dim].append({
            "precision": data["test_precision"],
            "recall": data["test_recall"],
            "true_detected": data["test_true_detected"],
            "true_total": data["test_true_total"],
            "fake_detected": data["test_fake_detected"],
            "fake_total": data["test_fake_total"],
            "negative_space_coverage": data["test_negative_space_coverage"],
            "time_to_build": data["test_time_to_build"],
            "detectors_count": data["test_detectors_count"],
            "time_to_infer": data["test_time_to_infer"],
            "self_region": data["self_region"],
            "stagnation": data["stagnation"]
        })
print(results)   

# Prepare CSV file
csv_file = 'report/results/averaged_results.csv'
csv_columns = [
    'algorithm', 'dataset', 'dimension', 'precision_avg', 'precision_stdev', 'recall_avg', 'recall_stdev',
    'accuracy_avg', 'accuracy_stdev', 'f1_avg', 'f1_stdev', 'true_detected_avg', 'true_detected_stdev', 'true_total_avg', 'true_total_stdev',
    'fake_detected_avg', 'fake_detected_stdev', 'fake_total_avg', 'fake_total_stdev',
    'negative_space_coverage_avg', 'negative_space_coverage_stdev', 'time_to_build_avg', 'time_to_build_stdev',
    'detectors_count_avg', 'detectors_count_stdev', 'time_to_infer_avg', 'time_to_infer_stdev', 'self_region_avg',
    'self_region_stdev', 'stagnation'
]

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

    for algo, datasets in results.items():
        for dataset, dims in datasets.items():
            for dim, experiments in dims.items():
                precision_list = [exp["precision"] for exp in experiments]
                recall_list = [exp["recall"] for exp in experiments]
                true_detected_list = [exp["true_detected"] for exp in experiments]
                true_total_list = [exp["true_total"] for exp in experiments]
                fake_detected_list = [exp["fake_detected"] for exp in experiments]
                fake_total_list = [exp["fake_total"] for exp in experiments]
                negative_space_coverage_list = [exp["negative_space_coverage"] for exp in experiments]
                time_to_build_list = [exp["time_to_build"] for exp in experiments]
                detectors_count_list = [exp["detectors_count"] for exp in experiments]
                time_to_infer_list = [exp["time_to_infer"] for exp in experiments]
                self_region_list = [exp["self_region"] for exp in experiments]

                true_positive_list = [exp["fake_detected"] for exp in experiments]
                false_negative_list = [exp["fake_total"] - exp["fake_detected"] for exp in experiments]
                false_positive_list = [exp["true_detected"] for exp in experiments]
                true_negative_list = [exp["true_total"] - exp["true_detected"] for exp in experiments]

                accuracy_list = [(tp + tn) / (tp + tn + fp + fn) for tp, tn, fp, fn in zip(true_positive_list, true_negative_list, false_positive_list, false_negative_list)]
                f1_list = [2 * (p * r) / (p + r) for p, r in zip(precision_list, recall_list)]

                writer.writerow({
                    'algorithm': algo,
                    'dataset': dataset,
                    'dimension': dim,
                    'precision_avg': statistics.mean(precision_list),
                    'precision_stdev': statistics.stdev(precision_list),
                    'recall_avg': statistics.mean(recall_list),
                    'recall_stdev': statistics.stdev(recall_list),
                    'accuracy_avg': statistics.mean(accuracy_list),
                    'accuracy_stdev': statistics.stdev(accuracy_list),
                    'f1_avg': statistics.mean(f1_list),
                    'f1_stdev': statistics.stdev(f1_list),
                    'true_detected_avg': statistics.mean(true_detected_list),
                    'true_total_avg': statistics.mean(true_total_list),
                    'fake_detected_avg': statistics.mean(fake_detected_list),
                    'fake_total_avg': statistics.mean(fake_total_list),
                    'negative_space_coverage_avg': statistics.mean(negative_space_coverage_list),
                    'time_to_build_avg': statistics.mean(time_to_build_list),
                    'time_to_build_stdev': statistics.stdev(time_to_build_list),
                    'detectors_count_avg': statistics.mean(detectors_count_list),
                    'detectors_count_stdev': statistics.stdev(detectors_count_list),
                    'time_to_infer_avg': statistics.mean(time_to_infer_list),
                    'time_to_infer_stdev': statistics.stdev(time_to_infer_list),
                    'self_region_avg': statistics.mean(self_region_list),
                    'self_region_stdev': statistics.stdev(self_region_list),
                    'stagnation': statistics.mean([exp["stagnation"] for exp in experiments])
                })

                

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Drop all columns that contain 'stdev'
df = df.loc[:, ~df.columns.str.contains('stdev')]

# Save the modified DataFrame to a new CSV file
excel_file = 'report/results/averaged_results_nostdev.xlsx'

# Round all numbers to max 3 decimals
df = df.round(3)
df.to_excel(excel_file, index=False)

# Display the DataFrame as a table
print(df.to_string(index=False))

# Create directories if they don't exist
os.makedirs('results', exist_ok=True)

# Create the tables
'''best_overall = create_best_overall_table()
per_embedding = create_per_embedding_tables()
full_table = create_full_table_sorted_by_f1()
create_negative_space_coverage_plot()
create_boxplot_by_dimension('f1_avg', 'F1-score')
create_boxplot_by_dimension('precision_avg', 'Precision')
create_boxplot_by_dimension('recall_avg', 'Recall')
create_boxplot_by_dimension('detectors_count_avg', 'Detectors')
create_boxplot_by_embedding('f1_avg', 'F1-score')
create_boxplot_by_embedding('precision_avg', 'Precision')
create_boxplot_by_embedding('recall_avg', 'Recall')
create_boxplot_by_embedding('detectors_count_avg', 'Detectors')
create_boxplot_by_algorithm('f1_avg', 'F1-score')
create_boxplot_by_algorithm('precision_avg', 'Precision')
create_boxplot_by_algorithm('recall_avg', 'Recall')
create_boxplot_by_algorithm('detectors_count_avg', 'Detectors')
plot_f1_precision_recall_negative_space_per_detector_amount()'''
plot_detector_models()