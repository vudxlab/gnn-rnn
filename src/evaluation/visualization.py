"""Visualization utilities for training and evaluation."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from typing import Dict, List, Optional, Tuple


def plot_training_history(
    history_dict: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history_dict: Dictionary containing 'train_accuracy', 'val_accuracy', 
                     'train_loss', 'val_loss'
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    ax1.plot(history_dict['train_accuracy'], label='Train', linewidth=2)
    ax1.plot(history_dict['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot loss
    ax2.plot(history_dict['train_loss'], label='Train', linewidth=2)
    ax2.plot(history_dict['val_loss'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
    
    plt.show()


def plot_comparison(
    models_results: Dict[str, Dict],
    metric: str = 'accuracy',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot comparison of multiple models.
    
    Args:
        models_results: Dictionary of {model_name: {'test_loss': x, 'test_accuracy': y}}
        metric: Metric to plot ('accuracy' or 'loss')
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    models = list(models_results.keys())
    
    if metric == 'accuracy':
        values = [models_results[m]['test_accuracy'] for m in models]
        ylabel = 'Accuracy'
        title = 'Model Comparison - Test Accuracy'
    else:
        values = [models_results[m]['test_loss'] for m in models]
        ylabel = 'Loss'
        title = 'Model Comparison - Test Loss'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(models, values, color='steelblue', edgecolor='black', linewidth=0.5, width=0.2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{value:.4f}',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Optional list of class names
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax)

    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')

    plt.show()


def plot_multiple_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 5)
):
    """
    Plot multiple confusion matrices side by side.

    Args:
        confusion_matrices: Dictionary of {dataset_name: confusion_matrix}
        class_names: Optional list of class names
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    n_matrices = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_matrices, figsize=figsize)

    if n_matrices == 1:
        axes = [axes]

    for ax, (dataset_name, cm) in zip(axes, confusion_matrices.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
        ax.set_title(f'{dataset_name.capitalize()} Set', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')

    plt.show()


def plot_multiple_histories(
    histories_dict: Dict[str, Dict],
    metric: str = 'val_accuracy',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    max_epochs: Optional[int] = None
):
    """
    Plot training curves for multiple models.
    
    Args:
        histories_dict: Dictionary of {model_name: history_dict}
        metric: Metric to plot ('val_accuracy', 'train_accuracy', 'val_loss', 'train_loss')
        save_path: Optional path to save the figure
        figsize: Figure size
        max_epochs: Optional maximum number of epochs to plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, history in histories_dict.items():
        if metric in history:
            values = history[metric]
            if max_epochs:
                values = values[:max_epochs]
            ax.plot(values, label=model_name, linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'Comparison of {metric.replace("_", " ").title()}', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if max_epochs:
        ax.set_xlim(0, max_epochs)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
    
    plt.show()


def generate_classification_report_table(
    models_results: Dict[str, Dict],
    dataset: str = 'test',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a detailed classification report table comparing multiple models.
    
    Args:
        models_results: Dictionary of {model_name: evaluation_results}
                       where evaluation_results contains 'classification_report' for each dataset
        dataset: Which dataset to use ('train', 'val', 'test')
        save_path: Optional path to save the table as CSV
    
    Returns:
        DataFrame with precision, recall, f1-score for each label and model
    """
    # Get all unique labels from the first model's report
    first_model = list(models_results.keys())[0]
    first_report = models_results[first_model].get(dataset, {}).get('classification_report', {})
    
    # Get numeric labels (exclude 'accuracy', 'macro avg', 'weighted avg')
    labels = [k for k in first_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    labels = sorted(labels, key=lambda x: int(x) if x.isdigit() else x)
    
    # Build table data
    rows = []
    for label in labels:
        row = {'Label': label}
        for model_name in models_results.keys():
            report = models_results[model_name].get(dataset, {}).get('classification_report', {})
            if label in report:
                row[f'{model_name}_precision'] = round(report[label]['precision'], 3)
                row[f'{model_name}_recall'] = round(report[label]['recall'], 3)
                row[f'{model_name}_f1-score'] = round(report[label]['f1-score'], 3)
            else:
                row[f'{model_name}_precision'] = None
                row[f'{model_name}_recall'] = None
                row[f'{model_name}_f1-score'] = None
        rows.append(row)
    
    # Add accuracy row
    accuracy_row = {'Label': 'accuracy'}
    for model_name in models_results.keys():
        report = models_results[model_name].get(dataset, {}).get('classification_report', {})
        acc = report.get('accuracy', None)
        accuracy_row[f'{model_name}_precision'] = ''
        accuracy_row[f'{model_name}_recall'] = ''
        accuracy_row[f'{model_name}_f1-score'] = round(acc, 3) if acc else None
    rows.append(accuracy_row)
    
    df = pd.DataFrame(rows)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f'Table saved to {save_path}')
    
    return df


def print_classification_report_table(
    models_results: Dict[str, Dict],
    dataset: str = 'test'
):
    """
    Print a formatted classification report table comparing multiple models.
    
    Args:
        models_results: Dictionary of {model_name: evaluation_results}
        dataset: Which dataset to use ('train', 'val', 'test')
    """
    model_names = list(models_results.keys())
    
    # Get all unique labels
    first_model = model_names[0]
    first_report = models_results[first_model].get(dataset, {}).get('classification_report', {})
    labels = [k for k in first_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    labels = sorted(labels, key=lambda x: int(x) if x.isdigit() else x)
    
    # Print header
    print(f"\n{'='*120}")
    print(f"Classification Report - {dataset.upper()} Set")
    print(f"{'='*120}")
    
    # Build header row
    header = f"{'Label':<8}"
    for model_name in model_names:
        header += f"| {model_name:^30} "
    print(header)
    
    # Sub-header for metrics
    sub_header = f"{'':<8}"
    for _ in model_names:
        sub_header += f"| {'precision':^9} {'recall':^9} {'f1-score':^9} "
    print(sub_header)
    print("-" * 120)
    
    # Print each label row
    for label in labels:
        row = f"{label:<8}"
        for model_name in model_names:
            report = models_results[model_name].get(dataset, {}).get('classification_report', {})
            if label in report:
                p = report[label]['precision']
                r = report[label]['recall']
                f1 = report[label]['f1-score']
                row += f"| {p:^9.3f} {r:^9.3f} {f1:^9.3f} "
            else:
                row += f"| {'N/A':^9} {'N/A':^9} {'N/A':^9} "
        print(row)
    
    # Print accuracy row
    print("-" * 120)
    acc_row = f"{'accuracy':<8}"
    for model_name in model_names:
        report = models_results[model_name].get(dataset, {}).get('classification_report', {})
        acc = report.get('accuracy', None)
        if acc:
            acc_row += f"| {'':^9} {'':^9} {acc:^9.3f} "
        else:
            acc_row += f"| {'':^9} {'':^9} {'N/A':^9} "
    print(acc_row)
    print(f"{'='*120}\n")


def generate_all_datasets_report(
    models_results: Dict[str, Dict],
    save_dir: Optional[str] = None
):
    """
    Generate and print classification report tables for all datasets (train, val, test).
    
    Args:
        models_results: Dictionary of {model_name: evaluation_results}
        save_dir: Optional directory to save CSV files
    """
    for dataset in ['train', 'val', 'test']:
        # Check if this dataset exists in results
        has_data = any(
            dataset in results 
            for results in models_results.values()
        )
        
        if has_data:
            print_classification_report_table(models_results, dataset)
            
            if save_dir:
                import os
                save_path = os.path.join(save_dir, f'classification_report_{dataset}.csv')
                generate_classification_report_table(models_results, dataset, save_path)

