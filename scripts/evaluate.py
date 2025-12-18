#!/usr/bin/env python3
"""Evaluate trained models and generate visualizations."""

import argparse
import os
import sys
import pickle
import numpy as np
from tensorflow import keras

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataPreprocessor
from src.evaluation import (
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    plot_multiple_confusion_matrices,
    plot_comparison,
    plot_multiple_histories,
    generate_classification_report_table,
    print_classification_report_table,
    generate_all_datasets_report
)
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model (overrides config)')
    parser.add_argument('--history-path', type=str, default=None,
                       help='Path to training history (overrides config)')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data file (overrides config)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Determine paths
    model_name = config['model']['name']
    
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(config['output']['model_dir'], f'best_{model_name}_model.h5')
    
    if args.history_path:
        history_path = args.history_path
    else:
        history_path = os.path.join(config['output']['history_dir'], f'best_{model_name}_history.pkl')
    
    if args.data_path:
        data_path = args.data_path
    else:
        if config['data']['pca']['enabled']:
            data_path = os.path.join(config['data']['output_dir'], 'pca_data.npy')
        else:
            data_path = os.path.join(config['data']['output_dir'], 'reshaped_data.npy')
    
    labels_path = os.path.join(config['data']['output_dir'], 'reshaped_labels.npy')
    
    # Create plots directory
    plots_dir = config['output']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {data_path}")
    data = np.load(data_path)
    labels = np.load(labels_path)
    
    # Split data
    preprocessor = DataPreprocessor(seed=config['training']['seed'])
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        data, labels,
        test_size=config['data']['split']['test_size'],
        val_size=config['data']['split']['val_size']
    )
    
    print(f"Test set shape: {X_test.shape}")
    
    # Load model
    print(f"\nLoading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    # Evaluate model
    print(f"\n{'='*60}")
    print("Evaluating model on all datasets")
    print(f"{'='*60}")

    results = evaluate_model(
        model, X_test, y_test,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        verbose=1
    )

    # Print results for all datasets
    for dataset_name in ['train', 'val', 'test']:
        if dataset_name in results:
            print(f"\n{dataset_name.capitalize()} Set:")
            print(f"  Loss: {results[dataset_name]['loss']:.4f}")
            print(f"  Accuracy: {results[dataset_name]['accuracy']:.4f}")

    # Plot confusion matrices for all datasets
    print("\nGenerating confusion matrices...")
    confusion_matrices = {}
    for dataset_name in ['train', 'val', 'test']:
        if dataset_name in results:
            confusion_matrices[dataset_name] = results[dataset_name]['confusion_matrix']

    if confusion_matrices:
        plot_multiple_confusion_matrices(
            confusion_matrices,
            save_path=os.path.join(plots_dir, f'{model_name}_confusion_matrices.png')
        )
    
    # Load and plot training history
    if os.path.exists(history_path):
        print(f"\nLoading training history from {history_path}")
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        print("Generating training history plots...")
        plot_training_history(
            history,
            save_path=os.path.join(plots_dir, f'{model_name}_training_history.png')
        )
    else:
        print(f"\nWarning: History file not found at {history_path}")
    
    # Compare multiple models if requested
    if args.compare:
        print(f"\n{'='*60}")
        print("Comparing multiple models")
        print(f"{'='*60}")

        # List of common model names to check (ordered: gnn, lstm, gnn_lstm first)
        model_names = ['gnn', 'lstm', 'gnn_lstm', 'gnn_rnn', 'rnn', 'attention_gnn_rnn', 'attention_gnn_lstm', 'bilstm_gnn']
        models_results = {}
        models_histories = {}
        models_full_results = {}  # Store full results including classification_report

        for name in model_names:
            model_file = os.path.join(config['output']['model_dir'], f'best_{name}_model.h5')
            history_file = os.path.join(config['output']['history_dir'], f'best_{name}_history.pkl')

            if os.path.exists(model_file):
                print(f"\nEvaluating {name}...")
                m = keras.models.load_model(model_file)

                # Evaluate on all datasets
                r = evaluate_model(
                    m, X_test, y_test,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    verbose=0
                )

                models_results[name] = {
                    'train_loss': r['train']['loss'] if 'train' in r else None,
                    'train_accuracy': r['train']['accuracy'] if 'train' in r else None,
                    'val_loss': r['val']['loss'] if 'val' in r else None,
                    'val_accuracy': r['val']['accuracy'] if 'val' in r else None,
                    'test_loss': r['test_loss'],
                    'test_accuracy': r['test_accuracy']
                }
                
                # Store full results for classification report
                models_full_results[name] = r

                # Load training history if available
                if os.path.exists(history_file):
                    with open(history_file, 'rb') as f:
                        history = pickle.load(f)
                        models_histories[name] = history

                    # Plot individual training history
                    print(f"Generating training history for {name}...")
                    plot_training_history(
                        history,
                        save_path=os.path.join(plots_dir, f'{name}_training_history.png')
                    )

                    # Generate confusion matrices for this model
                    print(f"Generating confusion matrices for {name}...")
                    confusion_matrices = {}
                    for dataset_name in ['train', 'val', 'test']:
                        if dataset_name in r:
                            confusion_matrices[dataset_name] = r[dataset_name]['confusion_matrix']

                    if confusion_matrices:
                        plot_multiple_confusion_matrices(
                            confusion_matrices,
                            save_path=os.path.join(plots_dir, f'{name}_confusion_matrices.png')
                        )

        if models_results:
            print("\n" + "="*60)
            print("Generating comparison plots...")
            print("="*60)

            # Compare test accuracy
            plot_comparison(
                models_results,
                metric='accuracy',
                save_path=os.path.join(plots_dir, 'models_comparison_test_accuracy.png')
            )

            # Compare test loss
            plot_comparison(
                models_results,
                metric='loss',
                save_path=os.path.join(plots_dir, 'models_comparison_test_loss.png')
            )

            # Compare training histories
            if models_histories:
                print("\nGenerating training history comparison plots...")

                # Compare validation accuracy across models
                plot_multiple_histories(
                    models_histories,
                    metric='val_accuracy',
                    save_path=os.path.join(plots_dir, 'models_comparison_val_accuracy_history.png')
                )

                # Compare validation loss across models
                plot_multiple_histories(
                    models_histories,
                    metric='val_loss',
                    save_path=os.path.join(plots_dir, 'models_comparison_val_loss_history.png')
                )

            # Print summary table
            print("\n" + "="*60)
            print("Models Comparison Summary")
            print("="*60)
            print(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
            print("-"*60)
            for name, results in models_results.items():
                train_acc = f"{results['train_accuracy']:.4f}" if results['train_accuracy'] is not None else "N/A"
                val_acc = f"{results['val_accuracy']:.4f}" if results['val_accuracy'] is not None else "N/A"
                test_acc = f"{results['test_accuracy']:.4f}"
                print(f"{name:<20} {train_acc:<12} {val_acc:<12} {test_acc:<12}")
            
            # Generate detailed classification report tables for all datasets
            print("\n" + "="*60)
            print("Generating Classification Report Tables...")
            print("="*60)
            generate_all_datasets_report(models_full_results, save_dir=plots_dir)

        else:
            print("No models found for comparison")
    
    print(f"\n{'='*60}")
    print("Evaluation completed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

