#!/usr/bin/env python3
"""Train models with K-Fold cross validation."""

import argparse
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import (
    RNNModel, LSTMModel,
    GNNModel,
    GNNRNNModel, AttentionGNNRNNModel,
    GNNLSTMModel, AttentionGNNLSTMModel, BiLSTMGNNModel
)
from src.data import DataPreprocessor
from src.training import ModelTrainer
from src.utils import load_config


def print_comparison_table(all_results):
    """Print comparison table of all trained models."""
    print(f"\n{'='*80}")
    print("Model Comparison Summary")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Val Accuracy':<20} {'Test Accuracy':<20} {'Best Fold':<10}")
    print(f"{'-'*80}")

    best_model = None
    best_accuracy = 0.0

    for model_name, results in all_results.items():
        val_acc = f"{results['avg_val_accuracy']:.4f} ± {results['std_val_accuracy']:.4f}"
        test_acc = f"{results['avg_test_accuracy']:.4f} ± {results['std_test_accuracy']:.4f}" if 'avg_test_accuracy' in results else "N/A"
        best_fold = results['best_fold']

        print(f"{model_name:<25} {val_acc:<20} {test_acc:<20} {best_fold:<10}")

        if results['avg_val_accuracy'] > best_accuracy:
            best_accuracy = results['avg_val_accuracy']
            best_model = model_name

    print(f"{'='*80}")
    print(f"🏆 Best Model: {best_model} with validation accuracy: {best_accuracy:.4f}")
    print(f"{'='*80}\n")


def get_model_builder(config, input_shape, num_classes):
    """Get model builder based on config."""
    model_name = config['model']['name'].lower()
    params = config['model']['params']

    if model_name == 'rnn':
        return RNNModel(
            input_shape, num_classes,
            units=params.get('rnn_units', 256),
            dropout=params.get('rnn_dropout', 0.6)
        )
    elif model_name == 'lstm':
        return LSTMModel(
            input_shape, num_classes,
            units=params.get('lstm_units', 256),
            dropout=params.get('lstm_dropout', 0.3)
        )
    elif model_name == 'gnn':
        return GNNModel(
            input_shape, num_classes,
            gnn_units=params.get('gnn_units', [128, 64]),
            dropout=params.get('gnn_dropout', 0.5),
            use_global_pool=params.get('gnn_pool', 'max')
        )
    elif model_name == 'gnn_rnn':
        return GNNRNNModel(
            input_shape, num_classes,
            gnn_units=params.get('gnn_units', [128, 64]),
            rnn_units=params.get('gnn_rnn_units', 256),
            dropout_gnn=params.get('dropout_gnn', 0.3),
            dropout_rnn=params.get('dropout_rnn', 0.6)
        )
    elif model_name == 'gnn_lstm':
        return GNNLSTMModel(
            input_shape, num_classes,
            gnn_units=params.get('gnn_units', [512, 256]),
            lstm_units=params.get('gnn_lstm_units', 512 ),
            dropout_gnn=params.get('dropout_gnn', 0.2),
            dropout_lstm=params.get('dropout_gnn_lstm', 0.2)
        )
    elif model_name == 'attention_gnn_rnn':
        return AttentionGNNRNNModel(
            input_shape, num_classes,
            gnn_units=params.get('gnn_units', [128, 64]),
            rnn_units=params.get('gnn_rnn_units', 256),
            dropout_gnn=params.get('dropout_gnn', 0.3),
            dropout_rnn=params.get('dropout_rnn', 0.6)
        )
    elif model_name == 'attention_gnn_lstm':
        return AttentionGNNLSTMModel(
            input_shape, num_classes,
            gnn_units=params.get('gnn_units', [128, 64]),
            lstm_units=params.get('gnn_lstm_units', 200),
            dropout_gnn=params.get('dropout_gnn', 0.3),
            dropout_lstm=params.get('dropout_gnn_lstm', 0.5)
        )
    elif model_name == 'bilstm_gnn':
        return BiLSTMGNNModel(
            input_shape, num_classes,
            gnn_units=params.get('gnn_units', [128, 64]),
            lstm_units=params.get('gnn_lstm_units', 256),
            dropout_gnn=params.get('dropout_gnn', 0.3),
            dropout_lstm=params.get('dropout_gnn_lstm', 0.6)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Train model with K-Fold cross validation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data file (overrides config)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (overrides config, single model)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Multiple model names to train (e.g., --models lstm gnn gnn_rnn)')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                       help='Verbosity level')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Determine which models to train
    models_to_train = []
    if args.models:
        models_to_train = args.models
    elif args.model:
        models_to_train = [args.model]
    else:
        models_to_train = [config['model']['name']]

    print(f"\nModels to train: {', '.join(models_to_train)}")

    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        # Check if PCA is enabled
        if config['data']['pca']['enabled']:
            data_path = os.path.join(config['data']['output_dir'], 'pca_data.npy')
        else:
            data_path = os.path.join(config['data']['output_dir'], 'reshaped_data.npy')

    labels_path = os.path.join(config['data']['output_dir'], 'reshaped_labels.npy')

    # Load data
    print(f"\nLoading data from {data_path}")
    data = np.load(data_path)
    labels = np.load(labels_path)
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")

    # Split data
    print("\nSplitting data into train/val/test sets")
    preprocessor = DataPreprocessor(seed=config['training']['seed'])
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        data, labels,
        test_size=config['data']['split']['test_size'],
        val_size=config['data']['split']['val_size']
    )

    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Get number of classes
    num_classes = len(np.unique(y_train))
    print(f"\nNumber of classes: {num_classes}")
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Create output directories
    for dir_key in ['model_dir', 'history_dir', 'results_dir']:
        os.makedirs(config['output'][dir_key], exist_ok=True)

    # Train each model
    all_results = {}

    for model_name in models_to_train:
        print(f"\n{'='*80}")
        print(f"Training Model: {model_name.upper()}")
        print(f"{'='*80}")

        # Update config with current model
        config['model']['name'] = model_name

        # Build model
        print(f"\nBuilding model: {model_name}")
        model_builder = get_model_builder(config, input_shape, num_classes)
        model_builder.summary()

        # Initialize trainer
        print(f"\n{'='*60}")
        print("Starting training with K-Fold cross validation")
        print(f"{'='*60}")

        # Get kfold_type from config, default to 'stratified' for backward compatibility
        kfold_type = config['training'].get('kfold_type', 'stratified')

        trainer = ModelTrainer(
            model_builder=model_builder,
            n_splits=config['training']['n_splits'],
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            seed=config['training']['seed'],
            device=config['training']['device'],
            kfold_type=kfold_type
        )

        # Train model
        results = trainer.train(
            X_train, y_train,
            X_test, y_test,
            callbacks_config=config['training']['callbacks'],
            verbose=args.verbose
        )

        # Print results
        print(f"\n{'='*60}")
        print(f"Training Results for {model_name}")
        print(f"{'='*60}")
        print(f"Average validation accuracy: {results['avg_val_accuracy']:.4f} ± {results['std_val_accuracy']:.4f}")
        if 'avg_test_accuracy' in results:
            print(f"Average test accuracy: {results['avg_test_accuracy']:.4f} ± {results['std_test_accuracy']:.4f}")
        print(f"Best fold: {results['best_fold']} with validation accuracy: {results['best_val_accuracy']:.4f}")

        # Save results
        trainer.save_best_model(
            os.path.join(config['output']['model_dir'], f'best_{model_name}_model.h5')
        )
        trainer.save_best_history(
            os.path.join(config['output']['history_dir'], f'best_{model_name}_history.pkl')
        )
        trainer.save_results(
            os.path.join(config['output']['results_dir'], f'{model_name}_results.txt'),
            results
        )

        # Store results for comparison
        all_results[model_name] = results

    # Print comparison table if multiple models were trained
    if len(models_to_train) > 1:
        print_comparison_table(all_results)

    print(f"\n{'='*80}")
    print("All training completed successfully!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

