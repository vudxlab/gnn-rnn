"""Model training with K-Fold cross validation."""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List, Optional, Tuple
from .callbacks import get_callbacks


class ModelTrainer:
    """Train models with K-Fold cross validation."""
    
    def __init__(
        self,
        model_builder,
        n_splits: int = 5,
        epochs: int = 1000,
        batch_size: int = 32,
        seed: int = 42,
        device: str = '/GPU:0',
        kfold_type: str = 'stratified'
    ):
        """
        Initialize ModelTrainer.

        Args:
            model_builder: Model builder instance (from models module)
            n_splits: Number of folds for cross validation
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            seed: Random seed
            device: Device to use for training ('/GPU:0' or '/CPU:0')
            kfold_type: Type of KFold ('kfold' or 'stratified')
                - 'stratified': StratifiedKFold - ensures balanced class distribution (recommended)
                - 'kfold': Standard KFold - random split without considering class balance
        """
        self.model_builder = model_builder
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.kfold_type = kfold_type.lower()

        # Set random seeds
        keras.utils.set_random_seed(seed)

        # Initialize KFold or StratifiedKFold (only if n_splits > 1)
        if n_splits > 1:
            if self.kfold_type == 'stratified':
                # StratifiedKFold ensures each fold has balanced class distribution
                self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                print(f"Using StratifiedKFold with {n_splits} splits (balanced class distribution)")
            elif self.kfold_type == 'kfold':
                # Standard KFold for random split
                self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                print(f"Using KFold with {n_splits} splits (random split)")
            else:
                raise ValueError(f"Invalid kfold_type: {kfold_type}. Must be 'kfold' or 'stratified'")
        else:
            self.kfold = None

        # Storage for results
        self.best_model = None
        self.best_history = None
        self.best_val_accuracy = 0
        self.fold_results = []
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        callbacks_config: Optional[Dict] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train model with K-Fold cross validation.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_test: Optional test data for evaluation
            y_test: Optional test labels
            callbacks_config: Optional config for callbacks
            verbose: Verbosity level (0, 1, or 2)
        
        Returns:
            Dictionary containing training results
        """
        if callbacks_config is None:
            callbacks_config = {}
        
        fold_accuracies = []
        fold_test_accuracies = []

        with tf.device(self.device):
            # Handle n_splits=1 case (no cross-validation)
            if self.kfold is None:
                # Use 80/20 split for train/val when n_splits=1
                print(f'\nTraining without cross-validation (n_splits=1)')
                n_train = int(0.8 * len(X_train))
                fold_splits = [(np.arange(n_train), np.arange(n_train, len(X_train)))]
                self.n_splits = 1
            else:
                # KFold or StratifiedKFold
                if self.kfold_type == 'stratified':
                    # StratifiedKFold requires labels to ensure balanced folds
                    fold_splits = list(enumerate(self.kfold.split(X_train, y_train)))
                else:
                    # Standard KFold doesn't need labels
                    fold_splits = list(enumerate(self.kfold.split(X_train)))

            for fold_info in fold_splits:
                if self.kfold is None:
                    fold = 0
                    train_idx, val_idx = fold_info
                else:
                    fold, (train_idx, val_idx) = fold_info

                print(f'\nTraining fold {fold + 1}/{self.n_splits}')
                
                # Split data for this fold
                X_train_fold = X_train[train_idx]
                y_train_fold = y_train[train_idx]
                X_val_fold = X_train[val_idx]
                y_val_fold = y_train[val_idx]
                
                # Build and compile model
                self.model_builder.compile()
                model = self.model_builder.get_model()
                
                # Train model
                history = model.fit(
                    X_train_fold, y_train_fold,
                    epochs=self.epochs,
                    callbacks=get_callbacks(**callbacks_config),
                    batch_size=self.batch_size,
                    verbose=verbose,
                    shuffle=True,  # Enable shuffling to avoid batch ordering issues
                    validation_data=(X_val_fold, y_val_fold)
                )
                
                # Evaluate on validation set
                val_accuracy = max(history.history['val_accuracy'][-1:])
                fold_accuracies.append(val_accuracy)
                
                # Evaluate on test set if provided
                test_accuracy = None
                if X_test is not None and y_test is not None:
                    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                    fold_test_accuracies.append(test_accuracy)
                
                # Store fold results
                fold_result = {
                    'fold': fold + 1,
                    'train_accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy'],
                    'train_loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'test_accuracy': test_accuracy
                }
                self.fold_results.append(fold_result)
                
                # Update best model if current fold is better
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_model = model
                    self.best_history = history
                    self.best_fold_idx = fold
                    print(f'New best model found with validation accuracy: {self.best_val_accuracy:.4f}')
        
        # Prepare results summary
        results = {
            'avg_val_accuracy': np.mean(fold_accuracies),
            'std_val_accuracy': np.std(fold_accuracies),
            'best_val_accuracy': self.best_val_accuracy,
            'best_fold': self.best_fold_idx + 1,
            'fold_accuracies': fold_accuracies,
            'fold_results': self.fold_results
        }
        
        if fold_test_accuracies:
            results['avg_test_accuracy'] = np.mean(fold_test_accuracies)
            results['std_test_accuracy'] = np.std(fold_test_accuracies)
            results['fold_test_accuracies'] = fold_test_accuracies
        
        return results
    
    def save_best_model(self, filepath: str):
        """Save the best model to file."""
        if self.best_model is not None:
            self.best_model.save(filepath)
            print(f'Best model saved to {filepath}')
        else:
            print('No model to save. Train first.')
    
    def save_best_history(self, filepath: str):
        """Save the best training history to pickle file."""
        if self.best_history is not None:
            history_dict = {
                'train_accuracy': self.best_history.history['accuracy'],
                'val_accuracy': self.best_history.history['val_accuracy'],
                'train_loss': self.best_history.history['loss'],
                'val_loss': self.best_history.history['val_loss']
            }
            with open(filepath, 'wb') as f:
                pickle.dump(history_dict, f)
            print(f'Best history saved to {filepath}')
        else:
            print('No history to save. Train first.')
    
    def save_results(self, filepath: str, results: Dict):
        """Save training results to text file."""
        with open(filepath, 'w') as f:
            f.write(f"Average validation accuracy across {self.n_splits} folds: "
                   f"{results['avg_val_accuracy']:.4f} ± {results['std_val_accuracy']:.4f}\n")
            
            if 'avg_test_accuracy' in results:
                f.write(f"Average test accuracy across {self.n_splits} folds: "
                       f"{results['avg_test_accuracy']:.4f} ± {results['std_test_accuracy']:.4f}\n")
            
            f.write(f"Best fold: {results['best_fold']} with validation accuracy: "
                   f"{results['best_val_accuracy']:.4f}\n")
            
            f.write('\nDetailed results for each fold:\n')
            for i, (val_acc, fold_result) in enumerate(zip(results['fold_accuracies'], 
                                                            results['fold_results'])):
                test_acc = fold_result['test_accuracy']
                if test_acc is not None:
                    f.write(f"Fold {i+1}: Validation accuracy = {val_acc:.4f}, "
                           f"Test accuracy = {test_acc:.4f}\n")
                else:
                    f.write(f"Fold {i+1}: Validation accuracy = {val_acc:.4f}\n")
        
        print(f'Results saved to {filepath}')

