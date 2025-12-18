"""Evaluation metrics and utilities."""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Tuple
from tensorflow import keras


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    verbose: int = 1
) -> Dict:
    """
    Evaluate model on test data and optionally on train/val data.

    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels
        X_train: Optional train data
        y_train: Optional train labels
        X_val: Optional validation data
        y_val: Optional validation labels
        verbose: Verbosity level

    Returns:
        Dictionary containing evaluation metrics for all provided datasets
    """
    results = {}

    # Evaluate on test set
    y_pred = model.predict(X_test, verbose=verbose)
    y_pred_classes = np.argmax(y_pred, axis=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=verbose)
    cm_test = confusion_matrix(y_test, y_pred_classes)
    report_test = classification_report(y_test, y_pred_classes, output_dict=True)

    results['test'] = {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'confusion_matrix': cm_test,
        'classification_report': report_test,
        'predictions': y_pred_classes
    }

    # Backward compatibility
    results['test_loss'] = test_loss
    results['test_accuracy'] = test_accuracy
    results['confusion_matrix'] = cm_test
    results['classification_report'] = report_test
    results['predictions'] = y_pred_classes

    # Evaluate on train set if provided
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_train_classes = np.argmax(y_pred_train, axis=1)
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        cm_train = confusion_matrix(y_train, y_pred_train_classes)
        report_train = classification_report(y_train, y_pred_train_classes, output_dict=True)

        results['train'] = {
            'loss': train_loss,
            'accuracy': train_accuracy,
            'confusion_matrix': cm_train,
            'classification_report': report_train,
            'predictions': y_pred_train_classes
        }

    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val, verbose=0)
        y_pred_val_classes = np.argmax(y_pred_val, axis=1)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        cm_val = confusion_matrix(y_val, y_pred_val_classes)
        report_val = classification_report(y_val, y_pred_val_classes, output_dict=True)

        results['val'] = {
            'loss': val_loss,
            'accuracy': val_accuracy,
            'confusion_matrix': cm_val,
            'classification_report': report_val,
            'predictions': y_pred_val_classes
        }

    return results


def compare_models(
    models_dict: Dict[str, keras.Model],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Compare multiple models on test data.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        X_test: Test data
        y_test: Test labels
    
    Returns:
        Dictionary containing comparison results
    """
    comparison = {}
    
    for model_name, model in models_dict.items():
        results = evaluate_model(model, X_test, y_test, verbose=0)
        comparison[model_name] = {
            'test_loss': results['test_loss'],
            'test_accuracy': results['test_accuracy']
        }
    
    return comparison

