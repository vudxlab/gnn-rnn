"""Evaluation metrics and visualization utilities."""

from .metrics import evaluate_model
from .visualization import (
    plot_training_history,
    plot_comparison,
    plot_confusion_matrix,
    plot_multiple_confusion_matrices,
    plot_multiple_histories,
    generate_classification_report_table,
    print_classification_report_table,
    generate_all_datasets_report
)

__all__ = [
    'evaluate_model',
    'plot_training_history',
    'plot_comparison',
    'plot_confusion_matrix',
    'plot_multiple_confusion_matrices',
    'plot_multiple_histories',
    'generate_classification_report_table',
    'print_classification_report_table',
    'generate_all_datasets_report'
]

