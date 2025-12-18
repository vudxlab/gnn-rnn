# GNN-RNN Framework for Structural Health Monitoring

A modular deep learning framework for time series structural health monitoring using Graph Neural Networks (GNN), Recurrent Neural Networks (RNN), LSTM, and hybrid architectures (GNN-RNN, GNN-LSTM).

## Features

- **Multiple Datasets Support**: Z24 Bridge and CDV103 datasets
- **Data Augmentation**: 5 augmentation techniques (noise, reverse, crop_pad, time_warp, random_shift)
- **Dimensionality Reduction**: PCA for feature reduction
- **Advanced Models**: 8 different model architectures including GNN-based models
- **K-Fold Cross Validation**: Robust model evaluation
- **Modular Design**: Easy to extend and customize
- **CLI Interface**: Simple command-line tools for data preparation, training, and evaluation
- **Configuration Files**: YAML-based configuration for reproducibility

## Project Structure

```
gnn-rnn/
├── configs/                    # Configuration files
│   ├── z24_config.yaml
│   └── cdv103_config.yaml
├── data/                       # Dataset directory
│   ├── cdv103/                 # CDV103 dataset
│   └── z24/                    # Z24 dataset
├── src/                        # Source code
│   ├── data/                   # Data loading and preprocessing
│   │   ├── loader.py
│   │   ├── augmentation.py
│   │   └── preprocessing.py
│   ├── models/                 # Model architectures
│   │   ├── base.py             # Base model class
│   │   ├── rnn.py              # RNN, LSTM
│   │   ├── gnn.py              # GNN
│   │   ├── gnn_rnn.py          # GNN-RNN hybrids
│   │   └── gnn_lstm.py         # GNN-LSTM hybrids
│   ├── training/               # Training utilities
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation/             # Evaluation and visualization
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/                  # Utility functions
│       └── config.py
├── scripts/                    # CLI scripts
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── outputs/                    # Output directory (created automatically)
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gnn-rnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Prepare data by loading, augmenting, reshaping, and optionally applying PCA:

```bash
# For Z24 dataset
python scripts/prepare_data.py --config configs/z24_config.yaml

# For CDV103 dataset
python scripts/prepare_data.py --config configs/cdv103_config.yaml
```

This will create:
- `reshaped_data.npy`: Reshaped time series data
- `reshaped_labels.npy`: Corresponding labels
- `pca_data.npy`: PCA-reduced data (if PCA is enabled)

### 2. Model Training

Train a model with K-Fold cross validation:

```bash
# Train GNN-LSTM model on Z24 dataset
python scripts/train.py --config configs/z24_config.yaml

# Train a different model (override config)
python scripts/train.py --config configs/z24_config.yaml --model gnn_rnn

# Train with custom verbosity
python scripts/train.py --config configs/z24_config.yaml --verbose 0
```

Available models:
- `rnn`: Simple RNN
- `lstm`: LSTM
- `gnn`: Graph Neural Network
- `gnn_rnn`: GNN + RNN hybrid
- `gnn_lstm`: GNN + LSTM hybrid
- `attention_gnn_rnn`: GNN-RNN with attention mechanism
- `attention_gnn_lstm`: GNN-LSTM with attention mechanism
- `bilstm_gnn`: Bidirectional LSTM + GNN

### 3. Model Evaluation

Evaluate trained models and generate visualizations:

```bash
# Evaluate single model
python scripts/evaluate.py --config configs/z24_config.yaml

# Compare multiple models
python scripts/evaluate.py --config configs/z24_config.yaml --compare

# Evaluate specific model
python scripts/evaluate.py --config configs/z24_config.yaml \
    --model-path outputs/z24/models/best_gnn_lstm_model.h5
```

This will generate:
- Confusion matrix
- Training history plots
- Model comparison plots (if --compare is used)

## Configuration

Configuration files are in YAML format. Key sections:

### Dataset Configuration
```yaml
dataset:
  type: z24  # or cdv103
  name: "Z24 Bridge"
```

### Data Configuration
```yaml
data:
  data_dir: "./data/z24"
  augmentation:
    enabled: true
    num_augmentations: 10
  reshape:
    segments_per_sample: 27
    segment_length: 4000
  pca:
    enabled: false
    n_components: 50
```

### Model Configuration
```yaml
model:
  name: "gnn_lstm"
  params:
    gnn_units: [128, 64]
    lstm_units: 200
    dropout_gnn: 0.3
    dropout_lstm: 0.5
```

### Training Configuration
```yaml
training:
  epochs: 1000
  batch_size: 32
  n_splits: 5
  seed: 42
  device: "/GPU:0"
  callbacks:
    patience: 50
    lr_patience: 10
```

## Model Architectures

### 1. RNN Models
- **RNN**: Simple recurrent neural network for temporal modeling
- **LSTM**: Long Short-Term Memory for capturing long-term dependencies

### 2. GNN Models
- **GNN**: Graph Convolutional Network treating sensors as graph nodes
  - Captures spatial relationships between sensors
  - Temporal graph structure for time series
  - Global pooling for classification

### 3. Hybrid GNN-RNN Models
- **GNN-RNN**: Combines GNN for spatial features with RNN for temporal dynamics
  - GNN layers: Extract spatial patterns from sensor network
  - RNN layers: Model temporal evolution
  - Best for short-term temporal dependencies

- **Attention-GNN-RNN**: GNN-RNN with attention mechanism
  - Focuses on important time steps
  - Better interpretability

### 4. Hybrid GNN-LSTM Models
- **GNN-LSTM**: Combines GNN with LSTM for long-term memory
  - GNN layers: Spatial feature extraction
  - LSTM layers: Long-term temporal modeling
  - Best for complex temporal patterns

- **Attention-GNN-LSTM**: GNN-LSTM with attention
  - Attention weights on temporal features
  - Improved focus on critical patterns

- **BiLSTM-GNN**: Bidirectional LSTM with GNN preprocessing
  - Captures both past and future context
  - Enhanced temporal understanding

## Datasets

### Z24 Bridge Dataset
- 17 .mat files (01setup05.mat to 17setup05.mat)
- Located in `data/z24/`
- Each file contains acceleration data with shape (>=64000, 27)
- 10 damage states used for classification

### CDV103 Dataset
- 6 .mat files (SETUP1_TH1.mat to SETUP1_TH6.mat)
- Located in `data/cdv103/`
- Complex nested structure with multiple 'Untitled*' keys
- 6 damage states

## Output Structure

After training, the following structure is created:

```
outputs/
└── z24/  (or cdv103)
    ├── models/
    │   └── best_<model_name>_model.h5
    ├── history/
    │   └── best_<model_name>_history.pkl
    ├── results/
    │   └── <model_name>_results.txt
    └── plots/
        ├── <model_name>_confusion_matrix.png
        ├── <model_name>_training_history.png
        └── models_comparison_*.png
```

## Results Format

Training results are saved in text format:

```
Average validation accuracy across 5 folds: 0.8911 ± 0.0123
Average test accuracy across 5 folds: 0.8829 ± 0.0156
Best fold: 3 with validation accuracy: 0.9012

Detailed results for each fold:
Fold 1: Validation accuracy = 0.8756, Test accuracy = 0.8645
Fold 2: Validation accuracy = 0.8823, Test accuracy = 0.8712
...
```

## Advanced Usage

### Custom Data Pipeline

```python
from src.data import DataLoader, DataAugmenter, DataPreprocessor

# Load data
loader = DataLoader(dataset_type='z24', data_dir='./data/z24')
data, labels = loader.load_data()

# Augment
augmenter = DataAugmenter(seed=42)
aug_data, aug_labels = augmenter.augment(data, labels, num_augmentations=10)

# Preprocess
preprocessor = DataPreprocessor(seed=42)
reshaped_data, reshaped_labels = preprocessor.reshape_data(
    aug_data, aug_labels,
    segments_per_sample=27,
    segment_length=4000
)
```

### Custom Model Training

```python
from src.models import GNNLSTMModel
from src.training import ModelTrainer

# Build model
model = GNNLSTMModel(input_shape=(27, 50), num_classes=10)

# Train with K-Fold
trainer = ModelTrainer(model, n_splits=5, epochs=1000, batch_size=32)
results = trainer.train(X_train, y_train, X_test, y_test)

# Save
trainer.save_best_model('best_model.h5')
```

## Why GNN for Structural Health Monitoring?

Graph Neural Networks are particularly well-suited for structural health monitoring:

1. **Sensor Network Modeling**: Treat sensors as graph nodes with spatial relationships
2. **Structural Topology**: Capture the physical structure of the monitored system
3. **Spatial-Temporal Features**: Combine spatial patterns (GNN) with temporal dynamics (RNN/LSTM)
4. **Better Interpretability**: Graph structure reflects real-world sensor placement

## Tips

1. **GPU Usage**: Set `device: "/GPU:0"` in config for GPU training
2. **Memory Issues**: Reduce `batch_size` if running out of memory
3. **Faster Training**: Reduce `epochs` or increase `patience` for early stopping
4. **Better Results**: Try GNN-LSTM models for complex temporal patterns
5. **Reproducibility**: Keep the same `seed` value across experiments
6. **Model Selection**:
   - Use GNN for spatial pattern recognition
   - Use GNN-RNN for short-term temporal dependencies
   - Use GNN-LSTM for long-term temporal patterns

## Citation

If you use this code in your research, please cite:

```
@article{your_paper,
  title={Graph Neural Networks for Structural Health Monitoring},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]
