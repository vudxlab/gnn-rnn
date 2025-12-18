# Quick Start Guide

This guide will help you get started with the GNN-RNN framework for structural health monitoring in just a few minutes.

## Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional, but recommended)
- Z24 or CDV103 dataset

## Installation

```bash
# Navigate to project directory
cd /path/to/gnn-rnn

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Dataset Setup

Your datasets should be organized as follows:

```
gnn-rnn/
└── data/
    ├── z24/
    │   ├── 01setup05.mat
    │   ├── 02setup05.mat
    │   └── ...
    └── cdv103/
        ├── SETUP1_TH1.mat
        ├── SETUP1_TH2.mat
        └── ...
```

## Quick Example: Z24 Dataset

### Step-by-Step Pipeline

```bash
# Step 1: Prepare data
python scripts/prepare_data.py --config configs/z24_config.yaml

# Step 2: Train model
python scripts/train.py --config configs/z24_config.yaml

# Step 3: Evaluate model
python scripts/evaluate.py --config configs/z24_config.yaml
```

### Check Results

After training, you'll find:
- **Models**: `outputs/z24/models/best_<model_name>_model.h5`
- **History**: `outputs/z24/history/best_<model_name>_history.pkl`
- **Results**: `outputs/z24/results/<model_name>_results.txt`
- **Plots**: `outputs/z24/plots/`

## Training Different Models

Train different models by changing the `--model` parameter:

```bash
# Train GNN model
python scripts/train.py --config configs/z24_config.yaml --model gnn

# Train GNN-RNN hybrid
python scripts/train.py --config configs/z24_config.yaml --model gnn_rnn

# Train GNN-LSTM hybrid (recommended for complex patterns)
python scripts/train.py --config configs/z24_config.yaml --model gnn_lstm

# Train with attention mechanism
python scripts/train.py --config configs/z24_config.yaml --model attention_gnn_lstm
```

### Available Models:

| Model | Description | Best For |
|-------|-------------|----------|
| `rnn` | Simple RNN | Basic temporal patterns |
| `lstm` | LSTM | Long-term dependencies |
| `gnn` | Graph Neural Network | Spatial relationships |
| `gnn_rnn` | GNN + RNN hybrid | Short-term spatial-temporal |
| `gnn_lstm` | GNN + LSTM hybrid | Complex spatial-temporal patterns |
| `attention_gnn_rnn` | GNN-RNN + Attention | Interpretable spatial-temporal |
| `attention_gnn_lstm` | GNN-LSTM + Attention | Advanced pattern recognition |
| `bilstm_gnn` | BiLSTM + GNN | Enhanced temporal context |

## Customizing Configuration

Edit the config file to customize training:

```yaml
# configs/z24_config.yaml

data:
  data_dir: "./data/z24"        # Dataset location
  augmentation:
    enabled: true
    num_augmentations: 10       # Increase for more training data

training:
  epochs: 500                   # Reduce for faster training
  batch_size: 64                # Increase if you have enough memory
  n_splits: 3                   # Reduce for faster training
  device: "/GPU:0"              # Use "/CPU:0" if no GPU

  callbacks:
    patience: 30                # Early stopping patience
    lr_patience: 10             # Learning rate reduction patience

model:
  name: "gnn_lstm"              # Change model here
  params:
    gnn_units: [128, 64]        # GNN layer sizes
    lstm_units: 200             # LSTM units
    dropout_gnn: 0.3            # Dropout for GNN
    dropout_lstm: 0.5           # Dropout for LSTM
```

## Using CDV103 Dataset

For CDV103 dataset, simply use the CDV103 config:

```bash
# Prepare data
python scripts/prepare_data.py --config configs/cdv103_config.yaml

# Train
python scripts/train.py --config configs/cdv103_config.yaml --model gnn_lstm

# Evaluate
python scripts/evaluate.py --config configs/cdv103_config.yaml
```

## Comparing Multiple Models

After training multiple models, compare them:

```bash
# Train different models
python scripts/train.py --config configs/z24_config.yaml --model rnn
python scripts/train.py --config configs/z24_config.yaml --model lstm
python scripts/train.py --config configs/z24_config.yaml --model gnn_lstm

# Compare all trained models
python scripts/evaluate.py --config configs/z24_config.yaml --compare
```

This will generate comparison plots for all trained models.

## Model Selection Guide

### 1. Start Simple
```bash
# Train baseline LSTM
python scripts/train.py --config configs/z24_config.yaml --model lstm
```

### 2. Add Spatial Modeling
```bash
# Train GNN to capture sensor relationships
python scripts/train.py --config configs/z24_config.yaml --model gnn
```

### 3. Combine Spatial-Temporal
```bash
# Train GNN-LSTM for best of both worlds
python scripts/train.py --config configs/z24_config.yaml --model gnn_lstm
```

### 4. Add Attention (Optional)
```bash
# Train with attention for interpretability
python scripts/train.py --config configs/z24_config.yaml --model attention_gnn_lstm
```

## Tips for Better Results

### 1. Enable PCA for Faster Training
```yaml
# In config file
pca:
  enabled: true
  n_components: 50
```

### 2. Use GPU Acceleration
```yaml
training:
  device: "/GPU:0"
```

### 3. Increase Data Augmentation
```yaml
augmentation:
  num_augmentations: 20  # More training data
```

### 4. Tune Hyperparameters
Try different values for:
- `gnn_units`: [64, 32] for faster, [256, 128] for more capacity
- `lstm_units`: 100-400 depending on data complexity
- `dropout`: 0.2-0.6 to prevent overfitting

### 5. Use Attention for Interpretability
Attention models show which time steps are important for classification.

## Troubleshooting

### Out of Memory Error
```yaml
# Reduce batch size
training:
  batch_size: 16  # or even 8

# Enable PCA
pca:
  enabled: true
  n_components: 30

# Use CPU
training:
  device: "/CPU:0"
```

### Training Too Slow
```yaml
# Reduce epochs
training:
  epochs: 300

# Increase patience for early stopping
callbacks:
  patience: 50

# Use smaller model
model:
  name: "rnn"  # instead of gnn_lstm
```

### Poor Accuracy
```bash
# 1. Check data loading
python scripts/prepare_data.py --config configs/z24_config.yaml

# 2. Try different models
python scripts/train.py --config configs/z24_config.yaml --model gnn_lstm

# 3. Increase augmentation
# Edit config: num_augmentations: 20

# 4. Adjust hyperparameters
# Edit config: increase units, adjust dropout
```

### Model Not Improving
```yaml
# Check learning rate
# It might be reduced too much

# Increase lr_patience
callbacks:
  lr_patience: 15

# Or disable lr reduction temporarily
```

## Understanding Results

### Training Output
```
Fold 1/5
Epoch 1/1000: loss: 1.2345 - accuracy: 0.7234 - val_loss: 1.4567 - val_accuracy: 0.6892
...
Best validation accuracy: 0.8911
```

### Results File
```
Average validation accuracy across 5 folds: 0.8911 ± 0.0123
Average test accuracy across 5 folds: 0.8829 ± 0.0156
Best fold: 3 with validation accuracy: 0.9012
```

## Next Steps

### Learn More
- Read the full [README.md](README.md) for detailed documentation
- Explore [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture details
- Check [PROJECT_STRUCTURE.txt](PROJECT_STRUCTURE.txt) for code organization

### Customize
- Modify models in [src/models/](src/models/) for custom architectures
- Add new augmentation techniques in [src/data/augmentation.py](src/data/augmentation.py)
- Implement custom metrics in [src/evaluation/metrics.py](src/evaluation/metrics.py)

### Experiment
1. Try all models and compare results
2. Experiment with different hyperparameters
3. Test with and without PCA
4. Adjust augmentation techniques
5. Create custom hybrid architectures

## Common Workflows

### Quick Test
```bash
# Fast training for testing
python scripts/train.py --config configs/z24_config.yaml \
  --model rnn --epochs 100
```

### Production Training
```bash
# Full training with best model
python scripts/train.py --config configs/z24_config.yaml \
  --model gnn_lstm --epochs 1000
```

### Hyperparameter Search
```bash
# Train with different configs
for model in rnn lstm gnn gnn_rnn gnn_lstm; do
  python scripts/train.py --config configs/z24_config.yaml --model $model
done

# Compare results
python scripts/evaluate.py --config configs/z24_config.yaml --compare
```

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Review the configuration file
3. Ensure data is in `data/z24/` or `data/cdv103/`
4. Check GPU availability: `nvidia-smi`
5. Read the full README.md
6. Open an issue on GitHub with:
   - Error message
   - Configuration file
   - Python version and dependencies

## Quick Reference

### File Locations
- **Data**: `data/z24/` or `data/cdv103/`
- **Configs**: `configs/z24_config.yaml`, `configs/cdv103_config.yaml`
- **Scripts**: `scripts/prepare_data.py`, `scripts/train.py`, `scripts/evaluate.py`
- **Models**: `src/models/*.py`
- **Outputs**: `outputs/z24/` or `outputs/cdv103/`

### Key Commands
```bash
# Prepare
python scripts/prepare_data.py --config configs/z24_config.yaml

# Train
python scripts/train.py --config configs/z24_config.yaml --model gnn_lstm

# Evaluate
python scripts/evaluate.py --config configs/z24_config.yaml --compare
```

Happy training!
