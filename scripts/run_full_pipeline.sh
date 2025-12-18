#!/bin/bash
# Full pipeline: prepare data, train model, and evaluate

# Configuration
CONFIG_FILE="configs/z24_config.yaml"
MODEL_NAME="pca_1dcnn_bigru"

echo "=========================================="
echo "Running Full Pipeline"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL_NAME"
echo ""

# Step 1: Prepare data
echo "Step 1: Preparing data..."
python scripts/prepare_data.py --config $CONFIG_FILE
if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed"
    exit 1
fi
echo ""

# Step 2: Train model
echo "Step 2: Training model..."
python scripts/train.py --config $CONFIG_FILE --model $MODEL_NAME
if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi
echo ""

# Step 3: Evaluate model
echo "Step 3: Evaluating model..."
python scripts/evaluate.py --config $CONFIG_FILE
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="

