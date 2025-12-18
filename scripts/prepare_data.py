#!/usr/bin/env python3
"""Prepare data: load, augment, reshape, and optionally apply PCA."""

import argparse
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataLoader, DataAugmenter, DataPreprocessor
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    print(f"\n{'='*60}")
    print("Step 1: Loading data")
    print(f"{'='*60}")
    
    loader = DataLoader(
        dataset_type=config['dataset']['type'],
        data_dir=config['data']['data_dir']
    )
    
    input_data, output_labels = loader.load_data()
    print(f"Loaded data shape: {input_data.shape}")
    print(f"Labels shape: {output_labels.shape}")
    
    # Step 2: Data augmentation (if enabled)
    if config['data']['augmentation']['enabled']:
        print(f"\n{'='*60}")
        print("Step 2: Augmenting data")
        print(f"{'='*60}")
        
        augmenter = DataAugmenter(seed=config['training']['seed'])
        augmented_data, augmented_labels = augmenter.augment(
            input_data,
            output_labels,
            num_augmentations=config['data']['augmentation']['num_augmentations'],
            weights=config['data']['augmentation']['weights']
        )
        print(f"Augmented data shape: {augmented_data.shape}")
        print(f"Augmented labels shape: {augmented_labels.shape}")
    else:
        print("\nSkipping data augmentation (disabled in config)")
        augmented_data = input_data
        augmented_labels = output_labels
    
    # Step 3: Reshape data
    print(f"\n{'='*60}")
    print("Step 3: Reshaping data")
    print(f"{'='*60}")

    preprocessor = DataPreprocessor(seed=config['training']['seed'])
    auto_adjust = config['data']['reshape'].get('auto_adjust', 'none')
    reshaped_data, reshaped_labels = preprocessor.reshape_data(
        augmented_data,
        augmented_labels,
        segments_per_sample=config['data']['reshape']['segments_per_sample'],
        segment_length=config['data']['reshape']['segment_length'],
        auto_adjust=auto_adjust
    )
    print(f"Reshaped data shape: {reshaped_data.shape}")
    print(f"Reshaped labels shape: {reshaped_labels.shape}")
    
    # Save reshaped data
    reshaped_data_path = os.path.join(output_dir, 'reshaped_data.npy')
    reshaped_labels_path = os.path.join(output_dir, 'reshaped_labels.npy')
    np.save(reshaped_data_path, reshaped_data)
    np.save(reshaped_labels_path, reshaped_labels)
    print(f"\nSaved reshaped data to {reshaped_data_path}")
    print(f"Saved reshaped labels to {reshaped_labels_path}")
    
    # Step 4: Apply PCA (if enabled)
    if config['data']['pca']['enabled']:
        print(f"\n{'='*60}")
        print("Step 4: Applying PCA")
        print(f"{'='*60}")
        
        if 'n_components' in config['data']['pca']:
            pca_data = preprocessor.apply_pca(
                reshaped_data,
                n_components=config['data']['pca']['n_components']
            )
        elif 'variance_ratio' in config['data']['pca']:
            pca_data = preprocessor.apply_pca(
                reshaped_data,
                variance_ratio=config['data']['pca']['variance_ratio']
            )
        else:
            raise ValueError("PCA enabled but no n_components or variance_ratio specified")
        
        print(f"PCA data shape: {pca_data.shape}")
        explained_variance = preprocessor.get_explained_variance()
        print(f"Explained variance ratio: {explained_variance:.4f}")
        
        # Save PCA data
        pca_data_path = os.path.join(output_dir, 'pca_data.npy')
        np.save(pca_data_path, pca_data)
        print(f"\nSaved PCA data to {pca_data_path}")
    else:
        print("\nSkipping PCA (disabled in config)")
    
    print(f"\n{'='*60}")
    print("Data preparation completed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

