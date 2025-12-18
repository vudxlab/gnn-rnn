"""Data augmentation utilities for time series data."""

import numpy as np
import random
from typing import Tuple, List


class DataAugmenter:
    """Augment time series data using various techniques."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize DataAugmenter.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
    
    def augment(
        self, 
        input_data: np.ndarray, 
        labels: np.ndarray, 
        num_augmentations: int = 10,
        weights: List[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment time series data.
        
        Args:
            input_data: Original data of shape (n_samples, n_channels, sequence_length)
            labels: Corresponding labels
            num_augmentations: Number of augmented samples per original sample
            weights: Weights for each augmentation technique [noise, reverse, crop_pad, time_warp, random_shift]
        
        Returns:
            Tuple of (augmented_data, augmented_labels)
        """
        if weights is None:
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        augmented_data = []
        augmented_labels = []
        
        num_samples, num_channels, sequence_length = input_data.shape
        
        for i in range(num_samples):
            for _ in range(num_augmentations):
                # Choose augmentation technique
                augmentation_type = random.choices(
                    ['noise', 'reverse', 'crop_pad', 'time_warp', 'random_shift'],
                    weights=weights
                )[0]
                
                if augmentation_type == 'noise':
                    augmented_sample = self._add_noise(input_data[i])
                elif augmentation_type == 'reverse':
                    augmented_sample = self._reverse(input_data[i])
                elif augmentation_type == 'crop_pad':
                    augmented_sample = self._crop_pad(input_data[i], sequence_length)
                elif augmentation_type == 'time_warp':
                    augmented_sample = self._time_warp(input_data[i], sequence_length)
                elif augmentation_type == 'random_shift':
                    augmented_sample = self._random_shift(input_data[i], sequence_length)
                
                # Only add if shape is correct
                if augmented_sample.shape == (num_channels, sequence_length):
                    augmented_data.append(augmented_sample)
                    augmented_labels.append(labels[i])
        
        return np.array(augmented_data), np.array(augmented_labels)
    
    def _add_noise(self, sample: np.ndarray, noise_level: float = 0.0001) -> np.ndarray:
        """Add Gaussian noise to the sample."""
        noise = np.random.normal(0, noise_level, sample.shape)
        return sample + noise
    
    def _reverse(self, sample: np.ndarray) -> np.ndarray:
        """Reverse the time sequence."""
        return np.flip(sample, axis=-1)
    
    def _crop_pad(self, sample: np.ndarray, sequence_length: int) -> np.ndarray:
        """Crop and pad the sequence."""
        crop_size = random.randint(1, sequence_length // 100)
        padded_sample = np.pad(sample, ((0, 0), (crop_size, 0)), mode='constant', constant_values=0)
        return padded_sample[:, :-crop_size]
    
    def _time_warp(self, sample: np.ndarray, sequence_length: int) -> np.ndarray:
        """Apply time warping by averaging a segment."""
        start_idx = random.randint(0, sequence_length // 2)
        end_idx = random.randint(start_idx, sequence_length)
        
        warped_segment = np.mean(sample[:, start_idx:end_idx], axis=1, keepdims=True)
        return np.concatenate((warped_segment, sample[:, end_idx:]), axis=1)
    
    def _random_shift(self, sample: np.ndarray, sequence_length: int) -> np.ndarray:
        """Randomly shift the sequence."""
        shift_amount = random.randint(-(sequence_length // 10), sequence_length // 10)
        return np.roll(sample, shift_amount, axis=-1)

