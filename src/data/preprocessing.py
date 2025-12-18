"""Data preprocessing utilities including reshaping and PCA."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class DataPreprocessor:
    """Preprocess time series data including reshaping and PCA."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize DataPreprocessor.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.pca = None
    
    def reshape_data(
        self,
        input_data: np.ndarray,
        label_data: np.ndarray,
        segments_per_sample: int,
        segment_length: int,
        auto_adjust: str = 'none'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape time series data into segments.

        Args:
            input_data: Data of shape (n_samples, n_channels, sequence_length)
            label_data: Labels of shape (n_samples,)
            segments_per_sample: Number of segments per new sample
            segment_length: Length of each segment
            auto_adjust: How to handle length mismatch ('none', 'truncate', 'pad')
                - 'none': Raise error if length doesn't divide evenly
                - 'truncate': Remove extra data to make it evenly divisible
                - 'pad': Add zeros to make it evenly divisible

        Returns:
            Tuple of (reshaped_data, reshaped_labels)
        """
        num_samples_original, num_channels, length_original = input_data.shape

        # Auto-adjust data length if needed
        if length_original % segment_length != 0:
            if auto_adjust == 'truncate':
                new_length = (length_original // segment_length) * segment_length
                input_data = input_data[:, :, :new_length]
                print(f"Auto-adjusted data length from {length_original} to {new_length} (truncated)")
                length_original = new_length
            elif auto_adjust == 'pad':
                new_length = ((length_original // segment_length) + 1) * segment_length
                padding = new_length - length_original
                input_data = np.pad(input_data, ((0, 0), (0, 0), (0, padding)), mode='constant')
                print(f"Auto-adjusted data length from {length_original} to {new_length} (padded with {padding} zeros)")
                length_original = new_length
            else:
                raise ValueError(f"Segment length must evenly divide the original length. "
                               f"Original length: {length_original}, Segment length: {segment_length}. "
                               f"Set auto_adjust='truncate' or 'pad' to handle this automatically.")
        
        total_segments_per_original_sample = (length_original // segment_length) * num_channels
        num_samples_new = (num_samples_original * total_segments_per_original_sample) // segments_per_sample
        
        if (num_samples_original * total_segments_per_original_sample) % segments_per_sample != 0:
            raise ValueError("Reshaping not possible with the given dimensions.")
        
        # Initialize reshaped arrays
        new_shape = (num_samples_new, segments_per_sample, segment_length)
        reshaped_data = np.zeros(new_shape)
        reshaped_labels = np.zeros(num_samples_new)
        
        # Reshape the data and labels
        count = 0
        for i in range(num_samples_original):
            segment_count = 0
            for j in range(num_channels):
                for k in range(length_original // segment_length):
                    start_idx = k * segment_length
                    end_idx = start_idx + segment_length
                    reshaped_data[count, segment_count % segments_per_sample, :] = \
                        input_data[i, j, start_idx:end_idx]
                    
                    if (segment_count + 1) % segments_per_sample == 0:
                        reshaped_labels[count] = label_data[i]
                        count += 1
                    segment_count += 1
        
        return reshaped_data, reshaped_labels
    
    def apply_pca(
        self,
        data: np.ndarray,
        n_components: Optional[int] = None,
        variance_ratio: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply PCA to reduce dimensionality.
        
        Args:
            data: Data of shape (n_samples, n_segments, n_features)
            n_components: Number of components to keep (if None, use variance_ratio)
            variance_ratio: Variance ratio to preserve (e.g., 0.95 for 95%)
        
        Returns:
            Reduced data of shape (n_samples, n_segments, n_components)
        """
        # Reshape to 2D for PCA
        n_samples, n_segments, n_features = data.shape
        X_2d = data.reshape(-1, n_features)
        
        # Initialize PCA
        if n_components is not None:
            self.pca = PCA(n_components=n_components, random_state=self.seed)
        elif variance_ratio is not None:
            self.pca = PCA(n_components=variance_ratio, random_state=self.seed)
        else:
            raise ValueError("Either n_components or variance_ratio must be specified")
        
        # Fit and transform
        X_2d_reduced = self.pca.fit_transform(X_2d)
        
        # Reshape back to 3D
        n_components_actual = X_2d_reduced.shape[1]
        X_reduced = X_2d_reduced.reshape(n_samples, n_segments, n_components_actual)
        
        return X_reduced
    
    def split_data(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.3,
        val_size: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input data
            labels: Labels
            test_size: Proportion of data for test+validation
            val_size: Proportion of test+validation data for validation
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train vs (test + validation)
        X_train, X_temp, y_train, y_temp = train_test_split(
            data, labels, test_size=test_size, random_state=self.seed
        )
        
        # Second split: validation vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.seed
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_explained_variance(self) -> Optional[float]:
        """Get the explained variance ratio from the last PCA transformation."""
        if self.pca is not None:
            return self.pca.explained_variance_ratio_.sum()
        return None

