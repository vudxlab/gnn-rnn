"""Data loading utilities for Z24 and CDV103 datasets."""

import os
import numpy as np
from scipy.io import loadmat
from typing import Tuple, Optional


class DataLoader:
    """Load data from .mat files for Z24 and CDV103 datasets."""
    
    def __init__(self, dataset_type: str, data_dir: str):
        """
        Initialize DataLoader.
        
        Args:
            dataset_type: Either 'z24' or 'cdv103'
            data_dir: Path to directory containing .mat files
        """
        self.dataset_type = dataset_type.lower()
        self.data_dir = data_dir
        
        if self.dataset_type not in ['z24', 'cdv103']:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'z24' or 'cdv103'")
    
    def load_data(self, file_indices: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from .mat files.
        
        Args:
            file_indices: Optional list of file indices to load. If None, load all files.
        
        Returns:
            Tuple of (data, labels) where:
                - data: numpy array of shape (n_samples, n_channels, sequence_length)
                - labels: numpy array of shape (n_samples + 1,)
        """
        if self.dataset_type == 'z24':
            return self._load_z24_data(file_indices)
        else:
            return self._load_cdv103_data(file_indices)
    
    def _load_z24_data(self, file_indices: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Z24 bridge dataset.
        
        Z24 dataset has 17 .mat files (01setup05.mat to 17setup05.mat).
        Each file contains a 'data' key with shape (>=64000, 27).
        We trim to 64000 rows and transpose to get (27, 64000).
        """
        # Get list of .mat files
        mat_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.mat')])
        
        if file_indices is not None:
            mat_files = [mat_files[i] for i in file_indices if i < len(mat_files)]
        
        all_data = []
        
        for file_name in mat_files:
            file_path = os.path.join(self.data_dir, file_name)
            mat_data = loadmat(file_path)
            
            if 'data' in mat_data:
                data_matrix = mat_data['data']
                trimmed_data = data_matrix[:64000, :]  # Trim to 64000 rows
                
                if trimmed_data.shape[1] != 27:
                    print(f"Warning: {file_name} has {trimmed_data.shape[1]} columns, expected 27")
                    continue
                
                all_data.append(trimmed_data)
            else:
                print(f"Warning: {file_name} does not contain 'data' key")
        
        # Stack and swap axes to get (n_files, n_channels, sequence_length)
        final_array = np.stack(all_data, axis=0)
        final_array = np.swapaxes(final_array, 1, 2)  # (n_files, 27, 64000)
        
        # Select subset of data (files 3-13 as in original notebook)
        input_data = final_array[3:13, :, :]
        
        # Create labels
        output_labels = np.linspace(0, input_data.shape[0], input_data.shape[0] + 1)
        
        return input_data, output_labels
    
    def _load_cdv103_data(self, file_indices: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CDV103 dataset.
        
        CDV103 dataset has 6 .mat files (SETUP1_TH1.mat to SETUP1_TH6.mat).
        Each file contains multiple 'Untitled*' keys with time series data.
        We extract 130000 samples from columns 570000:700000.
        """
        files = [f"SETUP1_TH{i}.mat" for i in range(1, 7)]
        
        if file_indices is not None:
            files = [files[i] for i in file_indices if i < len(files)]
        
        data_3d_list = []
        
        for file_name in files:
            file_path = os.path.join(self.data_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found")
                continue
            
            try:
                mat_file = loadmat(file_path)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue
            
            # Get all keys starting with "Untitled"
            untitled_keys = [key for key in mat_file.keys() if key.startswith("Untitled")]
            
            data_matrices = []
            
            for key in untitled_keys:
                try:
                    # Extract data from nested structure
                    data = mat_file[key][0, 0][0]
                    arr = data.flatten()
                    
                    if arr.size == 0:
                        continue
                    
                    data_matrices.append(arr)
                except Exception as e:
                    print(f"Error processing {key} in {file_name}: {e}")
                    continue
            
            if data_matrices:
                # Check if all arrays have same length
                lengths = [arr.size for arr in data_matrices]
                if len(set(lengths)) == 1:
                    # Stack arrays vertically
                    merged_matrix = np.vstack(data_matrices)
                    
                    # Extract specific region and skip first row
                    if merged_matrix.shape[1] >= 700000:
                        DATA = merged_matrix[1:, 300000:600000]
                    else:
                        print(f"Warning: {file_name} has fewer than 700000 columns")
                        DATA = merged_matrix[1:, :]
                    
                    data_3d_list.append(DATA)
                else:
                    print(f"Warning: Arrays in {file_name} have different lengths")
        
        if not data_3d_list:
            raise ValueError("No valid data loaded from CDV103 files")
        
        # Stack to create 3D array
        data_3d = np.stack(data_3d_list, axis=0)
        
        # Create labels
        output_labels = np.linspace(0, data_3d.shape[0], data_3d.shape[0] + 1)
        
        return data_3d, output_labels

