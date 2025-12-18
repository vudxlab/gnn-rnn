"""Configuration loading and saving utilities."""

import yaml
import json
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif config_path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    print(f"Config saved to {config_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['dataset', 'data', 'model', 'training']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate dataset config
    if 'type' not in config['dataset']:
        raise ValueError("Missing 'type' in dataset config")
    
    if config['dataset']['type'] not in ['z24', 'cdv103']:
        raise ValueError(f"Invalid dataset type: {config['dataset']['type']}")
    
    # Validate data config
    if 'data_dir' not in config['data']:
        raise ValueError("Missing 'data_dir' in data config")
    
    # Validate model config
    if 'name' not in config['model']:
        raise ValueError("Missing 'name' in model config")
    
    # Validate training config
    required_training_keys = ['epochs', 'batch_size', 'n_splits']
    for key in required_training_keys:
        if key not in config['training']:
            raise ValueError(f"Missing '{key}' in training config")
    
    return True

