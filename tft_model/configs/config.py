"""
Configuration file for TFT parking forecasting.
Modify these parameters to customize the model and training.
"""

# Data Configuration
DATA_CONFIG = {
    'data_dir': 'sfpark_garage_data',
    'processed_data_path': 'processed_data.csv',
    'encoder_length': 30,  # Days of historical context
    'decoder_length': 15,  # Days to forecast ahead
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}

# Model Configuration
MODEL_CONFIG = {
    'hidden_dim': 160,        # Hidden dimension size
    'num_heads': 4,           # Number of attention heads (must divide hidden_dim)
    'dropout': 0.1,           # Dropout rate
    'num_quantiles': 3,       # Number of quantiles to predict
    'quantiles': [0.1, 0.5, 0.9],  # Quantile values
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 15,
    'gradient_clip_norm': 1.0,
    'save_dir': 'models',
}

# Feature Configuration
FEATURE_CONFIG = {
    # Features that are known in the past
    'historical_features': [
        'Occupancy_Max', 'Occupancy_Min', 'Occupancy_Std',
        'Total_Entries', 'Total_Exits',
        'Occupancy_Mean_Lag1', 'Occupancy_Mean_Lag7', 'Occupancy_Mean_Lag14',
        'Occupancy_Mean_RollingMean7', 'Occupancy_Mean_RollingMean14',
    ],
    
    # Features that are known in advance (e.g., calendar features)
    'future_features': [
        'DayOfWeek', 'Month', 'Day', 'IsWeekend', 'Quarter'
    ],
    
    'target_col': 'Occupancy_Mean',
}

# Evaluation Configuration
EVAL_CONFIG = {
    'results_dir': 'results',
    'metrics': ['MAE', 'RMSE', 'MAPE', 'R2'],
    'create_plots': True,
    'plot_attention': True,
    'plot_feature_importance': True,
}

# Inference Configuration
INFERENCE_CONFIG = {
    'model_path': 'models/best_model.pt',
    'scaler_path': 'models/scaler.pkl',
    'feature_info_path': 'models/feature_info.json',
    'output_path': 'forecasts.csv',
}

# Garage Information
GARAGES = {
    '16th_and_hoff': {
        'name': '16th and Hoff',
        'capacity': None,  # Add capacity if known
        'location': 'Mission District'
    },
    'civic_center': {
        'name': 'Civic Center',
        'capacity': None,
        'location': 'Civic Center'
    },
    'ellis_ofarrell': {
        'name': 'Ellis O\'Farrell',
        'capacity': None,
        'location': 'Downtown'
    },
    'japan_center': {
        'name': 'Japan Center',
        'capacity': None,
        'location': 'Japantown'
    },
    'lombard': {
        'name': 'Lombard',
        'capacity': None,
        'location': 'Russian Hill'
    },
    'moscone': {
        'name': 'Moscone',
        'capacity': None,
        'location': 'SoMa'
    },
    'performing_arts': {
        'name': 'Performing Arts',
        'capacity': None,
        'location': 'Civic Center'
    },
    'sutter_stockton': {
        'name': 'Sutter Stockton',
        'capacity': None,
        'location': 'Chinatown'
    },
    'union_square': {
        'name': 'Union Square',
        'capacity': None,
        'location': 'Union Square'
    },
}


def get_config(config_name: str = 'all'):
    """
    Get configuration dictionary.
    
    Args:
        config_name: Name of config to retrieve ('data', 'model', 'training', 'all')
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'feature': FEATURE_CONFIG,
        'eval': EVAL_CONFIG,
        'inference': INFERENCE_CONFIG,
        'garages': GARAGES,
    }
    
    if config_name == 'all':
        return configs
    
    return configs.get(config_name, {})


if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("TFT Configuration")
    print("=" * 60)
    
    for name, config in get_config('all').items():
        print(f"\n{name.upper()}:")
        print(json.dumps(config, indent=2))
