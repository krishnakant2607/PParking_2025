rsync -avz --progress --exclude='*.pt' --exclude='*.pkl' --exclude='__pycache__' --exclude='.git' /Users/krishnakant/Documents/lastone/ subham_srf@iit-gpu:~/parking_forecasting/"""
Inference script for generating parking occupancy forecasts using trained TFT model.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import json
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from tft_model import TemporalFusionTransformer
from data_preprocessing import ParkingDataPreprocessor
import warnings
warnings.filterwarnings('ignore')


class SimpleScaler:
    """Simple scaler wrapper that doesn't validate feature names."""
    def __init__(self, scaler=None):
        if scaler is not None:
            self.mean_ = scaler.mean_
            self.scale_ = scaler.scale_
            self.n_features_in_ = scaler.n_features_in_
            if hasattr(scaler, 'feature_names_in_'):
                self.feature_names_in_ = scaler.feature_names_in_
            else:
                self.feature_names_in_ = None
    
    def transform(self, X):
        """Transform without feature name checks."""
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_
    
    def inverse_transform(self, X):
        """Inverse transform."""
        X = np.asarray(X)
        return X * self.scale_ + self.mean_


class ParkingForecaster:
    """Generate forecasts using trained TFT model."""
    
    def __init__(self, 
                 model_path: str = 'models/best_model.pt',
                 scaler_path: str = 'models/scaler.pkl',
                 feature_info_path: str = 'models/feature_info.json',
                 device: str = None):
        """
        Initialize forecaster with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            scaler_path: Path to fitted scaler
            feature_info_path: Path to feature information
            device: Device to run inference on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load feature info
        with open(feature_info_path, 'r') as f:
            self.feature_info = json.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Running on device: {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get feature dimensions from feature info
        historical_features = len(self.feature_info['historical_features'])
        future_features = len(self.feature_info['future_features'])
        encoder_length = self.feature_info['encoder_length']
        decoder_length = self.feature_info['decoder_length']
        
        # Initialize model with same architecture
        model = TemporalFusionTransformer(
            static_features=0,
            historical_features=historical_features,
            future_features=future_features,
            hidden_dim=160,
            num_heads=4,
            num_encoder_steps=encoder_length,
            num_decoder_steps=decoder_length,
            dropout=0.1,
            num_quantiles=3
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def prepare_input(self, 
                     historical_data: pd.DataFrame,
                     future_dates: List[datetime]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors from historical data.
        
        Args:
            historical_data: DataFrame with historical features (encoder_length days)
            future_dates: List of dates to forecast
            
        Returns:
            historical_tensor, future_tensor
        """
        encoder_length = self.feature_info['encoder_length']
        historical_features = self.feature_info['historical_features']
        future_features = self.feature_info['future_features']
        
        # Ensure we have enough historical data
        if len(historical_data) < encoder_length:
            raise ValueError(f"Need at least {encoder_length} days of historical data")
        
        # Take last encoder_length days
        historical_data = historical_data.tail(encoder_length).copy()
        
        # Extract historical features - make sure columns exist
        available_hist_features = [f for f in historical_features if f in historical_data.columns]
        if len(available_hist_features) < len(historical_features):
            missing = set(historical_features) - set(available_hist_features)
            print(f"Warning: Missing historical features: {missing}")
        
        # Create future features from dates
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['Date'] = pd.to_datetime(future_df['Date'])
        future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Day'] = future_df['Date'].dt.day
        future_df['IsWeekend'] = (future_df['DayOfWeek'] >= 5).astype(int)
        future_df['Quarter'] = future_df['Date'].dt.quarter
        
        # Extract and normalize future features
        future_values = future_df[future_features].values
        
        # Normalize using scaler
        # The scaler was fitted on all features including target
        # We need to provide all those features when transforming
        all_scaler_features = list(self.scaler.feature_names_in_)
        
        # For historical data: include all available features in correct order
        # The scaler was trained with Occupancy_Mean, so we need to include it
        hist_array = np.zeros((len(historical_data), len(all_scaler_features)))
        for i, feat in enumerate(all_scaler_features):
            if feat in historical_data.columns:
                hist_array[:, i] = historical_data[feat].values
            # Features not in historical_data will remain as 0
        
        # Transform all features
        hist_transformed = self.scaler.transform(hist_array)
        
        # Extract only the indices for historical_features (excluding target)
        hist_indices = [all_scaler_features.index(f) for f in available_hist_features if f in all_scaler_features]
        historical_normalized = hist_transformed[:, hist_indices]
        
        # For future data: include all available features in correct order
        future_array = np.zeros((len(future_df), len(all_scaler_features)))
        for i, feat in enumerate(all_scaler_features):
            if feat in future_df.columns:
                future_array[:, i] = future_df[feat].values
        
        # Transform and extract only future features
        future_transformed = self.scaler.transform(future_array)
        future_indices = [all_scaler_features.index(f) for f in future_features]
        future_normalized = future_transformed[:, future_indices]
        
        # Convert to tensors
        historical_tensor = torch.tensor(historical_normalized, dtype=torch.float32).unsqueeze(0)
        future_tensor = torch.tensor(future_normalized, dtype=torch.float32).unsqueeze(0)
        
        return historical_tensor, future_tensor
    
    def forecast(self, 
                garage_name: str,
                historical_data: pd.DataFrame,
                forecast_days: int = 15) -> Dict[str, np.ndarray]:
        """
        Generate forecast for a specific garage.
        
        Args:
            garage_name: Name of the garage
            historical_data: Historical data (must include last encoder_length days)
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with quantile predictions
        """
        # Get future dates
        last_date = historical_data['Date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        # Prepare inputs
        historical_tensor, future_tensor = self.prepare_input(historical_data, future_dates)
        
        # Move to device
        historical_tensor = historical_tensor.to(self.device)
        future_tensor = future_tensor.to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            predictions, interpretability = self.model(historical_tensor, future_tensor)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        # Create result dictionary
        quantiles = [0.1, 0.5, 0.9]
        results = {
            'dates': future_dates,
            'garage': garage_name,
        }
        
        for i, q in enumerate(quantiles):
            # Inverse transform (approximate - depends on your scaler setup)
            results[f'quantile_{q}'] = predictions[:, i]
        
        # Add interpretability outputs
        results['attention_weights'] = interpretability.get('attention_weights', None)
        results['feature_weights'] = interpretability.get('historical_weights', None)
        
        return results
    
    def forecast_all_garages(self, 
                            data_path: str = 'processed_data.csv',
                            output_path: str = 'forecasts.csv') -> pd.DataFrame:
        """
        Generate forecasts for all garages.
        
        Args:
            data_path: Path to processed data
            output_path: Path to save forecasts
            
        Returns:
            DataFrame with all forecasts
        """
        # Load data
        df = pd.read_csv(data_path, parse_dates=['Date'])
        
        all_forecasts = []
        
        garages = df['Garage'].unique()
        print(f"\nGenerating forecasts for {len(garages)} garages...")
        
        for garage in garages:
            print(f"\n  Processing {garage}...")
            
            # Get garage data
            garage_df = df[df['Garage'] == garage].sort_values('Date').reset_index(drop=True)
            
            # Skip if not enough data
            encoder_length = self.feature_info['encoder_length']
            if len(garage_df) < encoder_length:
                print(f"    ⚠ Skipping (insufficient data)")
                continue
            
            try:
                # Generate forecast
                forecast = self.forecast(garage, garage_df, forecast_days=15)
                
                # Convert to DataFrame
                forecast_df = pd.DataFrame({
                    'Garage': garage,
                    'Date': forecast['dates'],
                    'Forecast_Lower': forecast['quantile_0.1'],
                    'Forecast_Median': forecast['quantile_0.5'],
                    'Forecast_Upper': forecast['quantile_0.9']
                })
                
                all_forecasts.append(forecast_df)
                
                print(f"    ✓ Forecast generated for {len(forecast['dates'])} days")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        # Combine all forecasts
        if all_forecasts:
            combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
            
            # Save to CSV
            combined_forecasts.to_csv(output_path, index=False)
            print(f"\n✓ All forecasts saved to {output_path}")
            
            return combined_forecasts
        else:
            print("\n✗ No forecasts generated")
            return pd.DataFrame()
    
    def plot_forecast(self, 
                     forecast: Dict[str, np.ndarray],
                     historical_data: pd.DataFrame = None,
                     save_path: str = None):
        """
        Plot forecast with uncertainty bands.
        
        Args:
            forecast: Forecast dictionary from self.forecast()
            historical_data: Optional historical data to plot
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot historical data if provided
        if historical_data is not None:
            dates = pd.to_datetime(historical_data['Date'])
            values = historical_data[self.feature_info['target_col']]
            ax.plot(dates, values, 'o-', label='Historical', color='black', alpha=0.6)
        
        # Plot forecast
        forecast_dates = pd.to_datetime(forecast['dates'])
        median = forecast['quantile_0.5']
        lower = forecast['quantile_0.1']
        upper = forecast['quantile_0.9']
        
        ax.plot(forecast_dates, median, 'b-', label='Forecast (Median)', linewidth=2)
        ax.fill_between(forecast_dates, lower, upper, alpha=0.3, label='80% Prediction Interval')
        
        # Formatting
        ax.set_xlabel('Date')
        ax.set_ylabel('Occupancy')
        ax.set_title(f"15-Day Parking Occupancy Forecast - {forecast['garage']}")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main inference script."""
    print("=" * 60)
    print("TFT Parking Occupancy Forecasting - Inference")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = ParkingForecaster(
        model_path='models/best_model.pt',
        scaler_path='models/scaler.pkl',
        feature_info_path='models/feature_info.json'
    )
    
    # Generate forecasts for all garages
    forecasts_df = forecaster.forecast_all_garages(
        data_path='processed_data.csv',
        output_path='forecasts.csv'
    )
    
    # Display sample forecasts
    if not forecasts_df.empty:
        print("\n" + "=" * 60)
        print("Sample Forecasts:")
        print("=" * 60)
        print(forecasts_df.head(20))
        
        print("\nForecast Summary:")
        print(f"  Total predictions: {len(forecasts_df)}")
        print(f"  Garages: {forecasts_df['Garage'].nunique()}")
        print(f"  Date range: {forecasts_df['Date'].min()} to {forecasts_df['Date'].max()}")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
