"""
Data preprocessing module for SF parking garage data.
Aggregates hourly data to daily occupancy and prepares features for TFT.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class ParkingDataPreprocessor:
    """Preprocess parking garage data for time series forecasting."""
    
    def __init__(self, data_dir: str = "sfpark_data"):
        self.data_dir = Path(data_dir)
        self.garage_files = {
            "16th_and_hoff": "sfpark_garage_data_entriesexits_20112013_16th_and_hoff.csv",
            "civic_center": "sfpark_garage_data_entriesexits_20112013_civic_center.csv",
            "ellis_ofarrell": "sfpark_garage_data_entriesexits_20112013_ellis_ofarrell.csv",
            "japan_center": "sfpark_garage_data_entriesexits_20112013_japan_center.csv",
            "lombard_street": "sfpark_garage_data_entriesexits_20112013_lombard_street.csv",
            "moscone_center": "sfpark_garage_data_entriesexits_20112013_moscone_center.csv",
            "performing_arts": "sfpark_garage_data_entriesexits_20112013_performing_arts.csv",
            "st_marys_square": "sfpark_garage_data_entriesexits_20112013_st_marys_square.csv",
            "sutter_stockton": "sfpark_garage_data_entriesexits_20112013_sutter_stockton.csv",
            "union_square": "sfpark_garage_data_entriesexits_20112013_union_square.csv"
        }
        self.garage_names = list(self.garage_files.keys())
        
    def load_garage_data(self, garage_name: str) -> pd.DataFrame:
        """Load entries/exits data for a specific garage."""
        filename = self.garage_files.get(garage_name)
        if not filename:
            raise ValueError(f"Unknown garage: {garage_name}")
        
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Read CSV, skipping header rows
        df = pd.read_csv(file_path, skiprows=2)
        
        # Clean up column names
        df.columns = df.columns.str.strip()
        
        # The data has: Usage Type, Date, Total Entries, Total Exits
        # Rename columns for consistency
        df = df.rename(columns={
            'Date': 'DateTime',
            'Total Entries': 'Entries',
            'Total Exits': 'Exits'
        })
        
        # Drop empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Keep only needed columns
        if 'DateTime' in df.columns and 'Entries' in df.columns and 'Exits' in df.columns:
            df = df[['DateTime', 'Entries', 'Exits']]
        
        return df
    
    def calculate_occupancy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate occupancy from entries and exits.
        For daily data, we use entries as a proxy for demand/occupancy.
        """
        df = df.copy()
        
        # Parse datetime
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%Y', errors='coerce')
        df = df.dropna(subset=['DateTime'])
        df = df.sort_values('DateTime')
        
        # Convert to numeric, handling any string values
        df['Entries'] = pd.to_numeric(df['Entries'], errors='coerce')
        df['Exits'] = pd.to_numeric(df['Exits'], errors='coerce')
        
        # Drop rows with missing values
        df = df.dropna(subset=['Entries', 'Exits'])
        
        # For daily data, use entries as demand indicator
        # Or you can use average of entries and exits
        df['Occupancy'] = (df['Entries'] + df['Exits']) / 2
        
        # Also keep net change for additional insights
        df['Net_Change'] = df['Entries'] - df['Exits']
        
        return df
    
    def aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to daily statistics (data is already daily)."""
        df = df.copy()
        df['Date'] = df['DateTime'].dt.date
        
        # Data is already daily, so just organize it
        daily = df.groupby('Date').agg({
            'Occupancy': 'mean',  # Use mean in case there are duplicates
            'Entries': 'sum',
            'Exits': 'sum',
            'Net_Change': 'sum'
        }).reset_index()
        
        # Rename for consistency
        daily.columns = ['Date', 'Occupancy_Mean', 'Total_Entries', 'Total_Exits', 'Net_Change']
        
        # Convert Date to datetime
        daily['Date'] = pd.to_datetime(daily['Date'])
        
        # Add max and min as same as mean (since daily data)
        daily['Occupancy_Max'] = daily['Occupancy_Mean']
        daily['Occupancy_Min'] = daily['Occupancy_Mean']
        daily['Occupancy_Std'] = 0  # No intra-day variance
        
        # Ensure no negative values
        daily['Occupancy_Mean'] = daily['Occupancy_Mean'].clip(lower=0)
        daily['Total_Entries'] = daily['Total_Entries'].clip(lower=0)
        daily['Total_Exits'] = daily['Total_Exits'].clip(lower=0)
        
        return daily
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features (day of week, month, etc.)."""
        df = df.copy()
        
        # Day of week (0=Monday, 6=Sunday)
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Month
        df['Month'] = df['Date'].dt.month
        
        # Day of month
        df['Day'] = df['Date'].dt.day
        
        # Is weekend
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Quarter
        df['Quarter'] = df['Date'].dt.quarter
        
        # Week of year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, target_col: str = 'Occupancy_Mean', 
                        lags: list = [1, 7, 14]) -> pd.DataFrame:
        """Add lagged features."""
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_Lag{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, target_col: str = 'Occupancy_Mean',
                            windows: list = [7, 14, 30]) -> pd.DataFrame:
        """Add rolling window statistics."""
        df = df.copy()
        
        for window in windows:
            df[f'{target_col}_RollingMean{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'{target_col}_RollingStd{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
        
        return df
    
    def process_single_garage(self, garage_name: str, 
                            add_lags: bool = True,
                            add_rolling: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline for a single garage."""
        print(f"Processing {garage_name}...")
        
        # Load data
        df = self.load_garage_data(garage_name)
        
        # Calculate occupancy
        df = self.calculate_occupancy(df)
        
        # Aggregate to daily
        daily_df = self.aggregate_to_daily(df)
        
        # Add temporal features
        daily_df = self.add_temporal_features(daily_df)
        
        # Add lag features
        if add_lags:
            daily_df = self.add_lag_features(daily_df)
        
        # Add rolling features
        if add_rolling:
            daily_df = self.add_rolling_features(daily_df)
        
        # Add garage identifier
        daily_df['Garage'] = garage_name
        
        # Sort by date
        daily_df = daily_df.sort_values('Date').reset_index(drop=True)
        
        print(f"  - Date range: {daily_df['Date'].min()} to {daily_df['Date'].max()}")
        print(f"  - Total days: {len(daily_df)}")
        
        return daily_df
    
    def process_all_garages(self) -> pd.DataFrame:
        """Process all garages and combine into single dataset."""
        all_data = []
        
        for garage_name in self.garage_names:
            try:
                garage_df = self.process_single_garage(garage_name)
                all_data.append(garage_df)
            except Exception as e:
                print(f"Error processing {garage_name}: {e}")
        
        # Combine all garages
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\nCombined dataset:")
        print(f"  - Total records: {len(combined_df)}")
        print(f"  - Garages: {combined_df['Garage'].nunique()}")
        print(f"  - Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        
        return combined_df
    
    def create_sequences(self, df: pd.DataFrame, 
                        encoder_length: int = 30,
                        decoder_length: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences for TFT.
        
        Args:
            df: DataFrame with all features
            encoder_length: Historical context window (days)
            decoder_length: Forecast horizon (days)
        
        Returns:
            X: Input sequences
            y: Target sequences
        """
        # Sort by garage and date
        df = df.sort_values(['Garage', 'Date']).reset_index(drop=True)
        
        sequences_X = []
        sequences_y = []
        
        # Feature columns for encoder (exclude target)
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Garage', 'Occupancy_Mean']]
        
        # Process each garage separately
        for garage in df['Garage'].unique():
            garage_df = df[df['Garage'] == garage].reset_index(drop=True)
            
            # Skip if not enough data
            if len(garage_df) < encoder_length + decoder_length:
                continue
            
            # Create sequences
            for i in range(len(garage_df) - encoder_length - decoder_length + 1):
                # Encoder input (historical context)
                encoder_data = garage_df.iloc[i:i+encoder_length][feature_cols].values
                
                # Decoder target (future values to predict)
                decoder_target = garage_df.iloc[i+encoder_length:i+encoder_length+decoder_length]['Occupancy_Mean'].values
                
                sequences_X.append(encoder_data)
                sequences_y.append(decoder_target)
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str = "processed_data.csv"):
        """Save processed data to CSV."""
        df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to {output_path}")


def main():
    """Example usage."""
    preprocessor = ParkingDataPreprocessor()
    
    # Process all garages
    combined_df = preprocessor.process_all_garages()
    
    # Save processed data
    preprocessor.save_processed_data(combined_df)
    
    # Display sample
    print("\nSample of processed data:")
    print(combined_df.head())
    
    print("\nColumns:")
    print(combined_df.columns.tolist())
    
    # Check for missing values
    print("\nMissing values:")
    print(combined_df.isnull().sum())


if __name__ == "__main__":
    main()
