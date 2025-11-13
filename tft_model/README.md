# TFT Model - Parking Occupancy Forecasting 🚗

**Temporal Fusion Transformer for 15-Day Parking Occupancy Forecasting**

> 🔗 **Part of the [IntelliPark](https://github.com/krishnakant2607/PParking_2025) Project**  
> This folder contains the ML forecasting model. The main web application is in the root directory.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | **0.8262** | Explains 82.6% of variance |
| **MAE** | **0.2924** | Mean Absolute Error (normalized) |
| **RMSE** | **0.4571** | Root Mean Squared Error |
| **Median AE** | **0.1652** | Median Absolute Error (robust) |
| **Directional Accuracy** | **75.9%** | Correct trend prediction |

✅ **Production-Ready Performance** - Trained on 11,250 predictions across 10 San Francisco parking garages

---

## 🏗️ Architecture

**Temporal Fusion Transformer (TFT)** - A state-of-the-art multi-horizon forecasting model with:

- **Variable Selection Networks (VSN)** - Learns feature importance dynamically
- **LSTM Encoder-Decoder** - Captures temporal dependencies
- **Multi-Head Self-Attention** - Identifies relevant historical patterns
- **Quantile Forecasting** - Provides uncertainty estimates (10th, 50th, 90th percentiles)
- **Interpretability** - Attention weights and feature importance visualization

**Model Size:** 2.3M parameters | **Training Time:** ~3.4 minutes (GPU)

---

## 🚀 Quick Start

### 1. Installation

> **Note**: This model is separate from the main Next.js web app. It requires Python, not Node.js.

```bash
# Navigate to the TFT model folder
cd tft_model

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Inference

```python
import sys
sys.path.append('src')

from inference import ParkingForecaster
import pandas as pd

# Initialize forecaster
forecaster = ParkingForecaster(
    model_path='models/best_model.pt',
    scaler_path='models/scaler.pkl',
    feature_info_path='models/feature_info.json',
    device='cpu'  # or 'cuda' if GPU available
)

# Load your data (must have at least 30 days of history)
df = pd.read_csv('your_data.csv', parse_dates=['Date'])

# Generate 15-day forecast
forecast = forecaster.forecast(
    garage_name='civic_center',
    historical_data=df,
    forecast_days=15
)

print(f"Median forecast: {forecast['quantile_0.5']}")
print(f"80% prediction interval: [{forecast['quantile_0.1']}, {forecast['quantile_0.9']}]")
```

### 3. Batch Forecasting for All Garages

```python
# Generate forecasts for all garages in dataset
forecasts_df = forecaster.forecast_all_garages(
    data_path='data/processed_data.csv',
    output_path='forecasts.csv'
)
```

---

## 📁 Folder Structure

```
tft_model/                      # This folder (Python ML model)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── src/                        # Source code
│   ├── tft_model.py           # TFT model architecture
│   ├── train.py               # Training pipeline
│   ├── inference.py           # Inference and forecasting
│   ├── evaluation.py          # Evaluation and visualization
│   └── data_preprocessing.py  # Data preprocessing utilities
├── models/                     # Trained model artifacts
│   ├── best_model.pt          # Trained model checkpoint (28 MB)
│   ├── scaler.pkl             # Feature scaler
│   ├── feature_info.json      # Feature metadata
│   ├── test_metrics.json      # Test set metrics
│   ├── training_summary.json  # Training configuration
│   └── training_history.json  # Training loss history
├── configs/                    # Configuration files
│   └── config.py              # Model and training configs
├── data/                       # Sample data
│   └── sample_data.csv        # Sample processed data (first 20 rows)
└── docs/                       # Documentation
    ├── PERFORMANCE_REPORT.md  # Detailed performance analysis
    ├── SUCCESS_REPORT.md      # Training success report
    └── MODEL_DETAILS.md       # Model architecture details
```

> **Note**: The main IntelliPark web application files (Next.js) are in the parent directory.

---

## 🎯 Input Data Format

The model expects daily aggregated parking data with these features:

### Required Columns:
- `Date` - Date column (YYYY-MM-DD)
- `Garage` - Garage identifier
- `Occupancy_Mean` - Target variable (daily average occupancy)

### Historical Features (21 features):
- **Basic Stats**: Total_Entries, Total_Exits, Net_Change, Occupancy_Max, Occupancy_Min, Occupancy_Std
- **Calendar**: DayOfWeek, Month, Day, IsWeekend, Quarter, WeekOfYear
- **Lag Features**: Occupancy_Mean_Lag1, Occupancy_Mean_Lag7, Occupancy_Mean_Lag14
- **Rolling Stats**: RollingMean7, RollingStd7, RollingMean14, RollingStd14, RollingMean30, RollingStd30

### Future Known Features (5 features):
- DayOfWeek, Month, Day, IsWeekend, Quarter

---

## 🔧 Model Configuration

From `models/training_summary.json`:

```json
{
  "model_config": {
    "hidden_dim": 128,
    "num_heads": 4,
    "encoder_length": 30,
    "decoder_length": 15,
    "dropout": 0.1
  },
  "training_date": "2025-11-09T22:10:44",
  "total_epochs": 13,
  "best_val_loss": 0.1305,
  "device": "cuda",
  "total_parameters": 2296837
}
```

---

## 📈 Usage Examples

### Example 1: Single Garage Forecast with Visualization

```python
from inference import ParkingForecaster
import pandas as pd

forecaster = ParkingForecaster(
    model_path='models/best_model.pt',
    scaler_path='models/scaler.pkl',
    feature_info_path='models/feature_info.json'
)

# Load historical data
df = pd.read_csv('your_data.csv', parse_dates=['Date'])
garage_data = df[df['Garage'] == 'civic_center'].sort_values('Date')

# Generate forecast
forecast = forecaster.forecast('civic_center', garage_data, forecast_days=15)

# Plot with uncertainty bands
forecaster.plot_forecast(
    forecast,
    historical_data=garage_data.tail(30),
    save_path='forecast_civic_center.png'
)
```

### Example 2: Evaluate Model Performance

```python
from evaluation import ForecastEvaluator
import numpy as np

evaluator = ForecastEvaluator()

# Your predictions and ground truth
y_true = np.array(...)  # shape: [n_samples, 15]
y_pred = np.array(...)  # shape: [n_samples, 15]

# Calculate metrics
metrics = evaluator.calculate_metrics(y_true, y_pred)

print(f"R² Score: {metrics['R2']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
```

### Example 3: Train Your Own Model

```python
from train import TFTTrainer
from data_preprocessing import ParkingDataPreprocessor

# Preprocess your data
preprocessor = ParkingDataPreprocessor()
df = preprocessor.load_and_preprocess('raw_data.csv')

# Initialize trainer
trainer = TFTTrainer(
    data_path='processed_data.csv',
    model_dir='models',
    batch_size=32,
    epochs=50,
    learning_rate=0.001,
    device='cuda'
)

# Train model
trainer.train()
```

---

## 📊 Understanding the Output

The model returns **quantile predictions** for probabilistic forecasting:

- **quantile_0.1** - Lower bound (10th percentile)
- **quantile_0.5** - Median prediction (most likely value)
- **quantile_0.9** - Upper bound (90th percentile)

**80% Prediction Interval** = [quantile_0.1, quantile_0.9]

This means there's an 80% probability the true value will fall within this range.

---

## 🧠 Interpretability

The model provides interpretability outputs:

```python
forecast = forecaster.forecast(garage, data, forecast_days=15)

# Feature importance (which features drive predictions)
feature_weights = forecast['feature_weights']  # shape: [30, 21]

# Attention weights (which historical timesteps are important)
attention_weights = forecast['attention_weights']  # shape: [4, 45, 45]
```

Visualize with:
```python
from evaluation import ForecastVisualizer

visualizer = ForecastVisualizer(save_dir='results')
visualizer.plot_attention_weights(attention_weights)
visualizer.plot_feature_importance(feature_weights, feature_names)
```

---

## 💡 Business Applications

### 1. Dynamic Pricing
```python
if predicted_occupancy > 0.8:
    price_multiplier = 1.25  # Increase price by 25%
elif predicted_occupancy < 0.4:
    price_multiplier = 0.85  # Discount by 15%
```

### 2. Booking Recommendations
```python
if forecast['quantile_0.9'][day] > 0.85:
    send_alert("Book early - High demand predicted!")
```

### 3. Capacity Planning
```python
# Predict staff needs 2 weeks ahead
for day, occupancy in enumerate(forecast['quantile_0.5']):
    staff_needed = calculate_staff(occupancy)
    schedule_staff(day, staff_needed)
```

---

## 📦 Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Full list in `requirements.txt`

---

## 🔬 Model Details

### Training Data
- **Period**: April 2011 - May 2013
- **Garages**: 10 San Francisco off-street parking garages
- **Records**: 7,920 daily observations
- **Split**: 70% train, 15% validation, 15% test

### Data Preprocessing
1. Daily aggregation from hourly data
2. Feature engineering (lags, rolling stats, calendar features)
3. StandardScaler normalization
4. Sequence generation (30-day encoder, 15-day decoder)

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Quantile Loss (for 0.1, 0.5, 0.9 quantiles)
- **Batch Size**: 32
- **Early Stopping**: Patience=5 epochs
- **Device**: CUDA (GPU)

---

## 📖 Documentation

- **[Performance Report](docs/PERFORMANCE_REPORT.md)** - Detailed performance analysis and benchmarking
- **[Success Report](docs/SUCCESS_REPORT.md)** - Training process and validation
- **[Model Details](docs/MODEL_DETAILS.md)** - Architecture and implementation details

---

## 🐛 Troubleshooting

### ImportError: cannot import TemporalFusionTransformer
```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python your_script.py
```

### Feature Mismatch Error
Check that your data has all required columns from `models/feature_info.json`

### CUDA Out of Memory
Reduce batch size or use CPU:
```python
forecaster = ParkingForecaster(..., device='cpu')
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **TFT Paper**: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (Lim et al., 2021)
- **Dataset**: San Francisco Municipal Transportation Agency (SFMTA)

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]

---

**Built with ❤️ for parking forecasting**
