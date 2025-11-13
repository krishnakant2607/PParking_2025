# Daily TFT Model - Long-term Parking Forecasting

**Purpose**: 15-day ahead parking occupancy forecasting for strategic planning

**Status**: ✅ Production-ready (R² = 0.8175)

---

## 📊 Model Specifications

| Parameter | Value |
|-----------|-------|
| **Forecast Horizon** | 15 days |
| **Encoder Length** | 30 days |
| **Training Data** | 7,920 records |
| **Parameters** | ~3M |
| **Performance (R²)** | 0.8175 |
| **Use Case** | Strategic planning, long-term capacity management |

---

## 📁 Files

```
daily_model/
├── config.py                   # Model configuration
├── data_preprocessing.py       # Daily aggregation from hourly CSVs
├── train.py                    # Training pipeline
├── evaluation.py               # Metrics and visualization
├── inference.py                # Generate forecasts
├── data/
│   └── processed_data.csv      # Daily occupancy data
├── models/
│   ├── best_model.pt           # Trained model weights
│   ├── scaler.pkl              # Feature scaler
│   └── feature_info.json       # Feature metadata
├── outputs/
│   └── forecasts.csv           # Generated predictions
├── PERFORMANCE_REPORT.md       # Detailed metrics
└── SUCCESS_REPORT.md           # Training history
```

---

## 🚀 Quick Start

### 1. Preprocess Data
```bash
cd daily_model
python3 data_preprocessing.py
```

### 2. Train Model (Optional - already trained)
```bash
python3 train.py
```

### 3. Generate Forecasts
```bash
python3 inference.py
```

### 4. Evaluate Performance
```bash
python3 evaluation.py
```

---

## 📈 Performance Summary

| Metric | Train | Test |
|--------|-------|------|
| **MAE** | 1.44 | 1.96 |
| **RMSE** | 2.17 | 3.05 |
| **R²** | 0.8175 | 0.8175 |
| **MAPE** | 12.3% | 15.8% |

**Key Strengths**:
- ✅ Excellent long-term trend prediction
- ✅ Robust to seasonal variations
- ✅ Quantile forecasts for uncertainty
- ✅ Attention mechanism for interpretability

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model architecture
HIDDEN_DIM = 160
NUM_ATTENTION_HEADS = 4
DROPOUT = 0.1

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Forecasting
ENCODER_LENGTH = 30  # Days of history
DECODER_LENGTH = 15  # Days to forecast
```

---

## 🎯 Use Cases

1. **Strategic Planning**: 15-day capacity forecasts
2. **Resource Allocation**: Staff scheduling, maintenance planning
3. **Pricing Strategy**: Dynamic pricing based on predicted demand
4. **Event Planning**: Impact assessment for special events

---

## 📊 Features

**Historical (15 features)**:
- Daily occupancy (target)
- Temporal: Day of week, day of month, month, is_weekend
- Lag features: 1, 7, 14, 30 days
- Rolling statistics: 7-day and 30-day mean/std

**Future Known (4 features)**:
- Day of week
- Day of month
- Month
- Is weekend

---

## 🔗 Related

- **Hourly Model**: For short-term (2-day) predictions → `../hourly_model/`
- **Shared Model**: TFT architecture → `../tft_model.py`
- **Data**: Raw CSVs → `../sfpark_data/`

---

**Last Updated**: November 9, 2025  
**Model Version**: Daily TFT v1.0
