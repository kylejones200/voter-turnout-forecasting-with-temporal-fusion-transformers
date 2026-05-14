import logging

logger = logging.getLogger(__name__)

# Extracted code from '02_Retail-Sales-Forecasting-with-TFT.md'
# Blocks appear in the same order as in the markdown article.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load voter turnout data
data_path = Path("timeseries/2025-11-12_us_voter_turnout.csv")
df = pd.read_csv(data_path)

# Clean and prepare
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df = df.sort_values('Year')
df = df[df['Turnout Rate'].notna()]

# Create time series
ts = df.set_index('Year')['Turnout Rate']

logger.info(f"Time series length: {len(ts)}")
logger.info(f"Date range: {ts.index.min()} to {ts.index.max()}")
logger.info(f"Value range: {ts.min():.2f}% to {ts.max():.2f}%")
logger.info(f"\nFirst 10 values:\n{ts.head(10)}")
logger.info(f"\nLast 10 values:\n{ts.tail(10)}")

# Visualize
plt.rcParams.update({
    'axes.grid': False,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False
})

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts.index, ts.values, marker='o', linewidth=2, markersize=4, alpha=0.7)
ax.set_title('US Presidential Voter Turnout (1789-2024)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Turnout Rate (%)', fontsize=11)
plt.tight_layout()
plt.savefig('voter_turnout_series.png', dpi=300, bbox_inches='tight')
plt.show()

# Install: pip install pytorch-forecasting pytorch-lightning

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Prepare data for TFT
df_tft = df.copy()
df_tft['time_idx'] = range(len(df_tft))
df_tft['target'] = df_tft['Turnout Rate']

# Add time-based features
df_tft['year'] = df_tft['Year'].dt.year
df_tft['decade'] = (df_tft['year'] // 10) * 10
df_tft['century'] = (df_tft['year'] // 100) * 100

# Configuration
max_encoder_length = 20  # Use 20 elections as history
max_prediction_length = 4  # Forecast 4 elections ahead

logger.info(f"Encoder length: {max_encoder_length} elections")
logger.info(f"Prediction length: {max_prediction_length} elections")

# Create training dataset
training = TimeSeriesDataSet(
    df_tft[df_tft['time_idx'] < len(df_tft) - max_prediction_length],
    time_idx="time_idx",
    target="target",
    group_ids=["century"],  # Group by century for static features
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["century"],
    time_varying_known_reals=["time_idx", "year"],
    time_varying_unknown_reals=["target"],
    target_normalizer=None,  # Keep original scale
    add_relative_time_idx=True,
    add_target_scales=True,
    allow_missing_timesteps=True,
)

# Create validation set
validation = TimeSeriesDataSet.from_dataset(
    training, df_tft, predict=True, stop_randomization=True
)

# Create dataloaders
train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=32, num_workers=0)

logger.info(f"Training samples: {len(training)}")
logger.info(f"Validation samples: {len(validation)}")

# Initialize TFT model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles for uncertainty
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
    optimizer="Adam"
)

logger.info(f"Model parameters: {sum(p.numel() for p in tft.parameters()):,}")

# Train model
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
    ],
    enable_progress_bar=True,
)

logger.info("Training TFT model...")
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Load best model
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
logger.info(f"Best model saved at: {best_model_path}")

# Make predictions
predictions = best_tft.predict(val_dataloader)
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)], dim=0)

# Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Use median prediction (index 3 of 7 quantiles)
tft_pred_median = predictions.numpy()[:, 3]
actuals_np = actuals.numpy()

tft_mae = mean_absolute_error(actuals_np, tft_pred_median)
tft_rmse = np.sqrt(mean_squared_error(actuals_np, tft_pred_median))

logger.info(f"TFT MAE: {tft_mae:.2f}%")
logger.info(f"TFT RMSE: {tft_rmse:.2f}%")

from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA on training data
train_end_idx = len(df_tft) - max_prediction_length
ts_train = ts[:ts.index[train_end_idx]]
ts_test = ts[ts.index[train_end_idx]:]

logger.info(f"ARIMA training on {len(ts_train)} elections")
logger.info(f"Forecasting {len(ts_test)} elections ahead")

# Fit ARIMA
arima_model = ARIMA(ts_train, order=(2, 1, 2))
arima_fitted = arima_model.fit()

# Forecast
arima_forecast = arima_fitted.forecast(steps=len(ts_test))

# Evaluate
arima_mae = mean_absolute_error(ts_test.values, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(ts_test.values, arima_forecast))

logger.info(f"ARIMA MAE: {arima_mae:.2f}%")
logger.info(f"ARIMA RMSE: {arima_rmse:.2f}%")

# Create comparison
comparison_df = pd.DataFrame({
    'Year': ts_test.index.year,
    'Actual': ts_test.values,
    'TFT': tft_pred_median[:len(ts_test)],
    'ARIMA': arima_forecast[:len(ts_test)]
})

logger.info("=== FORECAST COMPARISON ===")
logger.info(comparison_df.to_string(index=False))

# Metrics comparison
results = {
    'TFT': {'MAE': tft_mae, 'RMSE': tft_rmse},
    'ARIMA': {'MAE': arima_mae, 'RMSE': arima_rmse}
}

logger.info("=== METRICS COMPARISON ===")
logger.info(f"{'Model':<10} {'MAE (%)':<12} {'RMSE (%)':<12}")
for model, metrics in results.items():
    logger.info(f"{model:<10} {metrics['MAE']:<12.2f} {metrics['RMSE']:<12.2f}")

# Plot comparison
fig, ax = plt.subplots(figsize=(14, 7))
plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Plot historical data
historical_dates = ts.index[:train_end_idx]
ax.plot(historical_dates[-30:], ts.values[train_end_idx-30:train_end_idx], 
        'k-', linewidth=2, label='Historical', alpha=0.7)

# Plot actual test values
ax.plot(ts_test.index, ts_test.values, 
        'o-', linewidth=2, markersize=10, label='Actual', color='black')

# Plot TFT forecast with uncertainty
tft_pred_lower = predictions.numpy()[:len(ts_test), 1]  # 10th percentile
tft_pred_upper = predictions.numpy()[:len(ts_test), 5]  # 90th percentile
ax.fill_between(ts_test.index, tft_pred_lower, tft_pred_upper, 
                alpha=0.2, color='green', label='TFT 80% Interval')
ax.plot(ts_test.index, tft_pred_median[:len(ts_test)], 
        '-', linewidth=2, label='TFT Forecast', color='green')

# Plot ARIMA forecast
ax.plot(ts_test.index, arima_forecast, 
        '--', linewidth=2, label='ARIMA Forecast', color='blue')

# Add forecast boundary
ax.axvline(ts_test.index[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Turnout Rate (%)', fontsize=11)
ax.set_title('TFT vs ARIMA: Multi-Horizon Forecasting', 
             fontsize=13, fontweight='bold')
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig('tft_vs_arima.png', dpi=300, bbox_inches='tight')
plt.show()

# Save model for production
torch.save(best_tft.state_dict(), 'tft_model.pth')

# Save dataset for inference
import pickle
with open('tft_dataset.pkl', 'wb') as f:
    pickle.dump(training, f)

# Load and use
loaded_tft = TemporalFusionTransformer.from_dataset(training)
loaded_tft.load_state_dict(torch.load('tft_model.pth'))
loaded_tft.eval()

# Make new predictions
new_predictions = loaded_tft.predict(val_dataloader)

# Complete code for reproducibility
# All imports, data loading, model training, and evaluation
# See individual code blocks above for full implementation
