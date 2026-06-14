#!/usr/bin/env python3
"""Train a Temporal Fusion Transformer and compare forecasts against ARIMA."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent
DATA_PATH = REPO_ROOT / "data" / "us_voter_turnout.csv"
IMAGES_DIR = REPO_ROOT / "images"


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as handle:
        return yaml.safe_load(handle) or {}


def load_turnout_series(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean the voter turnout series."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Expected data/us_voter_turnout.csv in the repo."
        )

    df = pd.read_csv(csv_path)
    if "Turnout_Rate" in df.columns:
        df = df.rename(columns={"Turnout_Rate": "Turnout Rate"})
    df["Year"] = pd.to_datetime(df["Year"], format="%Y")
    df = df.sort_values("Year")
    df = df[df["Turnout Rate"].notna()]

    ts = df.set_index("Year")["Turnout Rate"]
    logger.info("Time series length: %s", len(ts))
    logger.info("Date range: %s to %s", ts.index.min(), ts.index.max())
    logger.info("Value range: %.2f%% to %.2f%%", ts.min(), ts.max())
    return df, ts


def plot_turnout_series(ts: pd.Series, config: dict, images_dir: Path) -> None:
    figsize = tuple(config.get("output", {}).get("figsize", [14, 6]))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts.index, ts.values, marker="o", linewidth=2, markersize=4, alpha=0.7)
    ax.set_title("US Presidential Voter Turnout (1789-2024)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Turnout Rate (%)", fontsize=11)
    plt.tight_layout()
    output = images_dir / "voter_turnout_series.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", output)


def prepare_tft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_tft = df.copy()
    df_tft["time_idx"] = range(len(df_tft))
    df_tft["target"] = df_tft["Turnout Rate"]
    df_tft["year"] = df_tft["Year"].dt.year
    df_tft["decade"] = (df_tft["year"] // 10) * 10
    df_tft["century"] = ((df_tft["year"] // 100) * 100).astype(str)
    return df_tft


def build_datasets(
    df_tft: pd.DataFrame,
    max_encoder_length: int,
    max_prediction_length: int,
    batch_size: int,
):
    from pytorch_forecasting import TimeSeriesDataSet

    training = TimeSeriesDataSet(
        df_tft[df_tft["time_idx"] < len(df_tft) - max_prediction_length],
        time_idx="time_idx",
        target="target",
        group_ids=["century"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["century"],
        time_varying_known_reals=["time_idx", "year"],
        time_varying_unknown_reals=["target"],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, df_tft, predict=True, stop_randomization=True
    )
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    logger.info("Training samples: %s", len(training))
    logger.info("Validation samples: %s", len(validation))
    return training, train_dataloader, val_dataloader


def train_tft(
    training,
    train_dataloader,
    val_dataloader,
    config: dict,
    max_epochs: int = 30,
):
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    model_cfg = config.get("model", {})
    torch.manual_seed(config.get("data", {}).get("seed", 42))
    np.random.seed(config.get("data", {}).get("seed", 42))

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=model_cfg.get("learning_rate", 0.03),
        hidden_size=model_cfg.get("hidden_size", 16),
        attention_head_size=4,
        dropout=model_cfg.get("dropout", 0.1),
        hidden_continuous_size=8,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        optimizer="Adam",
    )
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in tft.parameters()):,}")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=10,
                verbose=False,
                mode="min",
            ),
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
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    logger.info("Best model saved at: %s", best_model_path)
    return best_tft


def evaluate_tft_and_arima(
    best_tft,
    val_dataloader,
    ts: pd.Series,
    df_tft: pd.DataFrame,
    max_prediction_length: int,
):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.tsa.arima.model import ARIMA

    predictions = best_tft.predict(val_dataloader)
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)], dim=0)
    tft_pred_median = predictions.numpy()[:, 3]
    actuals_np = actuals.numpy()
    tft_mae = mean_absolute_error(actuals_np, tft_pred_median)
    tft_rmse = np.sqrt(mean_squared_error(actuals_np, tft_pred_median))
    logger.info("TFT MAE: %.2f%%", tft_mae)
    logger.info("TFT RMSE: %.2f%%", tft_rmse)

    train_end_idx = len(df_tft) - max_prediction_length
    ts_train = ts[: ts.index[train_end_idx]]
    ts_test = ts[ts.index[train_end_idx] :]
    logger.info("ARIMA training on %s elections", len(ts_train))
    logger.info("Forecasting %s elections ahead", len(ts_test))

    arima_model = ARIMA(ts_train, order=(2, 1, 2))
    arima_fitted = arima_model.fit()
    arima_forecast = arima_fitted.forecast(steps=len(ts_test))
    arima_mae = mean_absolute_error(ts_test.values, arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(ts_test.values, arima_forecast))
    logger.info("ARIMA MAE: %.2f%%", arima_mae)
    logger.info("ARIMA RMSE: %.2f%%", arima_rmse)

    comparison_df = pd.DataFrame(
        {
            "Year": ts_test.index.year,
            "Actual": ts_test.values,
            "TFT": tft_pred_median[: len(ts_test)],
            "ARIMA": arima_forecast[: len(ts_test)],
        }
    )
    logger.info("=== FORECAST COMPARISON ===\n%s", comparison_df.to_string(index=False))
    logger.info(
        "=== METRICS COMPARISON ===\nModel     MAE (%)     RMSE (%)   \nTFT       %.2f        %.2f\nARIMA     %.2f        %.2f",
        tft_mae,
        tft_rmse,
        arima_mae,
        arima_rmse,
    )
    return predictions, tft_pred_median, ts_test, arima_forecast, train_end_idx


def plot_forecast_comparison(
    ts: pd.Series,
    ts_test: pd.Series,
    predictions,
    tft_pred_median: np.ndarray,
    arima_forecast: np.ndarray,
    train_end_idx: int,
    images_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    historical_dates = ts.index[:train_end_idx]
    ax.plot(
        historical_dates[-30:],
        ts.values[train_end_idx - 30 : train_end_idx],
        "k-",
        linewidth=2,
        label="Historical",
        alpha=0.7,
    )
    ax.plot(
        ts_test.index,
        ts_test.values,
        "o-",
        linewidth=2,
        markersize=10,
        label="Actual",
        color="black",
    )
    tft_pred_lower = predictions.numpy()[: len(ts_test), 1]
    tft_pred_upper = predictions.numpy()[: len(ts_test), 5]
    ax.fill_between(
        ts_test.index,
        tft_pred_lower,
        tft_pred_upper,
        alpha=0.2,
        color="green",
        label="TFT 80% Interval",
    )
    ax.plot(
        ts_test.index,
        tft_pred_median[: len(ts_test)],
        "-",
        linewidth=2,
        label="TFT Forecast",
        color="green",
    )
    ax.plot(
        ts_test.index,
        arima_forecast,
        "--",
        linewidth=2,
        label="ARIMA Forecast",
        color="blue",
    )
    ax.axvline(ts_test.index[0], color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_ylabel("Turnout Rate (%)", fontsize=11)
    ax.set_title("TFT vs ARIMA: Multi-Horizon Forecasting", fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    output = images_dir / "tft_vs_arima.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", output)


def save_model_and_dataset(best_tft, training, val_dataloader, output_dir: Path) -> None:
    from pytorch_forecasting import TemporalFusionTransformer

    model_path = output_dir / "tft_model.pth"
    dataset_path = output_dir / "tft_dataset.pkl"
    torch.save(best_tft.state_dict(), model_path)
    with open(dataset_path, "wb") as handle:
        pickle.dump(training, handle)

    loaded_tft = TemporalFusionTransformer.from_dataset(training)
    loaded_tft.load_state_dict(torch.load(model_path))
    loaded_tft.eval()
    _ = loaded_tft.predict(val_dataloader)
    logger.info("Saved model artifacts to %s", output_dir)


def main() -> None:
    config = load_config()
    signalplot.apply(font_family=config.get("output", {}).get("font_family", "serif"))
    IMAGES_DIR.mkdir(exist_ok=True)

    df, ts = load_turnout_series(DATA_PATH)
    plot_turnout_series(ts, config, IMAGES_DIR)

    df_tft = prepare_tft_dataframe(df)
    max_encoder_length = 20
    max_prediction_length = 4
    batch_size = config.get("model", {}).get("batch_size", 32)
    logger.info("Encoder length: %s elections", max_encoder_length)
    logger.info("Prediction length: %s elections", max_prediction_length)

    training, train_loader, val_loader = build_datasets(
        df_tft=df_tft,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size,
    )
    best_tft = train_tft(
        training=training,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
    )
    predictions, tft_pred_median, ts_test, arima_forecast, train_end_idx = evaluate_tft_and_arima(
        best_tft=best_tft,
        val_dataloader=val_loader,
        ts=ts,
        df_tft=df_tft,
        max_prediction_length=max_prediction_length,
    )
    plot_forecast_comparison(
        ts=ts,
        ts_test=ts_test,
        predictions=predictions,
        tft_pred_median=tft_pred_median,
        arima_forecast=arima_forecast,
        train_end_idx=train_end_idx,
        images_dir=IMAGES_DIR,
    )
    save_model_and_dataset(
        best_tft=best_tft,
        training=training,
        val_dataloader=val_loader,
        output_dir=IMAGES_DIR,
    )
    logger.info("All outputs saved to %s", IMAGES_DIR)


if __name__ == "__main__":
    main()
