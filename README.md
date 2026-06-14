# Voter Turnout Forecasting with Temporal Fusion Transformers

Companion code for a Medium article on multi-horizon voter turnout forecasting. This repo compares **Temporal Fusion Transformers (TFT)** against **ARIMA** on historical US election turnout data.

## What this repo contains

| File | Purpose |
|------|---------|
| `train_tft.py` | Train TFT, compare against ARIMA, save forecast plots |
| `analyze_voter_turnout.py` | Exploratory turnout analysis (trends, decomposition, linear forecast) |
| `minimalist_plotting.py` | Reusable minimalist plotting helpers |
| `article.md` | Article draft / source notes |
| `data/us_voter_turnout.csv` | Historical turnout by election year (1789–2022) |

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<your-username>/voter-turnout-forecasting-with-temporal-fusion-transformers.git
cd voter-turnout-forecasting-with-temporal-fusion-transformers
uv sync
uv run pytest
```

### Train TFT and compare with ARIMA

```bash
uv run python train_tft.py
```

Outputs are written to `images/` (gitignored): `voter_turnout_series.png`, `tft_vs_arima.png`, and model artifacts.

### Exploratory turnout analysis

```bash
uv run python analyze_voter_turnout.py
```

Outputs are written to `images_full/`, `images_modern/`, and `analysis_summary.txt`.

## Data

The dataset covers US presidential and midterm elections from 1789 through 2022. Turnout rates are expressed as a percentage of the voting-eligible population.

## Configuration

Hyperparameters and plot settings live in `config.yaml`.

## Development

```bash
uv run pytest
uv run ruff check .
pre-commit install   # optional: gitleaks secret scanning
```

## Disclaimer

Educational and demonstration code only. Not financial, safety, or engineering advice. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
