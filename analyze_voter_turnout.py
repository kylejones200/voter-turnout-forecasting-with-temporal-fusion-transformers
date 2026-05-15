"""
Complete Additive Time Series Example: Voter Turnout Analysis
Using actual US voter turnout data from 1789-2022
"""
import signalplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pathlib import Path

import logging
import yaml

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        return _yaml.safe_load(_f) or {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Set up paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "2025-11-12_us_voter_turnout.csv"
IMAGES_DIR_FULL = BASE_DIR / "images_full"
IMAGES_DIR_MODERN = BASE_DIR / "images_modern"

# Create images directories if they don't exist
IMAGES_DIR_FULL.mkdir(exist_ok=True)
IMAGES_DIR_MODERN.mkdir(exist_ok=True)

# Set minimalist style
plt.style.use('default')
signalplot.apply(font_family='serif')


def load_voter_turnout_data(data_path, start_year=1789):
    """Load actual US voter turnout data."""
    df = pd.read_csv(data_path)
    
    # Clean column names (remove duplicates)
    df.columns = ['Year', 'Turnout_Rate', 'Election_Type']
    
    # Convert turnout rate to numeric
    df['Turnout_Rate'] = pd.to_numeric(df['Turnout_Rate'], errors='coerce')
    
    # Create boolean for presidential elections
    df['Is_Presidential'] = df['Election_Type'] == 'Presidential'
    
    # Sort by year
    df = df.sort_values('Year').reset_index(drop=True)
    
    # Remove any rows with missing data
    df = df.dropna()
    
    # Filter to start year and later
    df = df[df['Year'] >= start_year].reset_index(drop=True)
    
    return df


def analyze_turnout_patterns(df):
    """Analyze presidential vs midterm election patterns."""
    presidential_elections = df[df['Is_Presidential']]
    midterm_elections = df[~df['Is_Presidential']]
    
    logger.info("=== Turnout Analysis - Actual US Data ===")
    logger.info(f"Total Elections: {len(df)}")
    logger.info(f"  Presidential: {len(presidential_elections)}")
    logger.info(f"  Midterm: {len(midterm_elections)}")
    logger.info(f"\nPresidential Elections:")
    logger.info(f"  Mean: {presidential_elections['Turnout_Rate'].mean():.1f}%")
    logger.info(f"  Std: {presidential_elections['Turnout_Rate'].std():.1f}%")
    logger.info(f"  Min: {presidential_elections['Turnout_Rate'].min():.1f}%")
    logger.info(f"  Max: {presidential_elections['Turnout_Rate'].max():.1f}%")
    logger.info(f"\nMidterm Elections:")
    logger.info(f"  Mean: {midterm_elections['Turnout_Rate'].mean():.1f}%")
    logger.info(f"  Std: {midterm_elections['Turnout_Rate'].std():.1f}%")
    logger.info(f"  Min: {midterm_elections['Turnout_Rate'].min():.1f}%")
    logger.info(f"  Max: {midterm_elections['Turnout_Rate'].max():.1f}%")
    
    presidential_bump = (presidential_elections['Turnout_Rate'].mean() - 
                         midterm_elections['Turnout_Rate'].mean())
    logger.info(f"\nAverage Presidential Bump: {presidential_bump:.1f}%")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        presidential_elections['Turnout_Rate'],
        midterm_elections['Turnout_Rate']
    )
    logger.info(f"\nT-test (presidential vs midterm):")
    logger.info(f"  t-statistic: {t_stat:.3f}")
    logger.info(f"  p-value: {p_value:.6f}")
    logger.info(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    return presidential_elections, midterm_elections, presidential_bump


def estimate_trend(df):
    """Estimate long-term trend using linear regression."""
    # Fit linear trend
    X = df[['Year']].values
    y = df['Turnout_Rate'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict trend
    trend = model.predict(X)
    
    # Calculate statistics
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    
    logger.info(f"\nTrend Analysis:")
    logger.info(f"  Slope: {slope:.4f}% per year")
    logger.info(f"  Intercept: {intercept:.2f}%")
    logger.info(f"  R-squared: {r_squared:.4f}")
    
    if slope > 0:
        logger.info(f"  Interpretation: Turnout increasing by {slope:.4f}% per year on average")
    else:
        logger.info(f"  Interpretation: Turnout decreasing by {abs(slope):.4f}% per year on average")
    
    return trend, slope, intercept, r_squared


def calculate_presidential_effect(df, trend):
    """Calculate the average presidential election effect."""
    # Remove trend to isolate seasonal effect
    detrended = df['Turnout_Rate'].values - trend
    
    presidential_detrended = detrended[df['Is_Presidential'].values]
    midterm_detrended = detrended[~df['Is_Presidential'].values]
    
    avg_presidential_effect = np.mean(presidential_detrended) - np.mean(midterm_detrended)
    
    logger.info(f"\nPresidential Effect (after detrending):")
    logger.info(f"  Average effect: {avg_presidential_effect:.2f}%")
    
    return avg_presidential_effect


def visualize_time_series(df, trend, images_dir, year_range_str=None, plot: bool = False):
    """Create minimalist time series visualizations."""
    pres_data = df[df['Is_Presidential']].sort_values('Year')
    midterm_data = df[~df['Is_Presidential']].sort_values('Year')
    
    # Generate title with year range
    if year_range_str is None:
        year_range_str = f"{df['Year'].min()}-{df['Year'].max()}"
    
    # Main time series plot: Presidential vs Midterm
    if plot:
        fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [12, 6])))
    
    # Plot lines for presidential and midterm elections
        ax.plot(pres_data['Year'], pres_data['Turnout_Rate'], 
                linewidth=1.5, color='black', label='Presidential', linestyle='-')
        ax.plot(midterm_data['Year'], midterm_data['Turnout_Rate'], 
                linewidth=1.5, color='gray', label='Midterm', linestyle='--')
    
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Turnout Rate (%)', fontsize=12)
        ax.set_title(f'US Voter Turnout ({year_range_str})', fontsize=13, pad=10)
        ax.legend(loc='upper right', frameon=False, fontsize=11)
    
        plt.tight_layout()
        plt.savefig(images_dir / 'voter_turnout_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved main time series to '{images_dir / 'voter_turnout_time_series.png'}'")
    
    # Trend plot
        fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [12, 6])))
        ax.plot(df['Year'], trend, linewidth=1.5, color='black', linestyle='-')
    
    # Add label at the end of the trendline
        last_year = df['Year'].iloc[-1]
        last_trend_value = trend[-1]
        ax.text(last_year, last_trend_value, ' Trend', 
                fontsize=11, verticalalignment='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Turnout Rate (%)', fontsize=12)
        ax.set_title('Long-term Trend', fontsize=13, pad=10)
        plt.tight_layout()
        plt.savefig(images_dir / 'voter_turnout_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Detrended plot
        detrended = df['Turnout_Rate'].values - trend
        pres_detrended = detrended[df['Is_Presidential'].values]
        midterm_detrended = detrended[~df['Is_Presidential'].values]
    
        fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [12, 6])))
        ax.plot(pres_data['Year'], pres_detrended, 
                linewidth=1.5, color='black', label='Presidential', linestyle='-')
        ax.plot(midterm_data['Year'], midterm_detrended, 
                linewidth=1.5, color='gray', label='Midterm', linestyle='--')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Deviation from Trend (%)', fontsize=12)
        ax.set_title('Detrended Data (Presidential Effect)', fontsize=13, pad=10)
        ax.legend(loc='upper right', frameon=False, fontsize=11)
        plt.tight_layout()
        plt.savefig(images_dir / 'voter_turnout_detrended.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved all visualizations to '{images_dir}'")


def perform_statistical_decomposition(df, images_dir, plot: bool = False):
    """Perform statistical decomposition using statsmodels with minimalist styling."""
    df_sorted = df.sort_values('Year').copy()
    all_years = np.arange(df_sorted['Year'].min(), df_sorted['Year'].max() + 1, 2)
    
    ts_indexed = df_sorted.set_index('Year')['Turnout_Rate']
    ts_full = pd.Series(index=all_years, dtype=float)
    for year in all_years:
        if year in ts_indexed.index:
            ts_full[year] = ts_indexed[year]
    
    ts_full = ts_full.interpolate(method='linear')
    ts_full.index = pd.to_datetime(ts_full.index.astype(str))
    
    try:
        decomposition = seasonal_decompose(
            ts_full,
            model='additive',
            period=2,
            extrapolate_trend='freq'
        )
        
        # Create minimalist decomposition plot
        if plot:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
            axes[0].plot(ts_full.index, ts_full.values, linewidth=1.5, color='black')
            axes[0].set_ylabel('Original', fontsize=11)
            year_range_str = f"{df_sorted['Year'].min()}-{df_sorted['Year'].max()}"
            axes[0].set_title(f'Statistical Decomposition of Voter Turnout ({year_range_str})', fontsize=13, pad=10)
        
            axes[1].plot(decomposition.trend.index, decomposition.trend.values, 
                        linewidth=1.5, color='black')
            axes[1].set_ylabel('Trend', fontsize=11)
        
            axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 
                        linewidth=1.5, color='black')
            axes[2].set_ylabel('Seasonal', fontsize=11)
        
            axes[3].plot(decomposition.resid.index, decomposition.resid.values, 
                        linewidth=1.5, color='black')
            axes[3].axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
            axes[3].set_ylabel('Residual', fontsize=11)
            axes[3].set_xlabel('Year', fontsize=12)
        
            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
            plt.tight_layout()
            plt.savefig(images_dir / 'statistical_decomposition.png', dpi=300, bbox_inches='tight')
            plt.close()
        logger.info(f"Saved statistical decomposition to '{images_dir / 'statistical_decomposition.png'}'")
        
        return decomposition
    except Exception as e:
        logger.error(f"Statistical decomposition failed: {e}", exc_info=True)
        return None


def forecast_turnout(df, trend_model, n_years_ahead=10, images_dir=None, plot: bool = False):
    """Forecast future turnout using linear trend with minimalist styling."""
    last_year = df['Year'].max()
    future_years = np.arange(last_year + 2, last_year + 2 + n_years_ahead * 2, 2)
    
    X_future = future_years.reshape(-1, 1)
    y_forecast = trend_model.predict(X_future)
    
    X_train = df[['Year']].values
    y_train = df['Turnout_Rate'].values
    y_pred_train = trend_model.predict(X_train)
    residuals = y_train - y_pred_train
    std_error = np.std(residuals)
    
    upper_bound = y_forecast + 1.96 * std_error
    lower_bound = y_forecast - 1.96 * std_error
    
    # Minimalist forecast visualization
    if plot:
        fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [12, 6])))
    
    # Historical data - separate lines for presidential and midterm
        pres_data = df[df['Is_Presidential']].sort_values('Year')
        midterm_data = df[~df['Is_Presidential']].sort_values('Year')
    
        ax.plot(pres_data['Year'], pres_data['Turnout_Rate'], 
                linewidth=1.5, color='black', label='Presidential (Historical)', linestyle='-')
        ax.plot(midterm_data['Year'], midterm_data['Turnout_Rate'], 
                linewidth=1.5, color='gray', label='Midterm (Historical)', linestyle='--')
    
    # Forecast
        ax.plot(future_years, y_forecast, linewidth=1.5, color='black', 
                linestyle=':')
        ax.fill_between(future_years, lower_bound, upper_bound, 
                        alpha=0.2, color='gray', label='95% CI')
    
    # Add label at the end of the forecast trendline
        last_forecast_year = future_years[-1]
        last_forecast_value = y_forecast[-1]
        ax.text(last_forecast_year, last_forecast_value, ' Forecast', 
                fontsize=11, verticalalignment='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    # Vertical line at last data point
        ax.axvline(x=last_year, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Turnout Rate (%)', fontsize=12)
        ax.set_title('Voter Turnout Forecast', fontsize=13, pad=10)
        ax.legend(loc='upper right', frameon=False, fontsize=11)
    
        plt.tight_layout()
        if images_dir:
            plt.savefig(images_dir / 'turnout_forecast.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved forecast to '{images_dir / 'turnout_forecast.png'}'")
        plt.close()
    
    return future_years, y_forecast, lower_bound, upper_bound


def create_summary_statistics(df, pres_elections, midterm_elections, 
                              presidential_bump, trend_slope, r_squared):
    """Create a summary of key findings."""
    summary = {
        'total_elections': len(df),
        'presidential_count': len(pres_elections),
        'midterm_count': len(midterm_elections),
        'presidential_mean': pres_elections['Turnout_Rate'].mean(),
        'presidential_std': pres_elections['Turnout_Rate'].std(),
        'midterm_mean': midterm_elections['Turnout_Rate'].mean(),
        'midterm_std': midterm_elections['Turnout_Rate'].std(),
        'presidential_bump': presidential_bump,
        'trend_slope': trend_slope,
        'r_squared': r_squared,
        'year_range': (df['Year'].min(), df['Year'].max()),
        'highest_turnout': df.loc[df['Turnout_Rate'].idxmax(), ['Year', 'Turnout_Rate', 'Election_Type']].to_dict(),
        'lowest_turnout': df.loc[df['Turnout_Rate'].idxmin(), ['Year', 'Turnout_Rate', 'Election_Type']].to_dict(),
    }
    
    return summary


def run_analysis(start_year, images_dir, period_name):
    """Run analysis for a specific time period."""
    logger.info("=== Analysis: {period_name} ===")
    
    # Load data
    logger.info(f"\n1. Loading voter turnout data ({start_year}+)...")
    df = load_voter_turnout_data(DATA_PATH, start_year=start_year)
    logger.info(f"   Loaded {len(df)} election observations")
    logger.info(f"   Years: {df['Year'].min()} to {df['Year'].max()}")
    
    # Analyze patterns
    logger.info("\n2. Analyzing turnout patterns...")
    pres_elections, midterm_elections, presidential_bump = analyze_turnout_patterns(df)
    
    # Estimate trend
    logger.info("\n3. Estimating long-term trend...")
    trend, slope, intercept, r_squared = estimate_trend(df)
    df['Trend'] = trend
    
    # Calculate presidential effect
    logger.info("\n4. Calculating presidential election effect...")
    avg_presidential_effect = calculate_presidential_effect(df, trend)
    
    # Create trend model for forecasting
    trend_model = LinearRegression()
    trend_model.fit(df[['Year']].values, df['Turnout_Rate'].values)
    
    # Visualize
    logger.info("\n5. Creating visualizations...")
    year_range_str = f"{df['Year'].min()}-{df['Year'].max()}"
    visualize_time_series(df, trend, images_dir, year_range_str)
    
    # Statistical decomposition
    logger.info("\n6. Performing statistical decomposition...")
    decomposition = perform_statistical_decomposition(df, images_dir)
    
    # Forecast
    logger.info("\n7. Forecasting future turnout...")
    future_years, forecast, lower_bound, upper_bound = forecast_turnout(
        df, trend_model, n_years_ahead=10, images_dir=images_dir
    )
    logger.info(f"   Forecasted turnout for {future_years[-1]}: {forecast[-1]:.1f}%")
    logger.info(f"   (95% CI: {lower_bound[-1]:.1f}% - {upper_bound[-1]:.1f}%)")
    
    return {
        'df': df,
        'pres_elections': pres_elections,
        'midterm_elections': midterm_elections,
        'presidential_bump': presidential_bump,
        'slope': slope,
        'r_squared': r_squared
    }


def main():
    """Run complete analysis for both time periods."""
    logger.info("Additive Time Series: Voter Turnout Analysis")
    logger.info("Generating images for both time periods")
    
    # Run analysis for full period (1789-2022)
    results_full = run_analysis(
        start_year=1789,
        images_dir=IMAGES_DIR_FULL,
        period_name="Full Period (1789-2022)"
    )
    
    # Run analysis for modern era (1945-2022)
    results_modern = run_analysis(
        start_year=1945,
        images_dir=IMAGES_DIR_MODERN,
        period_name="Modern Era (1945-2022)"
    )
    
    # Create summary for modern era (as requested)
    logger.info("\n8. Generating summary statistics...")
    summary = create_summary_statistics(
        results_modern['df'],
        results_modern['pres_elections'],
        results_modern['midterm_elections'],
        results_modern['presidential_bump'],
        results_modern['slope'],
        results_modern['r_squared']
    )
    
    # Save summary to file
    summary_path = BASE_DIR / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Voter Turnout Analysis Summary (Modern Era)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data Range: {summary['year_range'][0]} - {summary['year_range'][1]}\n")
        f.write(f"Total Elections: {summary['total_elections']}\n")
        f.write(f"  Presidential: {summary['presidential_count']}\n")
        f.write(f"  Midterm: {summary['midterm_count']}\n\n")
        f.write(f"Presidential Elections:\n")
        f.write(f"  Mean: {summary['presidential_mean']:.1f}%\n")
        f.write(f"  Std: {summary['presidential_std']:.1f}%\n\n")
        f.write(f"Midterm Elections:\n")
        f.write(f"  Mean: {summary['midterm_mean']:.1f}%\n")
        f.write(f"  Std: {summary['midterm_std']:.1f}%\n\n")
        f.write(f"Presidential Bump: {summary['presidential_bump']:.1f}%\n\n")
        f.write(f"Long-term Trend:\n")
        f.write(f"  Slope: {summary['trend_slope']:.4f}% per year\n")
        f.write(f"  R-squared: {summary['r_squared']:.4f}\n\n")
        f.write(f"Highest Turnout: {summary['highest_turnout']['Year']} "
               f"({summary['highest_turnout']['Turnout_Rate']:.1f}%, "
               f"{summary['highest_turnout']['Election_Type']})\n")
        f.write(f"Lowest Turnout: {summary['lowest_turnout']['Year']} "
               f"({summary['lowest_turnout']['Turnout_Rate']:.1f}%, "
               f"{summary['lowest_turnout']['Election_Type']})\n")
    
    logger.info(f"   Saved summary to '{summary_path}'")
    
    logger.info("=== Analysis completed successfully! ===")
    logger.info(f"\nAll outputs saved to: {BASE_DIR}")
    logger.info(f"  - Full period images: {IMAGES_DIR_FULL}")
    logger.info(f"  - Modern era images: {IMAGES_DIR_MODERN}")
    logger.info(f"  - Summary: {summary_path}")
    
    return summary


if __name__ == "__main__":
    summary = main()

