"""
Minimalist Time Series Plotting Utilities

A reusable module for creating clean, minimalist time series visualizations.
Designed for publication-quality figures with trendline labels at the end of lines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression


def setup_minimalist_style():
    """Configure matplotlib for minimalist plotting style."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
                'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    })


def plot_time_series_with_groups(
    df, 
    time_col, 
    value_col, 
    group_col=None,
    group_labels=None,
    colors=None,
    linestyles=None,
    title=None,
    xlabel=None,
    ylabel=None,
    figsize=(12, 6),
    save_path=None,
    dpi=300
):
    """
    Create a minimalist time series plot with multiple groups (e.g., presidential vs midterm).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    time_col : str
        Column name for time/x-axis values
    value_col : str
        Column name for y-axis values
    group_col : str, optional
        Column name to group by (e.g., 'Is_Presidential'). If None, plots single line
    group_labels : dict, optional
        Mapping of group values to display labels (e.g., {True: 'Presidential', False: 'Midterm'})
    colors : dict or list, optional
        Colors for each group. If dict, maps group values to colors. If list, uses in order.
    linestyles : dict or list, optional
        Line styles for each group. If dict, maps group values to styles. If list, uses in order.
    title : str, optional
        Plot title. If None, uses f"{value_col} over Time"
    xlabel : str, optional
        X-axis label. If None, uses time_col
    ylabel : str, optional
        Y-axis label. If None, uses value_col
    figsize : tuple, default (12, 6)
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the figure. If None, figure is not saved
    dpi : int, default 300
        Resolution for saved figure
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    setup_minimalist_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors and styles
    default_colors = ['black', 'gray', '#666666', '#999999']
    default_linestyles = ['-', '--', '-.', ':']
    
    if group_col is None:
        # Single line plot
        ax.plot(df[time_col], df[value_col], 
                linewidth=1.5, color=default_colors[0], linestyle=default_linestyles[0])
    else:
        # Multiple groups
        groups = df[group_col].unique()
        
        for i, group_val in enumerate(groups):
            group_data = df[df[group_col] == group_val].sort_values(time_col)
            
            # Get color and linestyle
            if colors is None:
                color = default_colors[i % len(default_colors)]
            elif isinstance(colors, dict):
                color = colors.get(group_val, default_colors[i % len(default_colors)])
            else:
                color = colors[i % len(colors)]
            
            if linestyles is None:
                linestyle = default_linestyles[i % len(default_linestyles)]
            elif isinstance(linestyles, dict):
                linestyle = linestyles.get(group_val, default_linestyles[i % len(default_linestyles)])
            else:
                linestyle = linestyles[i % len(linestyles)]
            
            # Get label
            if group_labels is None:
                label = str(group_val)
            else:
                label = group_labels.get(group_val, str(group_val))
            
            ax.plot(group_data[time_col], group_data[value_col],
                    linewidth=1.5, color=color, linestyle=linestyle, label=label)
    
    # Set labels and title
    ax.set_xlabel(xlabel or time_col, fontsize=12)
    ax.set_ylabel(ylabel or value_col, fontsize=12)
    ax.set_title(title or f"{value_col} over Time", fontsize=13, pad=10)
    
    # Add legend if groups are present
    if group_col is not None:
        ax.legend(loc='upper right', frameon=False, fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to '{save_path}'")
    
    return fig, ax


def plot_trend_with_label(
    df,
    time_col,
    trend_values,
    label_text='Trend',
    title='Long-term Trend',
    xlabel=None,
    ylabel=None,
    figsize=(12, 6),
    save_path=None,
    dpi=300
):
    """
    Plot a trend line with label at the end of the line.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time data
    time_col : str
        Column name for time/x-axis values
    trend_values : array-like
        Trend values to plot (same length as df)
    label_text : str, default 'Trend'
        Text to display at the end of the trendline
    title : str, default 'Long-term Trend'
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : tuple, default (12, 6)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
    dpi : int, default 300
        Resolution for saved figure
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    setup_minimalist_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[time_col], trend_values, linewidth=1.5, color='black', linestyle='-')
    
    # Add label at the end of the trendline
    last_time = df[time_col].iloc[-1]
    # Handle both array and Series
    if hasattr(trend_values, 'iloc'):
        last_trend_value = trend_values.iloc[-1]
    else:
        last_trend_value = trend_values[-1]
    ax.text(last_time, last_trend_value, f' {label_text}', 
            fontsize=11, verticalalignment='center', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    ax.set_xlabel(xlabel or time_col, fontsize=12)
    ax.set_ylabel(ylabel or 'Value', fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to '{save_path}'")
    
    return fig, ax


def plot_detrended_with_groups(
    df,
    time_col,
    detrended_values,
    group_col,
    group_labels=None,
    colors=None,
    linestyles=None,
    title='Detrended Data',
    xlabel=None,
    ylabel='Deviation from Trend',
    figsize=(12, 6),
    save_path=None,
    dpi=300
):
    """
    Plot detrended data with groups, showing deviation from trend.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    time_col : str
        Column name for time/x-axis values
    detrended_values : array-like
        Detrended values (same length as df)
    group_col : str
        Column name to group by
    group_labels : dict, optional
        Mapping of group values to display labels
    colors : dict or list, optional
        Colors for each group
    linestyles : dict or list, optional
        Line styles for each group
    title : str, default 'Detrended Data'
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, default 'Deviation from Trend'
        Y-axis label
    figsize : tuple, default (12, 6)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
    dpi : int, default 300
        Resolution for saved figure
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    setup_minimalist_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    default_colors = ['black', 'gray', '#666666', '#999999']
    default_linestyles = ['-', '--', '-.', ':']
    
    groups = df[group_col].unique()
    
    for i, group_val in enumerate(groups):
        group_mask = df[group_col] == group_val
        group_data = df[group_mask].sort_values(time_col)
        group_detrended = np.array(detrended_values)[group_mask]
        
        # Get color and linestyle
        if colors is None:
            color = default_colors[i % len(default_colors)]
        elif isinstance(colors, dict):
            color = colors.get(group_val, default_colors[i % len(default_colors)])
        else:
            color = colors[i % len(colors)]
        
        if linestyles is None:
            linestyle = default_linestyles[i % len(default_linestyles)]
        elif isinstance(linestyles, dict):
            linestyle = linestyles.get(group_val, default_linestyles[i % len(default_linestyles)])
        else:
            linestyle = linestyles[i % len(linestyles)]
        
        # Get label
        if group_labels is None:
            label = str(group_val)
        else:
            label = group_labels.get(group_val, str(group_val))
        
        ax.plot(group_data[time_col], group_detrended,
                linewidth=1.5, color=color, linestyle=linestyle, label=label)
    
    # Add zero reference line
    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel(xlabel or time_col, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to '{save_path}'")
    
    return fig, ax


def plot_forecast_with_history(
    historical_df,
    time_col,
    value_col,
    future_times,
    forecast_values,
    lower_bound=None,
    upper_bound=None,
    group_col=None,
    group_labels=None,
    title='Forecast',
    xlabel=None,
    ylabel=None,
    forecast_label='Forecast',
    figsize=(12, 6),
    save_path=None,
    dpi=300
):
    """
    Plot historical data with forecast, showing confidence intervals.
    
    Parameters
    ----------
    historical_df : pd.DataFrame
        Historical data
    time_col : str
        Column name for time values
    value_col : str
        Column name for values
    future_times : array-like
        Future time points for forecast
    forecast_values : array-like
        Forecasted values
    lower_bound : array-like, optional
        Lower confidence bound
    upper_bound : array-like, optional
        Upper confidence bound
    group_col : str, optional
        Column to group historical data by
    group_labels : dict, optional
        Labels for groups
    title : str, default 'Forecast'
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    forecast_label : str, default 'Forecast'
        Label text to place at end of forecast line
    figsize : tuple, default (12, 6)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
    dpi : int, default 300
        Resolution for saved figure
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    setup_minimalist_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical data
    if group_col is None:
        ax.plot(historical_df[time_col], historical_df[value_col],
                linewidth=1.5, color='black', linestyle='-', label='Historical')
    else:
        groups = historical_df[group_col].unique()
        default_colors = ['black', 'gray']
        default_linestyles = ['-', '--']
        
        for i, group_val in enumerate(groups):
            group_data = historical_df[historical_df[group_col] == group_val].sort_values(time_col)
            label = group_labels.get(group_val, str(group_val)) if group_labels else str(group_val)
            ax.plot(group_data[time_col], group_data[value_col],
                    linewidth=1.5, color=default_colors[i % len(default_colors)],
                    linestyle=default_linestyles[i % len(default_linestyles)],
                    label=label)
    
    # Plot forecast
    ax.plot(future_times, forecast_values, linewidth=1.5, color='black', linestyle=':')
    
    # Add confidence interval if provided
    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(future_times, lower_bound, upper_bound,
                       alpha=0.2, color='gray', label='95% CI')
    
    # Add label at end of forecast line
    # Handle both array and Series
    if hasattr(future_times, 'iloc'):
        last_forecast_time = future_times.iloc[-1]
    else:
        last_forecast_time = future_times[-1]
    
    if hasattr(forecast_values, 'iloc'):
        last_forecast_value = forecast_values.iloc[-1]
    else:
        last_forecast_value = forecast_values[-1]
    ax.text(last_forecast_time, last_forecast_value, f' {forecast_label}', 
            fontsize=11, verticalalignment='center', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    # Vertical line at last data point
    last_historical_time = historical_df[time_col].max()
    ax.axvline(x=last_historical_time, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel(xlabel or time_col, fontsize=12)
    ax.set_ylabel(ylabel or value_col, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to '{save_path}'")
    
    return fig, ax


def plot_statistical_decomposition(
    df,
    time_col,
    value_col,
    period=2,
    model='additive',
    title=None,
    figsize=(12, 10),
    save_path=None,
    dpi=300
):
    """
    Plot statistical decomposition (trend, seasonal, residual).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data
    time_col : str
        Column name for time values
    value_col : str
        Column name for values
    period : int, default 2
        Period for seasonal decomposition
    model : str, default 'additive'
        Decomposition model ('additive' or 'multiplicative')
    title : str, optional
        Plot title
    figsize : tuple, default (12, 10)
        Figure size
    save_path : str or Path, optional
        Path to save the figure
    dpi : int, default 300
        Resolution for saved figure
    
    Returns
    -------
    fig : matplotlib figure object
    decomposition : statsmodels decomposition object or None
    """
    setup_minimalist_style()
    
    df_sorted = df.sort_values(time_col).copy()
    
    # Create full time series index
    time_min = df_sorted[time_col].min()
    time_max = df_sorted[time_col].max()
    
    # For election data, assume 2-year intervals
    all_times = np.arange(time_min, time_max + 1, 2)
    
    # Create indexed series
    ts_indexed = df_sorted.set_index(time_col)[value_col]
    ts_full = pd.Series(index=all_times, dtype=float)
    
    for time_val in all_times:
        if time_val in ts_indexed.index:
            ts_full[time_val] = ts_indexed[time_val]
    
    # Interpolate missing values
    ts_full = ts_full.interpolate(method='linear')
    ts_full.index = pd.to_datetime(ts_full.index.astype(str))
    
    try:
        decomposition = seasonal_decompose(
            ts_full,
            model=model,
            period=period,
            extrapolate_trend='freq'
        )
        
        # Create minimalist decomposition plot
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        axes[0].plot(ts_full.index, ts_full.values, linewidth=1.5, color='black')
        axes[0].set_ylabel('Original', fontsize=11)
        
        year_range_str = f"{time_min}-{time_max}"
        axes[0].set_title(title or f'Statistical Decomposition ({year_range_str})', 
                         fontsize=13, pad=10)
        
        axes[1].plot(decomposition.trend.index, decomposition.trend.values, 
                    linewidth=1.5, color='black')
        axes[1].set_ylabel('Trend', fontsize=11)
        
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 
                    linewidth=1.5, color='black')
        axes[2].set_ylabel('Seasonal', fontsize=11)
        
        axes[3].plot(decomposition.resid.index, decomposition.resid.values, 
                    linewidth=1.5, color='black')
        axes[3].set_ylabel('Residual', fontsize=11)
        axes[3].set_xlabel('Year', fontsize=12)
        
        # Remove top and right spines
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved decomposition to '{save_path}'")
        
        return fig, decomposition
        
    except Exception as e:
        print(f"Statistical decomposition failed: {e}")
        return None, None


# Example usage function
def example_usage():
    """Example of how to use the plotting utilities."""
    # Create sample data
    years = np.arange(2000, 2023, 2)
    np.random.seed(42)
    values = 50 + 0.5 * (years - 2000) + np.random.randn(len(years)) * 5
    
    df = pd.DataFrame({
        'Year': years,
        'Value': values,
        'Group': ['A' if i % 2 == 0 else 'B' for i in range(len(years))]
    })
    
    # Example 1: Simple time series with groups
    plot_time_series_with_groups(
        df, 
        time_col='Year',
        value_col='Value',
        group_col='Group',
        group_labels={'A': 'Group A', 'B': 'Group B'},
        title='Example Time Series',
        save_path='example_time_series.png'
    )
    plt.close()
    
    # Example 2: Trend with label
    trend = 50 + 0.5 * (df['Year'] - 2000)
    plot_trend_with_label(
        df,
        time_col='Year',
        trend_values=trend,
        label_text='Trend',
        save_path='example_trend.png'
    )
    plt.close()
    
    print("Example plots created!")


if __name__ == "__main__":
    example_usage()

