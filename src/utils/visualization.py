import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

def plot_stock_prices(real_prices, synthetic_prices=None, dates=None, title='Precios de Acciones', future_dates=None):
    """
    Visualiza precios de acciones reales y sintéticos.
    
    Args:
        real_prices: Serie de precios reales
        synthetic_prices: Lista de series de precios sintéticos o None
        dates: Fechas correspondientes a los precios reales
        title: Título del gráfico
        future_dates: Fechas futuras para los precios sintéticos
        
    Returns:
        Figura de matplotlib
    """
    plt.figure(figsize=(12, 6))
    
    # Configurar eje x con fechas
    if dates is not None:
        if isinstance(dates, pd.DatetimeIndex):
            x_real = dates
        else:
            x_real = pd.to_datetime(dates)
    else:
        x_real = np.arange(len(real_prices))
    
    # Plotear precios reales
    plt.plot(x_real, real_prices, 'b-', linewidth=2, label='Datos Históricos')
    
    # Plotear precios sintéticos si se proporcionan
    if synthetic_prices is not None:
        if not isinstance(synthetic_prices, list):
            synthetic_prices = [synthetic_prices]
        
        if future_dates is not None:
            x_synth = future_dates
        else:
            # Si no hay fechas futuras, continuar desde el último dato real
            last_real_date = x_real[-1]
            if isinstance(last_real_date, pd.Timestamp):
                x_synth = pd.date_range(start=last_real_date + pd.Timedelta(days=1), 
                                        periods=len(synthetic_prices[0]))
            else:
                x_synth = np.arange(len(x_real), len(x_real) + len(synthetic_prices[0]))
        
        # Plotear múltiples trayectorias sintéticas
        for i, synth in enumerate(synthetic_prices):
            if i == 0:
                plt.plot(x_synth, synth, 'r-', alpha=0.8, linewidth=1.5, label='Datos Sintéticos')
            else:
                plt.plot(x_synth, synth, 'r-', alpha=0.3, linewidth=0.7)
    
    # Configurar el gráfico
    plt.title(title, fontsize=14)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Precio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Formatear eje x para fechas
    if isinstance(x_real[0], pd.Timestamp):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gcf().autofmt_xdate()
    
    return plt.gcf()

def plot_returns_distribution(real_returns, synthetic_returns, title='Distribución de Rendimientos'):
    """
    Compara las distribuciones de rendimientos reales y sintéticos.
    
    Args:
        real_returns: Array de rendimientos reales
        synthetic_returns: Array de rendimientos sintéticos
        title: Título del gráfico
        
    Returns:
        Figura de matplotlib
    """
    plt.figure(figsize=(12, 6))
    
    # Plotear histogramas
    sns.histplot(real_returns, color='blue', alpha=0.5, label='Rendimientos Reales', stat='density', kde=True)
    sns.histplot(synthetic_returns, color='red', alpha=0.5, label='Rendimientos Sintéticos', stat='density', kde=True)
    
    # Configurar el gráfico
    plt.title(title, fontsize=14)
    plt.xlabel('Rendimiento', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return plt.gcf()

def plot_acf_comparison(real_returns, synthetic_returns, lags=50, title='Comparación de Autocorrelación'):
    """
    Compara la función de autocorrelación entre datos reales y sintéticos.
    
    Args:
        real_returns: Array de rendimientos reales
        synthetic_returns: Array de rendimientos sintéticos
        lags: Número de rezagos a calcular
        title: Título del gráfico
        
    Returns:
        Figura de matplotlib
    """
    from statsmodels.graphics.tsaplots import plot_acf
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Autocorrelación de datos reales
    plot_acf(real_returns, lags=lags, alpha=0.05, title='ACF de Rendimientos Reales', ax=ax1)
    ax1.grid(True, alpha=0.3)
    
    # Autocorrelación de datos sintéticos
    plot_acf(synthetic_returns, lags=lags, alpha=0.05, title='ACF de Rendimientos Sintéticos', ax=ax2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=14, y=1.02)
    
    return fig

def plot_var_histogram(pnl_values, var_value, confidence_level=0.95, title='Distribución de P&L y VaR'):
    """
    Visualiza la distribución de P&L y el valor de VaR.
    
    Args:
        pnl_values: Array de valores de Pérdidas y Ganancias (P&L)
        var_value: Valor en Riesgo calculado
        confidence_level: Nivel de confianza del VaR
        title: Título del gráfico
        
    Returns:
        Figura de matplotlib
    """
    plt.figure(figsize=(12, 6))
    
    # Plotear histograma de P&L
    sns.histplot(pnl_values, color='blue', alpha=0.5, stat='density', kde=True)
    
    # Marcar el VaR
    plt.axvline(x=-var_value, color='red', linestyle='--', linewidth=2, 
                label=f'VaR ({confidence_level*100:.0f}%): {var_value:.2f}')
    
    # Sombrear la región por debajo del VaR
    x = np.linspace(min(pnl_values), -var_value, 1000)
    y = plt.gca().get_lines()[0].get_ydata()
    x_idx = np.searchsorted(plt.gca().get_lines()[0].get_xdata(), x)
    y_interp = np.interp(x, plt.gca().get_lines()[0].get_xdata(), y)
    plt.fill_between(x, y_interp, alpha=0.3, color='red')
    
    # Configurar el gráfico
    plt.title(title, fontsize=14)
    plt.xlabel('Pérdidas y Ganancias (P&L)', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return plt.gcf()

def plot_multiple_var_methods(var_values, methods, title='Comparación de Métodos de VaR'):
    """
    Compara diferentes métodos de cálculo de VaR.
    
    Args:
        var_values: Lista de valores de VaR calculados con diferentes métodos
        methods: Lista de nombres de los métodos
        title: Título del gráfico
        
    Returns:
        Figura de matplotlib
    """
    plt.figure(figsize=(10, 6))
    
    # Crear gráfico de barras
    bars = plt.bar(methods, var_values, color='skyblue', alpha=0.7)
    
    # Añadir valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=11)
    
    # Configurar el gráfico
    plt.title(title, fontsize=14)
    plt.xlabel('Método', fontsize=12)
    plt.ylabel('Valor en Riesgo (VaR)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    return plt.gcf()