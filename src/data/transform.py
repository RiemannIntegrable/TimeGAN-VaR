import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List, Union

def calculate_returns(df, method='log'):
    """
    Calcula los rendimientos a partir de precios.
    
    Args:
        df: DataFrame con columna 'Close'
        method: 'log' para rendimientos logarítmicos, 'simple' para rendimientos simples
        
    Returns:
        DataFrame con columna adicional de rendimientos
    """
    if method == 'log':
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    else:  # simple
        df['Returns'] = df['Close'].pct_change()
        
    # Eliminar la primera fila con NaN
    df = df.dropna()
    
    return df

def add_features(df, window_sizes=[5, 10, 20]):
    """
    Añade características técnicas como medias móviles y volatilidad.
    
    Args:
        df: DataFrame con columnas 'Close' y 'Returns'
        window_sizes: Lista de tamaños de ventana para calcular características
        
    Returns:
        DataFrame con características adicionales
    """
    # Crear una copia explícita al inicio de la función
    df = df.copy()
    
    for window in window_sizes:
        # Media móvil de precios
        df.loc[:, f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Volatilidad histórica (desviación estándar de rendimientos)
        df.loc[:, f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        
        # Rendimiento acumulado en la ventana
        df.loc[:, f'Return_{window}'] = df['Returns'].rolling(window=window).sum()
    
    # Eliminar filas con NaNs (primeras filas donde no hay suficientes datos para las ventanas)
    df = df.dropna()
    
    return df

def normalize_data(df, features=None, method='minmax'):
    """
    Normaliza las características seleccionadas.
    
    Args:
        df: DataFrame con datos
        features: Lista de columnas a normalizar, None para todas excepto 'Close'
        method: 'minmax' o 'zscore'
        
    Returns:
        DataFrame normalizado y parámetros de normalización para invertir posteriormente
    """
    if features is None:
        features = [col for col in df.columns if col != 'Close']
    
    norm_params = {}
    df_norm = df.copy()
    
    for feature in features:
        if method == 'minmax':
            min_val = df[feature].min()
            max_val = df[feature].max()
            df_norm[feature] = (df[feature] - min_val) / (max_val - min_val)
            norm_params[feature] = {'min': min_val, 'max': max_val}
        elif method == 'zscore':
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            df_norm[feature] = (df[feature] - mean_val) / std_val
            norm_params[feature] = {'mean': mean_val, 'std': std_val}
    
    return df_norm, norm_params

def inverse_normalize(df, norm_params, method='minmax'):
    """
    Invierte la normalización.
    
    Args:
        df: DataFrame normalizado
        norm_params: Parámetros de normalización
        method: 'minmax' o 'zscore'
        
    Returns:
        DataFrame con valores originales
    """
    df_orig = df.copy()
    
    for feature, params in norm_params.items():
        if feature in df.columns:
            if method == 'minmax':
                df_orig[feature] = df[feature] * (params['max'] - params['min']) + params['min']
            elif method == 'zscore':
                df_orig[feature] = df[feature] * params['std'] + params['mean']
    
    return df_orig

def build_multi_asset_portfolio(tickers, weights):
    """
    Construye un portafolio de múltiples activos descargando su historial máximo y aplicando pesos.
    
    Args:
        tickers: Tupla con los tickers de los activos (ej: ("^GSPC", "^STOXX"))
        weights: Tupla con los pesos de los activos (debe sumar 1)
        
    Returns:
        DataFrame con columnas 'Date' y 'Close' representando el portafolio
    """
    # Validación de argumentos
    n = len(tickers)
    if len(weights) != n:
        raise ValueError(f"La cantidad de tickers ({n}) debe coincidir con la cantidad de pesos ({len(weights)})")
    
    if abs(sum(weights) - 1.0) > 1e-10:
        raise ValueError(f"La suma de los pesos debe ser 1, pero es {sum(weights)}")
    
    # Descargar datos históricos para cada ticker
    print(f"Descargando datos históricos para {n} activos...")
    assets_data = []
    min_dates = []
    
    for i, ticker in enumerate(tickers):
        try:
            data = yf.Ticker(ticker).history(period="max")
            if data.empty:
                raise ValueError(f"No se pudieron obtener datos para el ticker {ticker}")
            
            data.reset_index(inplace=True)
            # Eliminar timezone para evitar problemas de alineación
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            
            min_dates.append(data['Date'].min())
            assets_data.append(data)
            print(f"  ✓ {ticker}: {len(data)} registros desde {data['Date'].min()} hasta {data['Date'].max()}")
            
        except Exception as e:
            raise ValueError(f"Error al descargar datos para {ticker}: {str(e)}")
    
    # Determinar la fecha de inicio común (la más reciente entre todas las fechas mínimas)
    common_start_date = max(min_dates)
    print(f"Fecha de inicio común: {common_start_date}")
    
    # Filtrar y alinear datos
    filtered_assets = []
    for i, data in enumerate(assets_data):
        # Filtrar por la fecha común
        filtered = data[data['Date'] >= common_start_date][['Date', 'Close']]
        filtered_assets.append(filtered.set_index('Date'))
        
        # Mostrar cuántos datos quedan después del filtrado
        print(f"  ✓ {tickers[i]}: {len(filtered)} registros después del filtrado")
    
    # Combinar todos los activos en un solo DataFrame
    combined = pd.concat(filtered_assets, axis=1, join='inner')
    
    if combined.empty:
        raise ValueError("No hay fechas comunes entre los activos después del filtrado")
    
    # Renombrar columnas para evitar duplicados
    combined.columns = [f'Close_{i+1}' for i in range(n)]
    
    # Calcular el portafolio ponderado
    portfolio_values = pd.Series(0.0, index=combined.index)
    for i in range(n):
        portfolio_values += combined[f'Close_{i+1}'] * weights[i]
    
    # Crear el DataFrame final
    result = pd.DataFrame({
        'Close': portfolio_values
    })
    result.reset_index(inplace=True)
    
    print(f"Portafolio creado exitosamente con {len(result)} observaciones")
    print(f"Período del portafolio: {result['Date'].min()} a {result['Date'].max()}")
    
    return result