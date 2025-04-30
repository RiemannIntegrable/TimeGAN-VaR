import numpy as np
import pandas as pd

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
    for window in window_sizes:
        # Media móvil de precios
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Volatilidad histórica (desviación estándar de rendimientos)
        df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        
        # Rendimiento acumulado en la ventana
        df[f'Return_{window}'] = df['Returns'].rolling(window=window).sum()
    
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