import numpy as np
import pandas as pd

def create_windows(data, window_size, stride=1, target_column=None):
    """
    Crea ventanas temporales para entrenamiento de TimeGAN.
    
    Args:
        data: Array de numpy o DataFrame
        window_size: Tamaño de la ventana temporal
        stride: Salto entre ventanas consecutivas
        target_column: Columna objetivo si data es DataFrame, None usa todas las columnas
        
    Returns:
        Array de numpy con ventanas de forma (n_windows, window_size, n_features)
    """
    if isinstance(data, pd.DataFrame):
        if target_column is not None:
            data = data[target_column].values
        else:
            data = data.values
    
    # Para series univariadas, asegurar que tengan forma (n_samples, 1)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // stride + 1
    
    windows = np.zeros((n_windows, window_size, n_features))
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]
    
    return windows

def recreate_time_series(generated_windows, original_data=None, stride=1, overlap_method='average'):
    """
    Recrea una serie temporal a partir de ventanas generadas.
    
    Args:
        generated_windows: Ventanas generadas por TimeGAN
        original_data: Datos originales para continuar la serie (opcional)
        stride: Salto utilizado en la creación de las ventanas originales
        overlap_method: Método para manejar superposiciones ('average', 'last')
        
    Returns:
        Serie temporal reconstruida
    """
    n_windows, window_size, n_features = generated_windows.shape
    
    # Calcular el tamaño de la serie temporal reconstruida
    if stride == 1:
        n_total = window_size + (n_windows - 1)
    else:
        n_total = window_size + (n_windows - 1) * stride
    
    # Inicializar arrays para acumular valores y contadores
    reconstructed = np.zeros((n_total, n_features))
    counts = np.zeros(n_total)
    
    # Reconstruir la serie con superposiciones
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        if overlap_method == 'average':
            # Acumular valores y contar ocurrencias para promediar después
            reconstructed[start_idx:end_idx] += generated_windows[i]
            counts[start_idx:end_idx] += 1
        elif overlap_method == 'last':
            # Simplemente sobrescribir con los valores más recientes
            reconstructed[start_idx:end_idx] = generated_windows[i]
            counts[start_idx:end_idx] = 1
    
    # Promediar valores en caso de superposiciones
    if overlap_method == 'average':
        reconstructed = reconstructed / counts.reshape(-1, 1)
    
    # Si se proporcionaron datos originales, concatenarlos con los generados
    if original_data is not None:
        if isinstance(original_data, pd.DataFrame):
            original_data = original_data.values
        
        if len(original_data.shape) == 1:
            original_data = original_data.reshape(-1, 1)
        
        # Concatenar los últimos valores originales con los generados
        last_original = original_data[-1:]
        reconstructed = np.vstack([last_original, reconstructed])
    
    return reconstructed