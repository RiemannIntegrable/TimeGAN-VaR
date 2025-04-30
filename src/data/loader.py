import pandas as pd
import numpy as np
import os

def load_stock_data(file_path):
    """
    Carga datos de acciones desde un archivo CSV.
    
    Args:
        file_path: Ruta al archivo CSV con columnas 'Date' y 'Close'
        
    Returns:
        DataFrame de pandas con datos de precios de cierre indexados por fecha
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def check_data_quality(df):
    """
    Verifica la calidad de los datos y maneja valores faltantes.
    
    Args:
        df: DataFrame con datos de precios
        
    Returns:
        DataFrame con datos limpios
    """
    # Verificar valores faltantes
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Valores faltantes encontrados: {missing_values}")
        # Método simple: forward fill seguido de backward fill
        df = df.ffill().bfill()
        
    # Verificar valores negativos o cero en precios
    if (df['Close'] <= 0).any():
        print("¡Advertencia! Hay precios negativos o cero en los datos.")
        
    return df

def save_processed_data(df, output_path):
    """
    Guarda los datos procesados en un archivo CSV.
    
    Args:
        df: DataFrame procesado
        output_path: Ruta de salida para el archivo CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Datos procesados guardados en: {output_path}")