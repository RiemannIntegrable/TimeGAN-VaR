import numpy as np
import pandas as pd
from scipy import stats
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def prepare_data_for_evaluation(real_series, synthetic_windows, window_size=250, stride=None):
    """
    Prepara datos reales y sintéticos para evaluación.
    
    Args:
        real_series: Serie temporal real (puede ser DataFrame o array)
        synthetic_windows: Ventanas sintéticas de forma (n_samples, window_size, n_features)
        window_size: Tamaño de ventana para los datos reales
        stride: Salto entre ventanas consecutivas (None para no superponer, 
                para solapar parcialmente usar un valor < window_size)
    
    Returns:
        Tupla de (real_windows, synthetic_windows) listas para evaluación
    """
    # Determinar características de los datos sintéticos
    n_synthetic, seq_length, n_features = synthetic_windows.shape
    
    # Verificar que real_series tenga la forma adecuada
    if isinstance(real_series, pd.DataFrame):
        real_series = real_series.values
        
    # Si real_series es 1D, convertirlo a 2D
    if len(real_series.shape) == 1:
        real_series = real_series.reshape(-1, 1)
    
    # Establecer stride por defecto
    if stride is None:
        stride = window_size  # Sin superposición
    
    # Crear ventanas a partir de los datos reales
    n_samples = (len(real_series) - window_size) // stride + 1
    real_windows = np.zeros((n_samples, window_size, real_series.shape[1]))
    
    for i in range(n_samples):
        start_idx = i * stride
        end_idx = start_idx + window_size
        real_windows[i] = real_series[start_idx:end_idx]
    
    # Asegurar que ambos conjuntos tienen las mismas dimensiones de características
    if real_windows.shape[2] != n_features:
        raise ValueError(f"La dimensión de características no coincide: real={real_windows.shape[2]}, sintético={n_features}")
    
    return real_windows, synthetic_windows

def calculate_kl_divergence(real_data, synthetic_data):
    """
    Calcula la divergencia KL entre distribuciones reales y sintéticas.
    
    Args:
        real_data: Array de datos reales
        synthetic_data: Array de datos sintéticos
        
    Returns:
        Divergencia KL estimada
    """
    # Aplanar los datos si son multidimensionales
    real_flat = real_data.flatten()
    synth_flat = synthetic_data.flatten()
    
    # Estimar densidades usando histogramas
    bins = min(100, int(np.sqrt(len(real_flat))))
    
    # Calcular histogramas normalizados
    hist_real, bin_edges = np.histogram(real_flat, bins=bins, density=True)
    hist_synth, _ = np.histogram(synth_flat, bins=bin_edges, density=True)
    
    # Evitar divisiones por cero y log(0)
    hist_real = np.maximum(hist_real, 1e-10)
    hist_synth = np.maximum(hist_synth, 1e-10)
    
    # Calcular KL-divergence
    kl_div = np.sum(hist_real * np.log(hist_real / hist_synth))
    
    return kl_div

def discriminative_score(real_data, synthetic_data):
    """
    Calcula el discriminative score, que mide la capacidad de un clasificador simple
    para distinguir entre datos reales y sintéticos.
    
    Args:
        real_data: Array de datos reales de forma (n_samples, time_series_len, n_features)
        synthetic_data: Array de datos sintéticos de la misma forma
        
    Returns:
        Exactitud del clasificador (score menor indica mejor generación)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Preparar los datos
    n_real = real_data.shape[0]
    n_synth = synthetic_data.shape[0]
    
    # Aplanar la dimensión temporal
    real_flat = real_data.reshape(n_real, -1)
    synth_flat = synthetic_data.reshape(n_synth, -1)
    
    # Crear etiquetas (0 para real, 1 para sintético)
    real_labels = np.zeros(n_real)
    synth_labels = np.ones(n_synth)
    
    # Combinar datos
    combined_data = np.vstack([real_flat, synth_flat])
    combined_labels = np.hstack([real_labels, synth_labels])
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        combined_data, combined_labels, test_size=0.2, random_state=42)
    
    # Entrenar un clasificador
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluar
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def predictive_score(real_data, synthetic_data, prediction_horizon=5):
    """
    Calcula el predictive score, que mide la precisión de un modelo
    entrenado con datos sintéticos para predecir datos reales.
    
    Args:
        real_data: Array de datos reales de forma (n_real_samples, time_series_len, n_features)
        synthetic_data: Array de datos sintéticos de forma (n_synth_samples, time_series_len, n_features)
        prediction_horizon: Número de pasos futuros a predecir
        
    Returns:
        Error cuadrático medio relativo (ratio entre el error del modelo sintético y el modelo real)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    
    # Preparar conjuntos de datos reales
    X_real = real_data[:, :-prediction_horizon, :]
    y_real = real_data[:, -prediction_horizon:, :]
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42)
    
    # Aplanar la dimensión temporal
    X_real_train_flat = X_real_train.reshape(X_real_train.shape[0], -1)
    y_real_train_flat = y_real_train.reshape(y_real_train.shape[0], -1)
    X_real_test_flat = X_real_test.reshape(X_real_test.shape[0], -1)
    y_real_test_flat = y_real_test.reshape(y_real_test.shape[0], -1)
    
    # Preparar datos sintéticos
    X_synth = synthetic_data[:, :-prediction_horizon, :]
    y_synth = synthetic_data[:, -prediction_horizon:, :]
    X_synth_flat = X_synth.reshape(X_synth.shape[0], -1)
    y_synth_flat = y_synth.reshape(y_synth.shape[0], -1)
    
    # 1. Entrenar un modelo con datos reales (línea base)
    model_real = LinearRegression()
    model_real.fit(X_real_train_flat, y_real_train_flat)
    y_pred_real = model_real.predict(X_real_test_flat)
    mse_real = mean_squared_error(y_real_test_flat, y_pred_real)
    
    # 2. Entrenar un modelo con datos sintéticos
    model_synth = LinearRegression()
    model_synth.fit(X_synth_flat, y_synth_flat)  # Entrenamos con datos sintéticos completos
    y_pred_synth = model_synth.predict(X_real_test_flat)  # Predecimos en datos reales
    mse_synth = mean_squared_error(y_real_test_flat, y_pred_synth)
    
    # Calcular el score relativo (más cercano a 1 es mejor)
    relative_score = mse_synth / mse_real if mse_real > 0 else float('inf')
    
    return relative_score

def visualize_tsne(real_data, synthetic_data, n_components=2, perplexity=30):
    """
    Visualiza la proyección t-SNE de datos reales y sintéticos.
    
    Args:
        real_data: Array de datos reales
        synthetic_data: Array de datos sintéticos
        n_components: Número de componentes para t-SNE
        perplexity: Parámetro de perplexidad para t-SNE
        
    Returns:
        Figura de matplotlib
    """
    # Aplanar la dimensión temporal
    real_flat = real_data.reshape(real_data.shape[0], -1)
    synth_flat = synthetic_data.reshape(synthetic_data.shape[0], -1)
    
    # Combinar datos
    combined_data = np.vstack([real_flat, synth_flat])
    
    # Crear etiquetas (0 para real, 1 para sintético)
    labels = np.hstack([np.zeros(real_data.shape[0]), np.ones(synthetic_data.shape[0])])
    
    # Aplicar t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    reduced_data = tsne.fit_transform(combined_data)
    
    # Visualizar
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, 
                         cmap='viridis', alpha=0.7, s=10)
    
    # Añadir leyenda
    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Tipo de datos")
    labels = ['Real', 'Sintético']
    for i, text in enumerate(legend.get_texts()):
        text.set_text(labels[i])
    
    ax.set_title('Visualización t-SNE de datos reales vs. sintéticos')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.grid(True, alpha=0.3)
    
    return fig