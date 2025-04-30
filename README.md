# 📊 TimeGAN-VaR: Generación Sintética de Series Temporales para VaR 📈

## 📋 Descripción

Este proyecto implementa un enfoque avanzado para el cálculo del Valor en Riesgo (VaR) mediante la generación de series temporales sintéticas utilizando Redes Generativas Adversarias Temporales (TimeGAN). Este método permite una estimación más robusta del riesgo financiero a través de la simulación de Monte Carlo con datos sintéticos que conservan las propiedades estadísticas y temporales de las series históricas.

**Proyecto desarrollado para:**  
Clase de Administración Cuantitativa de Riesgos Financieros  
Maestría en Actuaría y Finanzas  
Universidad Nacional de Colombia

**Autor:** Jose Miguel Acuña Hernandes (Migue)

## 🎯 Objetivos

* **Objetivo Principal:**
    * Implementar un modelo TimeGAN para la generación de series temporales sintéticas que capturen las propiedades estadísticas y temporales de datos financieros históricos
    * Aplicar dichas series sintéticas para calcular medidas de riesgo (VaR) mediante simulación de Monte Carlo

* **Objetivos Secundarios:**
    * Comparar el VaR estimado usando datos sintéticos contra métodos tradicionales
    * Evaluar la calidad de las series sintéticas generadas mediante métricas específicas
    * Desarrollar una estructura de proyecto modular y reutilizable para el análisis de riesgos financieros

## 🌟 Características

- 🧠 Implementación completa de TimeGAN en TensorFlow/Keras
- 📉 Generación de trayectorias sintéticas de precios de activos financieros
- 🔢 Cálculo de VaR mediante múltiples metodologías:
  - 🎲 Simulación de Monte Carlo
  - 📊 Enfoque Paramétrico (Normal y t-Student)
  - 📜 Método Histórico
- 📊 Visualizaciones avanzadas para análisis de resultados
- 🧪 Métricas de evaluación para datos sintéticos

## 🛠️ Stack Tecnológico

Este proyecto utiliza las siguientes tecnologías y librerías principales:

* **Lenguaje:** Python 🐍
* **Gestor de Entornos:** Conda 🧠
* **Análisis Numérico:** NumPy
* **Manipulación de Datos:** Pandas 🐼
* **Machine Learning / Modelos:** TensorFlow 🔥, Keras
* **Visualización:** Matplotlib 📈, Seaborn 📊
* **Estadística:** SciPy, Statsmodels
* **Entorno Interactivo:** Jupyter Notebooks 📓

## 🏗️ Estructura del Proyecto

```
proyecto_timegan_var/
│
├── 📁 src/
│   ├── 📁 data/           # Módulos para procesamiento de datos
│   │   ├── __init__.py     # Exporta funciones de datos
│   │   ├── loader.py       # Carga y preprocesamiento de datos
│   │   ├── transform.py    # Transformaciones para series financieras
│   │   └── windowing.py    # Creación de ventanas temporales
│   │
│   ├── 📁 models/         # Implementación de TimeGAN y cálculo de VaR
│   │   ├── __init__.py     # Exporta funciones de modelos
│   │   ├── timegan.py      # Implementación TimeGAN
│   │   └── var_model.py    # Cálculo del VaR
│   │
│   ├── 📁 utils/          # Utilidades de evaluación y visualización
│   │   ├── __init__.py     # Exporta funciones de utilidades
│   │   ├── evaluation.py   # Métricas de evaluación
│   │   └── visualization.py # Visualizaciones
│   │
│   └── __init__.py         # Archivo principal del paquete
│
├── 📁 notebooks/          # Jupyter notebooks con ejemplos y tutoriales
│   └── proyecto.ipynb     # Notebook principal
│
├── 📁 data/
│   ├── 📁 raw/            # Datos originales sin procesar
│   └── 📁 processed/      # Datos procesados listos para el modelo
│
├── 📁 models/             # Modelos entrenados guardados en formato .h5
│
└── 📄 README.md
```

## ⚙️ Instalación y Configuración del Entorno

Sigue estos pasos para configurar el entorno de desarrollo necesario para ejecutar el proyecto.

### Prerrequisitos

* Asegúrate de tener instalado [Anaconda](https://www.anaconda.com/products/distribution) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Pasos de Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/timegan-var.git
    cd timegan-var
    ```

2.  **📦 Crea el entorno Conda desde el archivo `environment.yml`:**
    ```bash
    conda env create -f environment.yml
    ```
    ✨ Este método lee directamente `environment.yml`, instalando todo lo necesario con las versiones correctas.

3.  **🚀 Activa el entorno:**
    ```bash
    conda activate timegan-var
    ```

4.  **✅ Verifica la instalación (Opcional):**
    ```bash
    conda list tensorflow
    ```
    Para Jupyter, registra el kernel si es necesario:
    ```bash
    python -m ipykernel install --user --name=timegan-var --display-name "Python (TimeGAN-VaR)"
    ```

### Contenido del `environment.yml`

```yaml
# Entorno Conda para TimeGAN-VaR
name: timegan-var
channels:
  - conda-forge
  - defaults
dependencies:
  # --- Core ---
  - python=3.8
  - pip

  # --- Análisis y Datos ---
  - numpy~=1.20.0
  - pandas~=1.3.0
  - scipy>=1.7.0
  - scikit-learn~=0.24.0
  - statsmodels>=0.12.0

  # --- Visualización ---
  - matplotlib>=3.4.0
  - seaborn>=0.11.0

  # --- Entorno Interactivo ---
  - jupyterlab
  - ipykernel

  # --- Dependencias instaladas con Pip ---
  - pip:
      - tensorflow==2.5.0
```

## 💻 Uso

### Preparación de Datos

```python
from src.data import load_stock_data, calculate_returns, add_features, normalize_data, create_windows

# Cargar datos históricos
df = load_stock_data('data/raw/stock_prices.csv')

# Calcular rendimientos y añadir características
df = calculate_returns(df, method='log')
df = add_features(df, window_sizes=[5, 10, 20])

# Normalizar datos
df_norm, norm_params = normalize_data(df, method='minmax')

# Crear ventanas para entrenamiento
windows = create_windows(df_norm, window_size=30, stride=5)
```

### Entrenamiento del Modelo TimeGAN

```python
from src.models import timegan_init, timegan_train, timegan_export_generator

# Inicializar y entrenar el modelo
timegan_tuple = timegan_init(time_series_len=30, features=5, rnn_units=64, rnn_layers=3)
trained_model = timegan_train(windows, timegan_tuple, epochs=2000, batch_size=32, learning_rate=0.001)

# Exportar el generador para su uso
generator = timegan_export_generator(trained_model)
```

### Generación de Series Sintéticas y Cálculo de VaR

```python
from src.models import generator_gen, monte_carlo_var

# Generar datos sintéticos
synthetic_windows = generator_gen(generator, generate_cnt=100)

# Calcular VaR mediante Monte Carlo
initial_price = 100.0
var, es = monte_carlo_var(synthetic_windows, initial_price, horizon=1, confidence_level=0.95)

print(f"VaR (95%): ${var:.2f}")
print(f"Expected Shortfall: ${es:.2f}")
```

### Visualización de Resultados

```python
from src.utils import plot_stock_prices, plot_returns_distribution, plot_var_histogram

# Visualizar precios históricos y proyecciones
plot_stock_prices(historical_prices, synthetic_prices, dates=historical_dates)

# Comparar distribuciones de rendimientos
plot_returns_distribution(real_returns, synthetic_returns)

# Visualizar VaR en la distribución de P&L
plot_var_histogram(pnl_values, var_value, confidence_level=0.95)
```

## 🔬 Marco Teórico

### TimeGAN

TimeGAN (Time-series Generative Adversarial Network) es un modelo propuesto por Yoon, Jarrett y van der Schaar que extiende la arquitectura GAN tradicional para trabajar específicamente con series temporales. El modelo combina:

1. 🧠 Un mecanismo de autoencoder (embedder-recovery) para codificar la estructura de los datos
2. 🎭 Una arquitectura adversarial clásica para la generación
3. 🔄 Un componente supervisor que garantiza la coherencia temporal

Esta combinación permite generar series temporales sintéticas que preservan tanto las propiedades distribucionales estáticas como las dinámicas temporales de los datos reales.

### VaR por Simulación de Monte Carlo

La simulación de Monte Carlo para el cálculo de VaR implica:

1. 🎲 Generar múltiples trayectorias futuras posibles
2. 📊 Calcular la distribución de pérdidas y ganancias (P&L)
3. 📉 Determinar el percentil correspondiente al nivel de confianza deseado

La ventaja de usar datos sintéticos generados por TimeGAN es que capturan mejor la estructura temporal y las propiedades estadísticas de los datos financieros que los métodos tradicionales.

## 📝 Evaluación y Validación

El proyecto incluye múltiples métricas para evaluar la calidad de los datos sintéticos generados:

- 📏 Divergencia KL para medir la similitud distribucional
- 🔍 Discriminative Score para evaluar la capacidad de distinguir datos reales y sintéticos
- 🎯 Predictive Score para medir la capacidad predictiva de modelos entrenados con datos sintéticos
- 👁️ Visualización t-SNE para análisis exploratorio

## 👨‍🏫 Aplicaciones Educativas

Este proyecto sirve como herramienta educativa para comprender:

- 🧠 El funcionamiento de los modelos generativos avanzados
- 📊 Los fundamentos del análisis de riesgo financiero
- 🔢 La aplicación de técnicas de simulación en finanzas
- 📈 La modelación de series temporales financieras

## 🙏 Agradecimientos

- A los profesores de la Maestría en Actuaría y Finanzas de la Universidad Nacional de Colombia por su guía y enseñanzas
- A los autores del paper original de TimeGAN (Yoon, Jarrett y van der Schaar)
- A la comunidad de código abierto por sus contribuciones en el campo del aprendizaje automático aplicado a finanzas

## 📜 Licencia

Este proyecto está licenciado bajo MIT License - ver el archivo LICENSE para más detalles.

---

⭐ Si encuentras útil este proyecto, no dudes en darle una estrella en GitHub ⭐

Para cualquier consulta o sugerencia, puedes contactarme en: [tu-email@example.com]