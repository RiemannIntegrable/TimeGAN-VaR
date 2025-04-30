---

<div align="center">

# **TimeGAN-VaR: Generación Sintética de Series Temporales para VaR**

</div>

---

## 📋 **Descripción**

Este proyecto implementa un enfoque avanzado para el cálculo del Valor en Riesgo (VaR) mediante la generación de series temporales sintéticas utilizando Redes Generativas Adversarias Temporales (TimeGAN). Este método permite una estimación más robusta del riesgo financiero a través de la simulación de Monte Carlo con datos sintéticos que conservan las propiedades estadísticas y temporales de las series históricas.

**Proyecto desarrollado para:*  
Clase de Administración Cuantitativa de Riesgos Financieros  
Maestría en Actuaría y Finanzas  
Universidad Nacional de Colombia

**Autor:** Jose Miguel Acuña Hernandes (RiemannIntegrable)

## 💻 **Uso**

En el noteboon timegan_var.ipynb, en la descarga y presocesamiento de los datos construya usted su propio portafolio descargando la informacion de las acciones en yahoofinance con la libreria yfinance como se ve en el notebook.

## 🎯 **Objetivos**

* **Objetivo Principal:**
    * Implementar un modelo TimeGAN para la generación de series temporales sintéticas que capturen las propiedades estadísticas y temporales de datos financieros históricos
    * Aplicar dichas series sintéticas para calcular medidas de riesgo (VaR) mediante simulación de Monte Carlo

* **Objetivos Secundarios:**
    * Comparar el VaR estimado usando datos sintéticos contra métodos tradicionales
    * Evaluar la calidad de las series sintéticas generadas mediante métricas específicas
    * Desarrollar una estructura de proyecto modular y reutilizable para el análisis de riesgos financieros

## 🌟 **Características**

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
timegan_var/
│
├── 💾 data/
│   ├── 📥 input/           # Datos originales para conformar el portafolio
│   ├── ⚙️ processed/       # Retornos logarítmicos procesados del portafolio
│   └── 📤 output/          # Resultados y métricas exportadas del análisis
│
├── 🖼️ images/              # Visualizaciones y diagramas generados
│
├── 🧠 models/              # Modelos TimeGAN entrenados (.h5)
│
├── 📓 notebooks/           # Cuadernos Jupyter interactivos
│   ├── 📊 timegan_var.ipynb # Notebook principal con implementación completa
│   └── 📈 yfinance.ipynb    # Exploración de datos de Yahoo Finance
│
├── ⚙️ src/                 # Código fuente modular del proyecto
│   ├── 📊 data/            # Procesamiento y manipulación de datos
│   │   ├── 🔄 __init__.py   # Exportación de funciones de datos
│   │   ├── 📥 loader.py     # Carga y validación de datos financieros
│   │   ├── 🔧 transform.py  # Transformaciones para series temporales
│   │   └── 🪟 windowing.py  # Creación de ventanas para entrenamiento
│   │
│   ├── 🤖 models/          # Implementaciones de modelos
│   │   ├── 🔄 __init__.py   # Exportación de funciones de modelos
│   │   ├── 🧮 timegan.py    # Arquitectura TimeGAN completa
│   │   └── 📉 var_model.py  # Cálculo de VaR con distintas metodologías
│   │
│   ├── 🛠️ utils/           # Utilidades y herramientas auxiliares
│   │   ├── 🔄 __init__.py    # Exportación de funciones de utilidades
│   │   ├── 📏 evaluation.py  # Métricas de evaluación y validación
│   │   └── 📊 visualization.py # Generación de gráficos y visualizaciones
│   │
│   └── 🔄 __init__.py      # Exportación de componentes principales
│
├── 📝 tex/                 # Proyecto entregable en LaTeX y PDF
│
├── 🙈 .gitignore           # Archivos y directorios excluidos del repositorio
├── ⚙️ environment.yml      # Especificación del entorno Conda
├── ⚖️ LICENSE              # Licencia MIT del proyecto
└── 📚 README.md            # Este archivo de documentación
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
  - python=3.11
  - pip

  # --- Análisis y Datos ---
  - numpy~=1.26.4
  - pandas~=2.2.3
  - scipy>=1.15.2
  - scikit-learn~=1.4.2
  - statsmodels>=0.12.0

  # --- Visualización ---
  - matplotlib>=3.10.1
  - seaborn>=0.13.2

  # --- Entorno Interactivo ---
  - jupyterlab
  - ipykernel
  - notebook-shim
  - jupyter_server

  # --- Utilidades ---
  - tqdm>=4.67.1
  - pyyaml>=6.0.2

  # --- Dependencias instaladas con Pip ---
  - pip:
      - tensorflow==2.16.1
      - keras==3.9.2
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

## 📝 Atribución del Código Original

La implementación base de TimeGAN en Keras utilizada en este proyecto está basada en el código desarrollado por [gusxo](https://github.com/gusxo) en el repositorio [TimeGAN-keras](https://github.com/gusxo/TimeGAN-keras). El repositorio original no especifica una licencia explícita.

A partir de esta implementación base, se han realizado las siguientes adaptaciones y extensiones:

- Estructuración del código en una arquitectura modular de proyecto
- Adaptación para el análisis de series temporales financieras
- Implementación de métodos para el cálculo de Valor en Riesgo (VaR)
- Adición de herramientas de evaluación y visualización específicas para datos financieros
- Desarrollo de interfaces para facilitar la experimentación y uso

Este reconocimiento se realiza como buena práctica académica y de desarrollo de software.

## 📜 Licencia

Las adaptaciones, extensiones y contribuciones originales de este proyecto están licenciadas bajo la Licencia MIT - ver el archivo LICENSE para más detalles. Esta licencia aplica únicamente a las contribuciones originales a este proyecto y no modifica ningún derecho existente sobre el código base de TimeGAN-keras.

---

⭐ Si encuentras útil este proyecto, no dudes en darle una estrella en GitHub ⭐

Para cualquier consulta o sugerencia, puedes contactarme en: [tu-email@example.com]