# 📊 Análisis Cuantitativo de Riesgos Financieros 📈

**Proyecto Final - Administración Cuantitativa de Riesgos Financieros**
**Universidad Nacional de Colombia** 🎓

**Autor:** Jose Miguel Acuña Hernandes (Migue)

---

## 📝 Descripción del Proyecto

*( **Migue, completa esta sección:** Describe brevemente de qué trata el proyecto. ¿Qué problema aborda? ¿Cuál es el enfoque principal? Ejemplo: "Este proyecto implementa modelos de Valor en Riesgo (VaR) y Expected Shortfall (ES) para analizar el riesgo de mercado de un portafolio de acciones colombianas..." )*

---

## 🎯 Objetivos

*( **Migue, completa esta sección:** Lista los objetivos principales y secundarios que buscas alcanzar con este proyecto. )*

* **Objetivo Principal:**
    * ...
* **Objetivos Secundarios:**
    * ...
    * ...

---

## 🛠️ Stack Tecnológico

Este proyecto utiliza las siguientes tecnologías y librerías principales:

* **Lenguaje:** Python 🐍
* **Gestor de Entornos:** Conda <0xF0><0x9F><0xA7><0xAD>
* **Análisis Numérico:** NumPy
* **Manipulación de Datos:** Pandas 🐼
* **Machine Learning / Modelos:** Scikit-learn 🤖, TensorFlow 🔥
* **Visualización:** Matplotlib 📈
* **Entorno Interactivo:** JupyterLab / Jupyter Notebooks 📓

---

## ⚙️ Instalación y Configuración del Entorno

Sigue estos pasos para configurar el entorno de desarrollo necesario para ejecutar el proyecto.

### Prerrequisitos

* Asegúrate de tener instalado [Anaconda](https://www.anaconda.com/products/distribution) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Pasos de Instalación

1.  **Clona el repositorio (si aún no lo has hecho):**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_DIRECTORIO_DEL_REPOSITORIO>
    ```

2.  **📦 Crea el entorno Conda desde el archivo `environment.yml`:**
    Abre una terminal o Anaconda Prompt y ejecuta el siguiente comando en el directorio raíz del proyecto:
    ```bash
    conda env create -f environment.yml
    ```
    ✨ ¡Este es el método recomendado! Lee directamente `environment.yml`, instala todo lo necesario (con las versiones correctas `~=`, `>=`) y evita los problemas de interpretación de caracteres especiales (`[]`, `>`) de la shell.

3.  **🚀 Activa el entorno:**
    ```bash
    conda activate mi_proyecto_env
    ```
    *(Asegúrate de que `mi_proyecto_env` coincida con el nombre especificado en `environment.yml`)*

4.  **✅ Verifica la instalación (Opcional):**
    ```bash
    conda list tensorflow
    ```
    Para Jupyter, registra el kernel si es necesario (aunque `ipykernel` está en el `yml`):
    ```bash
    python -m ipykernel install --user --name=mi_proyecto_env --display-name "Python (AQRF)"
    ```
    *(Reemplaza `mi_proyecto_env` por el nombre real del entorno si es diferente)*

### Contenido del `environment.yml` de Ejemplo

Este archivo define todas las dependencias. Asegúrate de que el tuyo esté completo.

```yaml
# Nombre descriptivo para el entorno Conda
name: mi_proyecto_env
channels:
  - conda-forge # Canal principal para muchas librerías
  - defaults
  # - nvidia # Necesario si instalas 'cudatoolkit' vía conda para TensorFlow GPU
dependencies:
  # --- Core ---
  - python=3.11 # O la versión que uses
  - pip

  # --- Análisis y Datos ---
  - numpy~=1.26.0
  - pandas~=2.2.0
  - scikit-learn~=1.4.0

  # --- Visualización ---
  - matplotlib>=3.5

  # --- Entorno Interactivo ---
  - jupyterlab
  - ipykernel

  # --- Dependencias de TensorFlow (GPU) ---
  # Opción 1: Instalar CUDA Toolkit vía Conda (recomendado si usas Conda)
  # - cudatoolkit=11.8 # Ajusta la versión según requiera TF 2.16.1
  # - cudnn=8.x # Ajusta la versión según requiera TF 2.16.1

  # --- Dependencias instaladas con Pip ---
  - pip:
      # Opción A (Si instalaste cudatoolkit/cudnn con Conda):
      # - tensorflow==2.16.1
      # Opción B (Usando extras de Pip - requiere drivers NVIDIA y toolkit/cudnn en el sistema):
      - "tensorflow[and-cuda]==2.16.1" # Comillas por seguridad con los corchetes

# Nota sobre GPU: Aparte de las librerías de Conda/Pip, necesitas tener los
# drivers de NVIDIA actualizados e instalados en tu sistema operativo.
```