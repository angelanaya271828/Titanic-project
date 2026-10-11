

# 🚢 Titanic MLOps Pipeline: Optimización y tracking modular

Este proyecto implementa un pipeline de Machine Learning (ML) end-to-end para el clásico problema del Titanic, el cual fue refactorizado buscando seguir las mejores prácticas de **MLOps**. 

El repositorio demuestra la transición de código tipo *notebook* hacia una arquitectura robusta orientada a objetos (POO), utilizando modularización, parametrización dinámica mediante interfaces de línea de comandos (CLI) y un control estricto del ciclo de vida del experimento con **MLFlow Tracking**.

---

## 🏗️ Arquitectura y modularidad
El código fuente ha sido refactorizado y dividido para separar las responsabilidades, facilitando el testing, la gobernanza y mitigando el sesgo entre la etapa del entrenamiento y una futura etapa del servicio (*Training-Serving Skew*):

Dicho código fue refactorizado en los siguientes archivos:
* `titanic_module/dataset.py`: Aprovisionamiento, limpieza estructural y split estratificado de la base de datos en Train/Test.
* `titanic_module/features.py`: Definición del proceso de Feature Engineering (Imputación y Escalado/Encoding de variables).
* `titanic_module/modeling/train.py`: Orquestador principal que inyecta dependencias, ensambla el pipeline unificado de `scikit-learn` y registra los experimentos en MLFlow.

Es importante señalar que la etapa inicial de Análisis Exploratorio de los datos (EDA) se encuentra en `notebooks/1.0-aaaa-Initial EDA.ipynb`.
---

## ⚙️ Configuración del entorno y dependencias

Para garantizar la higiene del proyecto y evitar colisiones de versiones, el proyecto utiliza un entorno virtual aislado.

**1. Crear y activar el entorno virtual:**
> python -m venv venv
> # En Windows:
> .\venv\Scripts\activate
> # En macOS/Linux:
> source venv/bin/activate

**2. Instalar las dependencias exactas:**
> pip install -r requirements.txt

**Nota de reproducibilidad:** El archivo `requirements.txt` contiene las versiones congeladas exactas de `scikit-learn`, `xgboost`, `mlflow`, etc., asegurando que el baseline de ML sea idéntico en cualquier ambiente.

---

## 🚀 Ejecución del pipeline

El orquestador de entrenamiento (`train.py`) ha sido diseñado utilizando `argparse` para permitir la inyección de hiperparámetros desde la terminal sin necesidad de modificar el código fuente.

### Consultar el Manual de ayuda
Para ver todos los argumentos disponibles y su documentación interna, utiliza la siguiente instrucción:
> python -m titanic_module.modeling.train --help

### Ejemplos de ejecución: Creación de experimentos

**1. Ejecución Base (Regresión logística con parámetros por defecto):**
> python -m titanic_module.modeling.train

**2. Ejecución iterando el algoritmo (XGBoost):**
> python -m titanic_module.modeling.train --model_type xgboost

**3. Ejecución con hiperparámetros personalizados (XGBoost):**
> python -m titanic_module.modeling.train --model_type xgboost --max_depth 8 --n_estimators 200 --learning_rate 0.1

---
## Reproducibilidad garantizada

Este pipeline cumple estrictamente con los criterios de reproducibilidad exigidos en los proyectos de Machine Learning:
1. **Control de aleatoriedad (Semillas):** El argumento `--random_state` (por defecto `42`) se propaga a lo largo de todo el ciclo de vida: desde la partición de los datos para conformar los conjuntos de Train y Test hasta la inicialización algorítmica de los clasificadores. Dos ejecuciones con los mismos parámetros de entrada generarán **exactamente** los mismos resultados.
2. **Registro de reglas de negocio:** Las funciones de aprovisionamiento generan diccionarios de metadatos (`structural_cleaning_params` y `data_cleaning_params`) que se envían a MLFlow, asegurando un registro auditable de la limpieza de datos.
3. **Empaquetado de Pipelines:** Se serializa el objeto `Pipeline` completo (Preprocesador + Clasificador), preparándolo como artefacto listo para su futura puesta en producción.

---

## 📊 Monitoreo y Tracking Avanzado con MLFlow

Todas las ejecuciones registran automáticamente hiperparámetros, métricas (Accuracy, ROC-AUC, F1-Score,...) y artefactos (Matriz de Confusión PNG, modelo empaquetado) en un servidor local de MLFlow.

Para visualizar y comparar los experimentos, levante el servidor de UI:
> mlflow ui

Abra su navegador en `[http://127.0.0.1:5000](http://127.0.0.1:5000)` y diríjase al experimento **Titanic_Modular_Optimization**.

---

