# 🌾 Sistema de Recomendación de Cultivos - Proyecto ML

## 📋 Descripción del Proyecto

Este proyecto desarrolla un sistema inteligente de Machine Learning para recomendar el cultivo más adecuado basándose en parámetros agroclimáticos y propiedades del suelo. Utiliza la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining) y está implementado en Python con Jupyter Notebooks.

### 🎯 Objetivo

Predecir el cultivo óptimo para sembrar considerando:
- Nutrientes del suelo (N, P, K)
- Condiciones climáticas (Temperatura, Humedad, Precipitación)
- Propiedades químicas (pH)

### 📊 Dataset

- **Fuente**: [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Tamaño**: 2,200 muestras
- **Características**: 7 variables numéricas
- **Clases**: 22 tipos de cultivos

## 🗂️ Estructura del Proyecto

```
proyecto-recomendacion-cultivos/
│
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias de Python
│
├── datos/
│   ├── crudos/                       # Dataset original
│   │   └── Crop_recommendation.csv
│   ├── procesados/                   # Datos preparados
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── externos/                     # Datos de referencia (opcional)
│
├── notebooks/
│   ├── 1_comprension_negocio.ipynb   # Fase 1 CRISP-DM
│   ├── 2_comprension_datos.ipynb     # Fase 2 CRISP-DM (EDA)
│   ├── 3_preparacion_datos.ipynb     # Fase 3 CRISP-DM
│   ├── 4_modelado.ipynb              # Fase 4 CRISP-DM
│   ├── 5_evaluacion.ipynb            # Fase 5 CRISP-DM
│   └── 6_despliegue.ipynb            # Fase 6 CRISP-DM
│
├── src/
│   ├── __init__.py
│   ├── procesamiento_datos.py        # Utilidades de datos
│   ├── ingenieria_features.py        # Feature engineering
│   ├── entrenamiento_modelo.py       # Entrenamiento
│   └── evaluacion.py                 # Métricas y evaluación
│
├── modelos/                          # Modelos guardados
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── mejor_modelo.pkl
│
├── reportes/
│   ├── figuras/                      # Visualizaciones
│   ├── reporte_entrega1.pdf
│   ├── reporte_entrega2.pdf
│   └── reporte_final.pdf
│
└── despliegue/                       # Scripts de producción (opcional)
    └── app.py
```

## 🚀 Inicio Rápido

### 1️⃣ Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

### 2️⃣ Instalación

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd proyecto-recomendacion-cultivos

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3️⃣ Descargar el Dataset

1. Ir a [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
2. Descargar `Crop_recommendation.csv`
3. Colocar el archivo en `datos/crudos/`

### 4️⃣ Ejecutar los Notebooks

```bash
# Iniciar Jupyter Notebook
jupyter notebook

# O usar Jupyter Lab
jupyter lab
```

Ejecutar los notebooks en orden:
1. `1_bu_du_eda.ipynb`


## 📅 Cronograma de Entregas

### Entrega 1 - 02/10/2025 (5%)
- ✅ Notebook de EDA completo
- ✅ Data Card del dataset
- ✅ Modelo baseline (Regresión Logística)
- ✅ Reporte PDF breve
- ✅ Repositorio Git

**Contenido**: Notebooks `1_bu_du_eda.ipynb`

### Entrega 2 - 09/10/2025 (10%)
- ✅ Comparación de 2-3 familias de modelos
- ✅ Pipelines de entrenamiento
- ✅ Validación cruzada
- ✅ Análisis de hiperparámetros
- ✅ Reporte de resultados

**Contenido**: 

### Entrega 3 - 05/11/2025 (20%)
- ✅ Modelo final optimizado
- ✅ Interpretación del modelo (SHAP)
- ✅ Reporte final completo
- ✅ Póster del proyecto
- ✅ Presentación

**Contenido**: 

## 🎯 Metodología CRISP-DM

### Fase 1: Comprensión del Negocio
- Definición del problema
- Objetivos y criterios de éxito
- Análisis de stakeholders

### Fase 2: Comprensión de los Datos
- Análisis Exploratorio de Datos (EDA)
- Estadísticas descriptivas
- Identificación de patrones
- Data Card

### Fase 3: Preparación de los Datos
- Limpieza de datos
- Manejo de outliers
- Escalado de características
- División train-test

### Fase 4: Modelado
- Modelo baseline (Regresión Logística)
- Árboles de Decisión
- Random Forest
- XGBoost / LightGBM
- SVM
- Ajuste de hiperparámetros

### Fase 5: Evaluación
- Métricas: Accuracy, Precision, Recall, F1-Score
- Matriz de confusión
- Validación cruzada
- Comparación de modelos
- Interpretabilidad (SHAP)

### Fase 6: Despliegue
- Serialización del modelo
- API de predicción (opcional)
- Documentación
- Recomendaciones

## 📊 Características del Dataset

| Característica | Tipo | Descripción | Unidad |
|---------------|------|-------------|--------|
| N | Numérico | Contenido de Nitrógeno | kg/ha |
| P | Numérico | Contenido de Fósforo | kg/ha |
| K | Numérico | Contenido de Potasio | kg/ha |
| temperature | Numérico | Temperatura | °C |
| humidity | Numérico | Humedad | % |
| ph | Numérico | Valor de pH | - |
| rainfall | Numérico | Precipitación | mm |
| label | Categórico | Tipo de cultivo | 22 clases |

### 🌱 Cultivos Incluidos (22 clases)

Arroz, Maíz, Garbanzo, Frijol, Guisante, Frijol Polilla, Frijol Mungo, Granada, Plátano, Mango, Uvas, Algodón, Yute, Gramo Negro, Sandía, Melón, Manzana, Naranja, Papaya, Coco, Lenteja, Café

## 📈 Criterios de Éxito

### Técnicos
- ✅ Accuracy ≥ 95%
- ✅ Precision y Recall ≥ 90% por clase
- ✅ F1-Score ≥ 90%
- ✅ Modelo interpretable

### De Negocio
- ✅ Recomendaciones accionables
- ✅ Tiempo de predicción < 1 segundo
- ✅ Sistema escalable
- ✅ Confianza del usuario (explicabilidad)

## 🛠️ Tecnologías Utilizadas

- **Lenguaje**: Python 3.12+
- **Análisis de Datos**: pandas, numpy
- **Visualización**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Interpretabilidad**: SHAP
- **Notebooks**: Jupyter Lab
- **Control de versiones**: Git

## 📝 Guía de Contribución

1. Fork el proyecto
2. Crear una rama (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 👥 Autor(es)

- Daniel Alejandro Garcia Zuluaica
- Elizabeth Toro Chalarca
- Edward Alejandro Rayo Cortés

**Universidad**: EAFIT
**Curso**: Aprendizaje de Máquina Aplicado

## 📚 Referencias

1. IBM SPSS Modeler. (2012). *CRISP-DM Guide*. IBM Corporation.
2. Crop Recommendation Dataset. Kaggle. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
3. Scikit-learn Documentation. https://scikit-learn.org/
4. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NeurIPS.

## 📄 Licencia

Este proyecto es un trabajo académico para el curso de Aprendizaje de Máquina Aplicado en EAFIT.

## 🤝 Agradecimientos

- Profesor: Marco Terán
- Universidad EAFIT
- Comunidad de Kaggle por el dataset

---

**Última actualización**: Septiembre 2025

**Estado del Proyecto**: 🚧 En desarrollo