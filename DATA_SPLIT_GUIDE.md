# Guía del Sistema de Split Centralizado de Datos

## 📋 Descripción General

Este proyecto implementa un **sistema de split Train/Test centralizado** que garantiza que todos los notebooks usen exactamente los mismos datos de entrenamiento y prueba, siguiendo las mejores prácticas de Machine Learning.

## 🎯 Objetivo

**Prevenir data leakage** y garantizar evaluaciones justas al:

1. ✅ Realizar el split Train/Test **UNA SOLA VEZ** al inicio
2. ✅ Usar **SOLO datos de entrenamiento** para:
   - Feature engineering
   - Entrenamiento de modelos
   - Optimización de hiperparámetros
   - Validación cruzada
3. ✅ Reservar **datos de prueba** únicamente para evaluación final

## 📁 Estructura

```
proyecto_final/
├── src/
│   ├── data_loader.py          # Carga el dataset desde Kaggle
│   └── data_split.py           # ⭐ Módulo de split centralizado
├── models/
│   └── train_test_split_cache.pkl  # Cache del split (automático)
└── notebooks/
    ├── 2_crops_modeling.ipynb
    ├── 2.5_hyperparameter_optimization.ipynb
    └── 3_crops_feature_engineering.ipynb
```

## 🚀 Uso en Notebooks

### Importar el módulo

```python
import sys
sys.path.append('../src')

from data_split import get_train_test_split

# Obtener split (automáticamente usa el mismo split en todos los notebooks)
data = get_train_test_split()
```

### Extraer datos

```python
# Features de entrenamiento y prueba
X_train = data['X_train']
X_test = data['X_test']

# Target de entrenamiento y prueba (codificado)
y_train = data['y_train']
y_test = data['y_test']

# Target con etiquetas originales (opcional)
y_train_labels = data['y_train_labels']
y_test_labels = data['y_test_labels']

# Label encoder (para decodificar predicciones)
label_encoder = data['label_encoder']

# Metadata
feature_names = data['feature_names']  # ['N', 'P', 'K', ...]
class_names = data['class_names']      # ['apple', 'banana', ...]
split_info = data['split_info']        # Info del split
```

### Estructura del diccionario `data`

```python
{
    'X_train': DataFrame,           # Features de entrenamiento (1760 × 7)
    'X_test': DataFrame,            # Features de prueba (440 × 7)
    'y_train': ndarray,             # Target entrenamiento codificado (1760,)
    'y_test': ndarray,              # Target prueba codificado (440,)
    'y_train_labels': Series,       # Target entrenamiento original (1760,)
    'y_test_labels': Series,        # Target prueba original (440,)
    'label_encoder': LabelEncoder,  # Encoder entrenado
    'feature_names': list,          # Nombres de features
    'class_names': list,            # Nombres de clases
    'split_info': dict              # Metadata del split
}
```

## 📊 Configuración del Split

Por defecto, el split usa:

- **Test size**: 20% (440 muestras)
- **Train size**: 80% (1760 muestras)
- **Random state**: 42 (reproducibilidad)
- **Estratificación**: Sí (mantiene proporción de clases)

### Personalizar el split

```python
# Crear un nuevo split con diferentes parámetros
data = get_train_test_split(
    use_cache=False,      # No usar cache, crear nuevo split
    random_state=42,      # Semilla
    test_size=0.25        # 25% para test
)
```

### Regenerar el split

```python
from data_split import clear_cache

# Eliminar cache actual
clear_cache()

# El próximo get_train_test_split() creará un nuevo split
data = get_train_test_split()
```

## 🔄 Flujo de Trabajo

### Notebook 2: Modelado Base

```python
# 1. Obtener split
data = get_train_test_split()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# 2. Entrenar SOLO con datos de train
model.fit(X_train, y_train)

# 3. Evaluar SOLO con datos de test (al final)
accuracy = model.score(X_test, y_test)
```

### Notebook 2.5: Optimización de Hiperparámetros

```python
# 1. Obtener el MISMO split
data = get_train_test_split()
X_train, y_train = data['X_train'], data['y_train']

# 2. Optimizar SOLO con datos de train (usando CV interno)
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 3. Evaluar SOLO con datos de test (al final)
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
```

### Notebook 3: Feature Engineering

```python
# 1. Obtener el MISMO split
data = get_train_test_split()
X_train_original = data['X_train']

# 2. Feature engineering SOLO en datos de train
X_train_engineered = create_features(X_train_original)

# 3. Aplicar MISMAS transformaciones a test
X_test_original = data['X_test']
X_test_engineered = create_features(X_test_original)

# 4. Entrenar y evaluar
model.fit(X_train_engineered, y_train)
accuracy = model.score(X_test_engineered, y_test)
```

## ✅ Ventajas del Sistema

1. **Consistencia**: Todos los notebooks usan exactamente el mismo split
2. **Prevención de data leakage**: Los datos de test nunca se usan para entrenamiento
3. **Reproducibilidad**: Random state fijo + cache garantiza resultados consistentes
4. **Simplicidad**: Una sola línea de código para obtener todos los datos
5. **Cache automático**: El split se guarda y reutiliza automáticamente
6. **Trazabilidad**: Metadata completa del split disponible en `split_info`

## 🔍 Verificar Consistencia

Para verificar que todos los notebooks usan el mismo split:

```python
from data_split import print_split_summary

data = get_train_test_split()
print_split_summary(data)
```

Output esperado:

```
================================================================================
                          RESUMEN DEL SPLIT TRAIN/TEST
================================================================================

📊 Tamaño del Dataset:
   - Total: 2200 muestras
   - Train: 1760 muestras (80.0%)
   - Test: 440 muestras (20.0%)

🔢 Características:
   - Features: 7
   - Clases: 22

⚙️  Configuración:
   - Random state: 42
   - Estratificado: Sí
```

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'data_split'"

**Solución**: Asegúrate de agregar el path al sys.path:

```python
import sys
sys.path.append('../src')  # Ajustar según ubicación del notebook
```

### Error: Cache corrupto

**Solución**: Regenerar el cache:

```python
from data_split import clear_cache
clear_cache()
data = get_train_test_split()
```

### Los notebooks muestran diferentes resultados

**Verificación**: Confirmar que todos usan el mismo split:

```python
# En cada notebook:
print(f"Train samples: {len(X_train)}")  # Debe ser 1760
print(f"Test samples: {len(X_test)}")     # Debe ser 440
print(f"First train index: {X_train.index[0]}")  # Debe ser igual en todos
```

## 📝 Notas Importantes

1. **El split se hace UNA VEZ**: El primer notebook que se ejecute creará el cache
2. **Cache compartido**: El archivo `train_test_split_cache.pkl` es usado por todos los notebooks
3. **No modificar X_train/X_test manualmente**: Usar copias si necesitas transformaciones
4. **Feature engineering**: Aplicar transformaciones a train primero, luego a test con los mismos parámetros

## 📚 Referencias

- **Archivo fuente**: `src/data_split.py`
- **Documentación sklearn**: [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- **Data leakage**: [Kaggle Guide](https://www.kaggle.com/code/alexisbcook/data-leakage)

---

**Creado para**: Proyecto Final - Machine Learning Aplicado
**Universidad**: EAFIT
**Fecha**: Octubre 2025

