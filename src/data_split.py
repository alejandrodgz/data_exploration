"""
Módulo para manejo centralizado del split Train/Test del dataset de cultivos.

Este módulo garantiza que todos los notebooks usen exactamente el mismo split
de datos, siguiendo las mejores prácticas de ML:

1. Split inicial único (80/20) con random_state fijo
2. Datos de entrenamiento para:
   - Feature engineering
   - Entrenamiento de modelos
   - Optimización de hiperparámetros
   - Validación cruzada
3. Datos de prueba SOLO para evaluación final

Esto previene data leakage y garantiza evaluaciones justas.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle

from data_loader import load_crop_data

# Configuración global
RANDOM_STATE = 42
TEST_SIZE = 0.2
CACHE_DIR = Path("../models")


def get_train_test_split(use_cache=True, random_state=RANDOM_STATE, test_size=TEST_SIZE):
    """
    Obtiene el split Train/Test del dataset de cultivos.

    Este split se realiza UNA SOLA VEZ y se cachea para garantizar que todos
    los notebooks usen exactamente los mismos datos de train/test.

    Args:
        use_cache (bool): Si True, intenta cargar el split desde cache.
                         Si False, crea un nuevo split. Default: True
        random_state (int): Semilla para reproducibilidad. Default: 42
        test_size (float): Proporción del test set (0.0 a 1.0). Default: 0.2

    Returns:
        dict: Diccionario con las siguientes llaves:
            - 'X_train': Features de entrenamiento (DataFrame)
            - 'X_test': Features de prueba (DataFrame)
            - 'y_train': Target de entrenamiento (array codificado)
            - 'y_test': Target de prueba (array codificado)
            - 'y_train_labels': Target de entrenamiento (etiquetas originales)
            - 'y_test_labels': Target de prueba (etiquetas originales)
            - 'label_encoder': Codificador de etiquetas entrenado
            - 'feature_names': Lista de nombres de features originales
            - 'class_names': Lista de nombres de clases
            - 'split_info': Información sobre el split

    Example:
        >>> from data_split import get_train_test_split
        >>> data = get_train_test_split()
        >>> X_train = data['X_train']
        >>> y_train = data['y_train']
        >>> print(f"Train samples: {len(X_train)}")
    """

    # Crear directorio de cache si no existe
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "train_test_split_cache.pkl"

    # Intentar cargar desde cache
    if use_cache and cache_file.exists():
        print("="*80)
        print("CARGANDO SPLIT TRAIN/TEST DESDE CACHE".center(80))
        print("="*80)
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            print(f"\n✅ Split cargado desde cache: {cache_file.name}")
            print(f"\n📊 Información del Split:")
            print(f"   - Train samples: {len(data['X_train'])} ({len(data['X_train'])/len(data['X_train'] + data['X_test'])*100:.1f}%)")
            print(f"   - Test samples: {len(data['X_test'])} ({len(data['X_test'])/len(data['X_train'] + data['X_test'])*100:.1f}%)")
            print(f"   - Features: {data['X_train'].shape[1]}")
            print(f"   - Clases: {len(data['class_names'])}")
            print(f"   - Random state: {data['split_info']['random_state']}")
            print(f"   - Test size: {data['split_info']['test_size']}")

            return data

        except Exception as e:
            print(f"⚠️  Error cargando cache: {e}")
            print(f"   Creando nuevo split...")

    # Crear nuevo split
    print("="*80)
    print("CREANDO NUEVO SPLIT TRAIN/TEST".center(80))
    print("="*80)

    # Cargar datos
    print("\n📥 Cargando dataset...")
    crop_data = load_crop_data()

    if crop_data is None:
        raise ValueError("No se pudo cargar el dataset")

    print(f"✅ Dataset cargado: {crop_data.shape[0]} filas × {crop_data.shape[1]} columnas")

    # Separar features y target
    X = crop_data.drop(columns=["label"])
    y = crop_data["label"]

    print(f"\n📊 Separación de datos:")
    print(f"   - Features (X): {X.shape[1]} columnas")
    print(f"   - Target (y): {y.nunique()} clases únicas")

    # Codificar target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\n🔢 Encoding del target:")
    print(f"   - Clases codificadas: {len(label_encoder.classes_)}")
    print(f"   - Ejemplo: '{label_encoder.classes_[0]}' → 0")

    # Split estratificado
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
        X,
        y_encoded,
        y,  # También guardamos las etiquetas originales
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    print(f"\n✂️  Split estratificado:")
    print(f"   - Train: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   - Test: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   - Random state: {random_state}")
    print(f"   - Estratificado: Sí (mantiene proporción de clases)")

    # Verificar estratificación
    train_class_counts = pd.Series(y_train).value_counts().sort_index()
    test_class_counts = pd.Series(y_test).value_counts().sort_index()

    print(f"\n🔍 Verificación de estratificación:")
    print(f"   - Clases en train: {len(np.unique(y_train))}")
    print(f"   - Clases en test: {len(np.unique(y_test))}")
    print(f"   - Muestras por clase (train): min={train_class_counts.min()}, max={train_class_counts.max()}, avg={train_class_counts.mean():.1f}")
    print(f"   - Muestras por clase (test): min={test_class_counts.min()}, max={test_class_counts.max()}, avg={test_class_counts.mean():.1f}")

    # Crear diccionario con toda la información
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_labels': y_train_labels,
        'y_test_labels': y_test_labels,
        'label_encoder': label_encoder,
        'feature_names': list(X.columns),
        'class_names': list(label_encoder.classes_),
        'split_info': {
            'random_state': random_state,
            'test_size': test_size,
            'train_size': 1 - test_size,
            'n_samples_total': len(X),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': X.shape[1],
            'n_classes': len(label_encoder.classes_),
            'stratified': True
        }
    }

    # Guardar en cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n💾 Split guardado en cache: {cache_file}")
        print(f"   Los próximos notebooks usarán este mismo split automáticamente")
    except Exception as e:
        print(f"\n⚠️  No se pudo guardar cache: {e}")
        print(f"   El split seguirá funcionando, pero no se compartirá entre notebooks")

    print("\n" + "="*80)
    print("✅ Split Train/Test creado exitosamente".center(80))
    print("="*80)

    return data


def clear_cache():
    """
    Elimina el cache del split Train/Test.

    Úsalo cuando quieras regenerar el split con diferentes parámetros.
    """
    cache_file = CACHE_DIR / "train_test_split_cache.pkl"

    if cache_file.exists():
        cache_file.unlink()
        print(f"✅ Cache eliminado: {cache_file}")
        print(f"   El próximo llamado a get_train_test_split() creará un nuevo split")
    else:
        print(f"ℹ️  No hay cache para eliminar")


def print_split_summary(data):
    """
    Imprime un resumen detallado del split Train/Test.

    Args:
        data (dict): Diccionario retornado por get_train_test_split()
    """
    print("="*80)
    print("RESUMEN DEL SPLIT TRAIN/TEST".center(80))
    print("="*80)

    info = data['split_info']

    print(f"\n📊 Tamaño del Dataset:")
    print(f"   - Total: {info['n_samples_total']} muestras")
    print(f"   - Train: {info['n_samples_train']} muestras ({info['train_size']*100:.1f}%)")
    print(f"   - Test: {info['n_samples_test']} muestras ({info['test_size']*100:.1f}%)")

    print(f"\n🔢 Características:")
    print(f"   - Features: {info['n_features']}")
    print(f"   - Clases: {info['n_classes']}")

    print(f"\n⚙️  Configuración:")
    print(f"   - Random state: {info['random_state']}")
    print(f"   - Estratificado: {'Sí' if info['stratified'] else 'No'}")

    print(f"\n📝 Features originales:")
    print(f"   {data['feature_names']}")

    print(f"\n🌾 Clases (primeras 10):")
    print(f"   {data['class_names'][:10]}")
    if len(data['class_names']) > 10:
        print(f"   ... y {len(data['class_names']) - 10} más")

    print("\n" + "="*80)


# Ejemplo de uso
if __name__ == "__main__":
    # Obtener split
    data = get_train_test_split(use_cache=True)

    # Imprimir resumen
    print_split_summary(data)

    # Ejemplo de uso en un notebook
    print("\n📝 Ejemplo de uso en notebooks:")
    print("""
    # En cualquier notebook (2, 2.5, 3):
    from data_split import get_train_test_split

    # Cargar datos (automáticamente usa el mismo split)
    data = get_train_test_split()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    label_encoder = data['label_encoder']

    # Entrenar modelo SOLO con X_train, y_train
    model.fit(X_train, y_train)

    # Evaluar SOLO con X_test, y_test (al final)
    accuracy = model.score(X_test, y_test)
    """)

