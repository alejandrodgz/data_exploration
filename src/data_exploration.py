import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from packaging import version
import sklearn
import kagglehub
from data_loader import load_crop_data
from analyze_missing_values import analyze_missing_values
from univariate_analysis import univariate_analysis
from analyze_categorical import analyze_categorical
from correlation_analysis import correlation_analysis
from detect_outliers import detect_outliers

# # Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_palette("husl")

# # Configuración de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)



# Load crop data using the improved function
crop_data = load_crop_data()

if crop_data is not None:
    print(f"✅ Datos cargados: {crop_data.shape[0]:,} filas × {crop_data.shape[1]} columnas")
else:
    print("❌ Error cargando los datos")
    sys.exit(1)


print("=" * 80)
print("INFORMACIÓN GENERAL DEL DATASET".center(80))
print("=" * 80)

# Mostrar primeras filas
print("\n📋 PRIMERAS 5 FILAS:")
print(crop_data.head())

# Información detallada
print("\n" + "=" * 80)
print("ESTRUCTURA DE DATOS".center(80))
print("=" * 80)
crop_data.info()

# Estadísticas descriptivas
print("\n" + "=" * 80)
print("ESTADÍSTICAS DESCRIPTIVAS".center(80))
print("=" * 80)
print("\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
print(crop_data.describe().round(2).T)

# Diccionario de metadatos para crop data
print("\n" + "=" * 80)
print("METADATOS DEL DATASET".center(80))
print("=" * 80)

metadata = {
    'Variable': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'],
    'Tipo': ['Numérica', 'Numérica', 'Numérica', 'Numérica', 'Numérica', 
             'Numérica', 'Numérica', 'Categórica (Target)'],
    'Descripción': [
        'Contenido de Nitrógeno en el suelo (kg/ha)',
        'Contenido de Fósforo en el suelo (kg/ha)', 
        'Contenido de Potasio en el suelo (kg/ha)',
        'Temperatura ambiente (°C)',
        'Humedad relativa (%)',
        'Nivel de pH del suelo',
        'Precipitación pluvial (mm)',
        '🎯 Tipo de cultivo recomendado'
    ],
    'Valores Faltantes': [
        crop_data['N'].isnull().sum(),
        crop_data['P'].isnull().sum(),
        crop_data['K'].isnull().sum(),
        crop_data['temperature'].isnull().sum(),
        crop_data['humidity'].isnull().sum(),
        crop_data['ph'].isnull().sum(),
        crop_data['rainfall'].isnull().sum(),
        crop_data['label'].isnull().sum()
    ],
    'Rango': [
        f"{crop_data['N'].min():.1f} - {crop_data['N'].max():.1f}",
        f"{crop_data['P'].min():.1f} - {crop_data['P'].max():.1f}",
        f"{crop_data['K'].min():.1f} - {crop_data['K'].max():.1f}",
        f"{crop_data['temperature'].min():.1f} - {crop_data['temperature'].max():.1f}",
        f"{crop_data['humidity'].min():.1f} - {crop_data['humidity'].max():.1f}",
        f"{crop_data['ph'].min():.2f} - {crop_data['ph'].max():.2f}",
        f"{crop_data['rainfall'].min():.1f} - {crop_data['rainfall'].max():.1f}",
        f"{crop_data['label'].nunique()} categorías únicas"
    ]
}

df_metadata = pd.DataFrame(metadata)

# Mostrar metadata con formato mejorado
print("\n📋 INFORMACIÓN DETALLADA DE VARIABLES:")
print(df_metadata.to_string(index=False))

# Información adicional sobre las categorías del target
print("\n" + "=" * 80)
print("ANÁLISIS DEL TARGET (LABEL)".center(80))
print("=" * 80)

print(f"\n🌱 TIPOS DE CULTIVOS DISPONIBLES ({crop_data['label'].nunique()} categorías):")
crop_counts = crop_data['label'].value_counts()
for i, (crop, count) in enumerate(crop_counts.items(), 1):
    percentage = (count / len(crop_data)) * 100
    print(f"{i:2d}. {crop:<15} - {count:4d} muestras ({percentage:5.1f}%)")

print(f"\n📊 DISTRIBUCIÓN:")
print(f"   • Cultivo más común: {crop_counts.index[0]} ({crop_counts.iloc[0]} muestras)")
print(f"   • Cultivo menos común: {crop_counts.index[-1]} ({crop_counts.iloc[-1]} muestras)")
print(f"   • Ratio máximo/mínimo: {crop_counts.iloc[0]/crop_counts.iloc[-1]:.1f}x")

missing_analysis = analyze_missing_values(crop_data)
if missing_analysis is not None:
    print(missing_analysis)

for col in ['rainfall', 'humidity', 'temperature']:
    univariate_analysis(crop_data, col, 'temperature')

analyze_categorical(crop_data, 'label', 'temperature')

correlation_analysis(crop_data)

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

X = crop_data[['N','P','K','temperature','humidity','ph','rainfall']].dropna()
vif = pd.DataFrame({
    'feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print(vif.sort_values('VIF', ascending=False))

from sklearn.feature_selection import mutual_info_classif
X = crop_data[['N','P','K','temperature','humidity','ph','rainfall']]
y = crop_data['label']  # tu etiqueta de cultivo
mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
print(pd.Series(mi, index=X.columns).sort_values(ascending=False))


outliers_iqr, outliers_zscore, outliers_iso = detect_outliers(crop_data)