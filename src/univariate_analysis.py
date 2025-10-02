from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def interpretar_nivel_nutriente(var_name, q1, median, q3):
    """
    Interpreta cuartiles usando un mapa de configuración.
    Los cuartiles dividen las 2,200 MUESTRAS en 4 grupos iguales (550 cada uno).
    """
    
    # Mapa de interpretaciones por variable
    interpretaciones = {
        'N': {
            'unidad': 'kg/ha',
            'nombre': 'nitrógeno',
            'niveles': ['bajo', 'moderadamente bajo', 'moderadamente alto', 'alto'],
            'decimales': 0  # Sin decimales
        },
        'P': {
            'unidad': 'kg/ha',
            'nombre': 'fósforo',
            'niveles': ['bajo', 'moderadamente bajo', 'moderadamente alto', 'alto'],
            'decimales': 0
        },
        'K': {
            'unidad': 'kg/ha',
            'nombre': 'potasio',
            'niveles': ['bajo', 'moderadamente bajo', 'moderadamente alto', 'alto'],
            'decimales': 0
        },
        'temperature': {
            'unidad': '°C',
            'nombre': 'temperatura',
            'niveles': ['fría', 'templada', 'cálida', 'muy cálida'],
            'contexto': ['clima frío', 'clima templado moderado', 'clima cálido', 'clima tropical/subtropical'],
            'decimales': 1  # Con 1 decimal
        },
        'humidity': {
            'unidad': '%',
            'nombre': 'humedad relativa',
            'niveles': ['baja', 'media-baja', 'media-alta', 'alta'],
            'contexto': ['condiciones de baja humedad', 'humedad moderadamente baja', 
                        'humedad moderadamente alta', 'condiciones de alta humedad'],
            'decimales': 0
        },
        'ph': {
            'unidad': '',
            'nombre': 'pH',
            'niveles': ['ácido', 'ligeramente ácido', 'neutro', 'alcalino'],
            'contexto': ['suelos ácidos', 'suelos ligeramente ácidos', 
                        'suelos neutros a ligeramente alcalinos', 'suelos alcalinos'],
            'decimales': 1  # Con 1 decimal
        },
        'rainfall': {
            'unidad': 'mm',
            'nombre': 'precipitación',
            'niveles': ['baja', 'media-baja', 'media-alta', 'alta'],
            'contexto': ['zona con baja precipitación anual', 'precipitación moderadamente baja',
                        'precipitación moderadamente alta', 'zona con alta precipitación anual'],
            'decimales': 0
        }
    }
    
    # Configuración por defecto para variables no mapeadas
    config_default = {
        'unidad': '',
        'nombre': var_name,
        'niveles': ['bajo', 'medio-bajo', 'medio-alto', 'alto'],
        'contexto': ['valores bajos', 'valores moderadamente bajos', 
                    'valores moderadamente altos', 'valores altos'],
        'decimales': 2
    }
    
    # Obtener configuración de la variable
    config = interpretaciones.get(var_name, config_default)
    decimales = config['decimales']
    
    # Valores de los rangos
    rangos = [
        (0, q1),
        (q1, median),
        (median, q3),
        (q3, None)  # None para indicar infinito
    ]
    
    # Construir diccionario de resultados
    resultado = {}
    keys = ['bajo', 'medio_bajo', 'medio_alto', 'alto']
    
    for i, (key, nivel, (min_val, max_val)) in enumerate(zip(keys, config['niveles'], rangos)):
        # Formatear valores según los decimales configurados
        if decimales == 0:
            formato_min = f"{int(min_val)}"
            formato_max = f"{int(max_val)}" if max_val is not None else None
        else:
            formato_min = f"{min_val:.{decimales}f}"
            formato_max = f"{max_val:.{decimales}f}" if max_val is not None else None
        
        # Formatear rango según si es el último
        if max_val is None:
            rango_str = f"> {formato_min}"
        else:
            rango_str = f"{formato_min}-{formato_max}"
        
        # Construir descripción
        unidad = f" {config['unidad']}" if config['unidad'] else ""
        
        descripcion = f"{rango_str}{unidad}"
        
        # Agregar contexto si existe
        if 'contexto' in config:
            descripcion += f" ({config['contexto'][i]})"
        
        resultado[key] = descripcion
    
    return resultado


def graficar_variable(df, var_name, skewness, has_outliers):
    """
    Genera gráficos relevantes según características de la variable:
    - Histograma con KDE: Siempre (muestra distribución)
    - Boxplot: Solo si hay outliers o asimetría significativa
    """
    
    # Determinar número de subplots necesarios
    if has_outliers or abs(skewness) > 0.5:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]  # Convertir a lista para consistencia
    
    # 1. HISTOGRAMA CON KDE (siempre se muestra)
    ax1 = axes[0]
    
    # Histograma
    ax1.hist(df[var_name], bins=30, edgecolor='black', alpha=0.6, 
             color='steelblue', density=True, label='Frecuencia')
    
    # KDE (Kernel Density Estimation)
    df[var_name].plot(kind='density', ax=ax1, color='red', 
                      linewidth=2, label='Densidad estimada')
    
    # Líneas de tendencia central
    mean_val = df[var_name].mean()
    median_val = df[var_name].median()
    
    ax1.axvline(mean_val, color='darkred', linestyle='--', 
                linewidth=2, label=f'Media ({mean_val:.2f})')
    ax1.axvline(median_val, color='darkgreen', linestyle='--', 
                linewidth=2, label=f'Mediana ({median_val:.2f})')
    
    ax1.set_title(f'{var_name} - Distribución', fontsize=12, fontweight='bold')
    ax1.set_xlabel(var_name, fontsize=10)
    ax1.set_ylabel('Densidad', fontsize=10)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. BOXPLOT (solo si hay outliers o asimetría significativa)
    if has_outliers or abs(skewness) > 0.5:
        ax2 = axes[1]
        
        bp = ax2.boxplot(df[var_name], vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(color='blue', linewidth=1.5),
                         capprops=dict(color='blue', linewidth=1.5),
                         flierprops=dict(marker='o', markerfacecolor='red', 
                                       markersize=6, alpha=0.5))
        
        # Añadir etiquetas de cuartiles
        Q1 = df[var_name].quantile(0.25)
        Q2 = df[var_name].quantile(0.50)
        Q3 = df[var_name].quantile(0.75)
        
        ax2.text(1.15, Q1, f'Q1: {Q1:.2f}', fontsize=9, va='center')
        ax2.text(1.15, Q2, f'Q2: {Q2:.2f}', fontsize=9, va='center')
        ax2.text(1.15, Q3, f'Q3: {Q3:.2f}', fontsize=9, va='center')
        
        if has_outliers:
            titulo = f'{var_name} - Boxplot (Detecta Outliers)'
        else:
            titulo = f'{var_name} - Boxplot (Asimetría Detectada)'
        
        ax2.set_title(titulo, fontsize=12, fontweight='bold')
        ax2.set_ylabel(var_name, fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def analyze_variable(df, var_name):
    """Análisis univariado completo de una variable numérica"""
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS UNIVARIADO: {var_name}")
    print(f"{'='*70}")
    
    # 1. FORMA DE LA DISTRIBUCIÓN
    skewness = df[var_name].skew()
    kurt = df[var_name].kurtosis()
    
    if abs(skewness) < 0.5:
        skew_interp = "Simétrica"
    elif skewness > 0:
        skew_interp = "Asimétrica positiva (cola derecha)"
    else:
        skew_interp = "Asimétrica negativa (cola izquierda)"
    
    if abs(kurt) < 0.5:
        kurt_interp = "Normal (mesocúrtica)"
    elif kurt > 0:
        kurt_interp = "Picos altos (leptocúrtica)"
    else:
        kurt_interp = "Aplastada (platicúrtica)"
    
    tabla_forma = pd.DataFrame({
        'Métrica': ['Asimetría (Skewness)', 'Curtosis'],
        'Valor': [f'{skewness:.3f}', f'{kurt:.3f}'],
        'Interpretación': [skew_interp, kurt_interp]
    })
    
    print("1. Forma de la Distribución:")
    display(tabla_forma)
    print()

    # 2. OUTLIERS Y NORMALIDAD
    Q1 = df[var_name].quantile(0.25)
    Q3 = df[var_name].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[var_name] < Q1 - 1.5*IQR) | (df[var_name] > Q3 + 1.5*IQR)]
    
    # Test para la validar si la muestra es normal
    _, p_value = stats.normaltest(df[var_name])

    norm_interp = "Normal" if p_value > 0.05 else "NO normal"
    
    tabla_calidad = pd.DataFrame({
        'Aspecto': ['Outliers', 'Test de Normalidad'],
        'Resultado': [f'{len(outliers)} ({len(outliers)/len(df)*100:.2f}%)', 
                      f'p-value = {p_value:.4f}'],
        'Interpretación': ['Sin outliers' if len(outliers)==0 else 'Outliers detectados',
                          norm_interp]
    })
    
    print("2. Calidad de Datos:")
    display(tabla_calidad)
    print()
    
    # 3. RESUMEN EJECUTIVO
    stats_desc = df[var_name].describe()
    
    mean_val = stats_desc['mean']
    median_val = stats_desc['50%']
    std_val = stats_desc['std']
    min_val = stats_desc['min']
    max_val = stats_desc['max']
    
    if mean_val > median_val * 1.1:
        centralidad_impl = "Valores altos presentes"
    elif mean_val < median_val * 0.9:
        centralidad_impl = "Valores bajos dominan"
    else:
        centralidad_impl = "Distribución equilibrada"
    
    cv = (std_val / mean_val) * 100
    if cv > 50:
        dispersion_impl = f"Muy alta (CV={cv:.1f}%)"
    elif cv > 30:
        dispersion_impl = f"Alta (CV={cv:.1f}%)"
    elif cv > 15:
        dispersion_impl = f"Moderada (CV={cv:.1f}%)"
    else:
        dispersion_impl = f"Baja (CV={cv:.1f}%)"
    
    if abs(skewness) < 0.5:
        forma_impl = "Distribución simétrica"
    elif skewness > 0:
        forma_impl = "Mayoría valores bajos"
    else:
        forma_impl = "Mayoría valores altos"
    
    outliers_size = len(outliers)
    has_outliers = outliers_size==0
    tabla_resumen = pd.DataFrame({
        'Aspecto': ['Rango', 'Centralidad', 'Dispersión', 'Forma', 
                    'Calidad', 'Normalidad'],
        'Hallazgo': [f'{min_val:.2f} - {max_val:.2f}',
                     f'μ={mean_val:.2f}, M={median_val:.2f}',
                     f'σ={std_val:.2f}',
                     f'Skew={skewness:.2f}',
                     f'{outliers_size} outliers',
                     norm_interp],
        'Implicación': [dispersion_impl,
                        centralidad_impl,
                        dispersion_impl,
                        forma_impl,
                        'Datos confiables' if has_outliers else 'Revisar outliers',
                        'Escalado necesario' if p_value<0.05 else 'Datos normales']
    })
    
    print("3. Resumen Ejecutivo:")
    display(tabla_resumen)
    print()
    
    # 5. INTERPRETACIÓN POR CUARTILES
    niveles = interpretar_nivel_nutriente(var_name, Q1, median_val, Q3)
    
    tabla_cuartiles = pd.DataFrame({
        'Nivel': ['Bajo (0-25%)', 'Medio-Bajo (25-50%)', 
                  'Medio-Alto (50-75%)', 'Alto (75-100%)'],
        'Descripción': list(niveles.values())
    })
    
    print("5. Interpretación por Cuartiles:")
    display(tabla_cuartiles)
    print()
    
    # Generar gráficos
    graficar_variable(df, var_name, skewness, has_outliers)
    
    print(f"\n{'='*70}\n")
    
    return {
        'variable': var_name,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurt,
        'outliers': outliers_size,
        'normal': p_value > 0.05,
        'p_value': p_value
    }

def analyze_numeric_values(df, num_vars):
    """Análisis univariado con estadísticas robustas"""

    resultados = []
    
    for var in num_vars:
        resultado = analyze_variable(df, var)
        resultados.append(resultado)

    # Crear tabla comparativa final
    print("\n" + "="*90)
    print("📊 TABLA COMPARATIVA: TODAS LAS VARIABLES NUMÉRICAS")
    print("="*90 + "\n")

    df_resumen = pd.DataFrame(resultados)
    df_resumen = df_resumen.round(2)

    display(df_resumen)

    print("\n" + "="*90)
    print("CONCLUSIONES GENERALES:")
    print("="*90)
    print(f"✓ Variables analizadas: {len(num_vars)}")
    print(f"✓ Variables normales: {df_resumen['normal'].sum()} de {len(num_vars)}")
    print(f"✓ Variables con outliers: {(df_resumen['outliers'] > 0).sum()} de {len(num_vars)}")
    print("="*90 + "\n")
