import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def analyze_categorical(df, cat_col):
    # %% [markdown]
    # ###  Variable Categórica (Target)

    # %%
    # Análisis de la variable objetivo (label)
    print(f"\n{'='*60}")
    print(f"ANÁLISIS UNIVARIABLE: {cat_col} (Variable Objetivo)")
    print(f"{'='*60}")

    # Frecuencias
    freq_cultivos = df[cat_col].value_counts()
    freq_relativa = df[cat_col].value_counts(normalize=True)

    print("\n1. Tabla de Frecuencias:")
    tabla_freq = pd.DataFrame({
        'Frecuencia Absoluta': freq_cultivos,
        'Frecuencia Relativa': freq_relativa,
        'Porcentaje': freq_relativa * 100
    })
    print(tabla_freq)

    # Moda
    moda = df[cat_col].mode()
    if freq_cultivos.nunique() == 1:
        print("\n2. Moda (cultivo más frecuente): No hay moda, todas las clases tienen la misma frecuencia.")
    else:
        print(f"\n2. Moda (cultivo más frecuente): {', '.join(moda.astype(str))}")


    # Balance de clases
    ratio_desbalance = freq_cultivos.max() / freq_cultivos.min()
    print(f"\n3. Balance de Clases:")
    print(f"   Cultivo más frecuente: {freq_cultivos.idxmax()} ({freq_cultivos.max()} muestras)")
    print(f"   Cultivo menos frecuente: {freq_cultivos.idxmin()} ({freq_cultivos.min()} muestras)")
    print(f"   Razón de desbalance: {ratio_desbalance:.2f}:1")
    print(f"   Interpretación: ", end="")
    if freq_cultivos.nunique() == 1:
        print("Clases perfectamente balanceadas")
    elif ratio_desbalance < 1.5:
        print("Clases balanceadas")
    elif ratio_desbalance < 3:
        print("Desbalance moderado")
    else:
        print("Desbalance alto - considerar técnicas de balanceo")


    # Entropía (medida de incertidumbre)
    from scipy.stats import entropy
    entropia = entropy(freq_relativa)
    entropia_max = np.log(len(freq_cultivos))
    print(f"\n4. Entropía:")
    print(f"   Entropía: {entropia:.3f}")
    print(f"   Entropía máxima: {entropia_max:.3f}")
    print(f"   Entropía normalizada: {entropia/entropia_max:.3f}")
    print(f"   - Interpretación: ", end="")
    if entropia/entropia_max > 0.9:
        print("Alta diversidad, clases bien distribuidas")
    else:
        print("Baja diversidad, algunas clases dominan")

    # %%
    # Visualizaciones para variable categórica
    plt.close('all') 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico de barras
    freq_cultivos.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Distribución de Cultivos - Gráfico de Barras', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tipo de Cultivo')
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    # Gráfico de pastel (top 10 para legibilidad)
    top_10 = freq_cultivos.head(10)
    otros = freq_cultivos[10:].sum()
    if len(freq_cultivos) <= 20:
        freq_plot = freq_cultivos
    else:
        top_10 = freq_cultivos.head(10)
        otros = freq_cultivos[10:].sum()
        freq_plot = pd.concat([top_10, pd.Series({'Otros': otros})])


    axes[1].pie(freq_plot, labels=freq_plot.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Distribución de Cultivos - Gráfico de Pastel', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
