import matplotlib.pyplot as plt
import pandas as pd
import os
import sys


def _running_in_notebook():
    """Detecta si se está ejecutando dentro de un entorno de notebook (Jupyter / VSCode)."""
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        if 'ipykernel' in sys.modules:
            return True
        shell_name = ip.__class__.__name__
        return shell_name in ('ZMQInteractiveShell', 'Shell')
    except Exception:
        return False


# Análisis detallado de valores faltantes
def analyze_missing_values(df, save_plots=True, output_dir='plots', notebook_mode=None, show=True):
    """Análisis completo de valores faltantes con soporte para notebook.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a analizar.
    save_plots : bool, por defecto True
        Si True guarda la figura en el directorio indicado.
    output_dir : str
        Directorio donde se guardará el PNG.
    notebook_mode : bool | None
        Forzar comportamiento de notebook (True) o modo script/terminal (False).
        Si es None, se detecta automáticamente.
    show : bool
        Si True muestra la figura (en notebook siempre se intentará mostrar). Si False, solo guarda.
    """

    if notebook_mode is None:
        notebook_mode = _running_in_notebook()

    # Solo forzamos backend no interactivo si no estamos en notebook
    if not notebook_mode:
        try:
            if 'agg' not in plt.get_backend().lower():
                plt.switch_backend('Agg')
        except Exception:
            pass

    missing_df = pd.DataFrame({
        'Columna': df.columns,
        'Valores_Faltantes': df.isnull().sum(),
        'Porcentaje': (df.isnull().sum() / len(df)) * 100,
        'Tipo_Dato': df.dtypes
    })

    missing_df = missing_df[missing_df['Valores_Faltantes'] > 0].sort_values(
        'Porcentaje', ascending=False
    )

    if len(missing_df) == 0:
        print("\n✅ No hay valores faltantes en el dataset")
        return None

    print(f"\n⚠️  VALORES FALTANTES DETECTADOS:")
    print("=" * 60)
    for idx, row in missing_df.iterrows():
        print(f"   • {row['Columna']:<15}: {row['Valores_Faltantes']:>4} valores ({row['Porcentaje']:5.1f}%)")

    # Crear directorio si vamos a guardar
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    # Siempre generar figura (para poder mostrarla en notebook aunque save_plots=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de barras
    ax1.bar(missing_df['Columna'], missing_df['Porcentaje'], color='coral')
    ax1.set_xlabel('Columna')
    ax1.set_ylabel('Porcentaje de Valores Faltantes (%)')
    ax1.set_title('Valores Faltantes por Columna')
    ax1.axhline(y=5, color='r', linestyle='--', label='Umbral 5%')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Heatmap de patrones
    import seaborn as sns
    msno_data = df[missing_df['Columna'].tolist()].isnull().astype(int)
    sns.heatmap(msno_data.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax2, vmin=-1, vmax=1)
    ax2.set_title('Correlación de Patrones de Valores Faltantes')

    plt.tight_layout()

    plot_path = None
    if save_plots:
        plot_path = os.path.join(output_dir, 'missing_values_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráfico guardado en: {plot_path}")

    # Mostrar figura según modo
    if notebook_mode and show:
        try:
            from IPython.display import display  # type: ignore
            display(fig)
        except Exception:
            plt.show()
    elif (not notebook_mode) and show and not save_plots:
        # En terminal solo mostramos si no se guardó (para evitar abrir backend interactivo innecesario)
        plt.show()

    plt.close(fig)

    return missing_df
