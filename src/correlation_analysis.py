import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import math
import sys

def _running_in_notebook():
    """Detect if running inside a Jupyter/Kernel environment (VSCode, JupyterLab, etc.)."""
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

# No cambiamos backend aquí: se decide dinámicamente en la función.

def correlation_analysis(df, target_col='temperature', save_plots=True, output_dir='plots', notebook_mode=None):
    """Análisis de correlación con múltiples métricas.

    Ajuste solicitado: el tercer gráfico ya no muestra solo la correlación contra
    `target_col`, sino que genera un gráfico de barras por CADA variable numérica
    (correlación de Pearson respecto al resto) y todos se guardan en el MISMO PNG
    junto con los heatmaps de Pearson y Spearman.
    Parámetros adicionales
    ----------------------
    notebook_mode : bool | None
        Forzar comportamiento de notebook (True) o terminal (False). Si es None se detecta automáticamente.
        - Modo terminal: se guarda el PNG (si save_plots=True) y solo se imprime ruta.
        - Modo notebook: se muestra inline la figura; si save_plots=True también se guarda.
    """

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    if notebook_mode is None:
        notebook_mode = _running_in_notebook()

    # Solo forzar backend no interactivo si estamos en modo terminal (para evitar bloquear)
    if not notebook_mode:
        try:
            current_backend = plt.get_backend().lower()
            if 'agg' not in current_backend:
                plt.switch_backend('Agg')
        except Exception:
            pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("❌ No hay columnas numéricas para analizar.")
        return

    # Correlation matrices
    corr_pearson = df[numeric_cols].corr(method='pearson')
    corr_spearman = df[numeric_cols].corr(method='spearman')
    mask = np.triu(np.ones_like(corr_pearson), k=1)

    # Total subplots: 2 heatmaps + one bar plot per variable
    n_vars = len(numeric_cols)
    total_plots = 2 + n_vars
    n_cols = 3  # layout ancho fijo razonable
    n_rows = math.ceil(total_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    # 1. Pearson heatmap
    sns.heatmap(corr_pearson, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=axes[0], vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    axes[0].set_title('Correlación de Pearson (Lineal)')

    # 2. Spearman heatmap
    sns.heatmap(corr_spearman, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=axes[1], vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    axes[1].set_title('Correlación de Spearman (Monotónica)')

    # 3+. Bar plots for each variable (Pearson correlations sorted desc)
    for i, var in enumerate(numeric_cols):
        ax = axes[2 + i]
        series = corr_pearson[var].sort_values(ascending=False)
        colors = ['green' if v > 0 else 'red' for v in series.values]
        series.plot(kind='barh', ax=ax, color=colors)
        ax.set_title(f"Correlación con {var}")
        ax.set_xlabel('Coeficiente (Pearson)')
        ax.axvline(0, color='black', lw=0.6)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.margins(y=0.02)

    # Hide unused axes if any
    for j in range(2 + n_vars, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Análisis de Correlación: Heatmaps + Barras por Variable', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plot_filename = 'correlation_analysis_multimetric.png'
    plot_path = os.path.join(output_dir, plot_filename)
    if save_plots:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Análisis de correlación (heatmaps + barras) guardado en: {plot_path}")

    if notebook_mode:
        # Mostrar inline en notebook (aunque se haya guardado)
        try:
            from IPython.display import display  # type: ignore
            display(fig)
        except Exception:
            # Fallback a plt.show()
            plt.show()
    else:
        # En modo terminal solo mostramos si no se pidió guardar
        if not save_plots:
            plt.show()

    plt.close(fig)

    # Resumen completo de correlaciones (triángulo superior) con clasificación
    print("\n🔗 Todas las correlaciones (Pearson) entre pares de variables:")
    print("=" * 70)
    print(f"{'Variable A':15s} {'Variable B':15s} {'r':>8s}  Fuerza  Dirección")
    print("-" * 70)
    for a_idx in range(len(numeric_cols)):
        for b_idx in range(a_idx + 1, len(numeric_cols)):
            a = numeric_cols[a_idx]
            b = numeric_cols[b_idx]
            v = corr_pearson.iloc[a_idx, b_idx]
            abs_v = abs(v)
            if abs_v > 0.7:
                strength = 'Muy Fuerte'
            elif abs_v > 0.5:
                strength = 'Fuerte'
            elif abs_v > 0.3:
                strength = 'Moderada'
            elif abs_v > 0.1:
                strength = 'Débil'
            else:
                strength = 'Muy Débil'
            direction = 'Positiva' if v > 0 else 'Negativa' if v < 0 else 'Nula'
            print(f"{a:15s} {b:15s} {v:+8.3f}  {strength:10s} {direction}")
