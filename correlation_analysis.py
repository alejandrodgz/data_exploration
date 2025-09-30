import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')

def correlation_analysis(df, target_col='temperature', save_plots=True, output_dir='plots'):
    """Análisis de correlación con múltiples métricas"""
    
    if save_plots:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Correlación de Pearson
    corr_pearson = df[numeric_cols].corr(method='pearson')
    mask = np.triu(np.ones_like(corr_pearson), k=1)
    sns.heatmap(corr_pearson, mask=mask, annot=True, fmt='.2f', 
               cmap='coolwarm', center=0, ax=axes[0],
               vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    axes[0].set_title('Correlación de Pearson (Lineal)')
    
    # 2. Correlación de Spearman  
    corr_spearman = df[numeric_cols].corr(method='spearman')
    sns.heatmap(corr_spearman, mask=mask, annot=True, fmt='.2f',
               cmap='coolwarm', center=0, ax=axes[1],
               vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    axes[1].set_title('Correlación de Spearman (Monotónica)')
    
    # 3. Correlación con variable objetivo
    if target_col in numeric_cols:
        target_corr = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        colors = ['green' if x > 0 else 'red' for x in target_corr.values]
        target_corr.plot(kind='barh', ax=axes[2], color=colors)
        axes[2].set_title(f'Correlación con {target_col}')
        axes[2].set_xlabel('Coeficiente de Correlación')
        axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, f'Variable objetivo "{target_col}" no encontrada\nen columnas numéricas', 
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        axes[2].set_title('Variable objetivo no válida')
    
    plt.suptitle('Análisis de Correlación Multi-métrica', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_plots:
        # Save plot with descriptive filename
        plot_filename = f'correlation_analysis_multimetric.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"📊 Análisis de correlación multi-métrica guardado en: {plot_path}")
    else:
        plt.show()
    
    # Tabla de correlaciones importantes
    if target_col in numeric_cols:
        print(f"\n🔗 Correlaciones Significativas con {target_col}:")
        print("=" * 50)
        significant_corr = target_corr[abs(target_corr) > 0.1].drop(target_col)
        if len(significant_corr) > 0:
            for var, corr in significant_corr.items():
                strength = "Fuerte" if abs(corr) > 0.5 else "Moderada" if abs(corr) > 0.3 else "Débil"
                direction = "Positiva" if corr > 0 else "Negativa"
                print(f"  • {var:20s}: {corr:+.3f} ({strength} {direction})")
        else:
            print(f"  No se encontraron correlaciones significativas (>0.1) con {target_col}")
    else:
        print(f"\n❌ Variable objetivo '{target_col}' no encontrada en columnas numéricas.")
        print(f"Columnas numéricas disponibles: {', '.join(numeric_cols)}")