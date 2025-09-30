import matplotlib.pyplot as plt
import pandas as pd
import os

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')

# Análisis detallado de valores faltantes
def analyze_missing_values(df, save_plots=True, output_dir='plots'):
    """Análisis completo de valores faltantes"""
    missing_df = pd.DataFrame({
        'Columna': df.columns,
        'Valores_Faltantes': df.isnull().sum(),
        'Porcentaje': (df.isnull().sum() / len(df)) * 100,
        'Tipo_Dato': df.dtypes
    })
    
    missing_df = missing_df[missing_df['Valores_Faltantes'] > 0].sort_values(
        'Porcentaje', ascending=False
    )
    
    if len(missing_df) > 0:
        print(f"\n⚠️  VALORES FALTANTES DETECTADOS:")
        print("=" * 60)
        for idx, row in missing_df.iterrows():
            print(f"   • {row['Columna']:<15}: {row['Valores_Faltantes']:>4} valores ({row['Porcentaje']:5.1f}%)")
        
        if save_plots:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Visualización
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
            
            # Save plot
            plot_path = os.path.join(output_dir, 'missing_values_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            print(f"\n📊 Gráfico guardado en: {plot_path}")
        
        return missing_df
    else:
        print("\n✅ No hay valores faltantes en el dataset")
        return None
