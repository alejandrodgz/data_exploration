import matplotlib.pyplot as plt
import os

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')

def analyze_categorical(df, cat_col, target_col, save_plots=True, output_dir='plots'):
    """Análisis completo de variable categórica"""
    
    if save_plots:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribución de categorías
    ax1 = axes[0, 0]
    counts = df[cat_col].value_counts()
    ax1.bar(counts.index, counts.values, color=plt.cm.Set3(range(len(counts))))
    ax1.set_title(f'Distribución de {cat_col}')
    ax1.set_xlabel(cat_col)
    ax1.set_ylabel('Frecuencia')
    ax1.tick_params(axis='x', rotation=45)
    
    # Agregar porcentajes
    for i, (idx, val) in enumerate(counts.items()):
        ax1.text(i, val, f'{val}\n({val/len(df)*100:.1f}%)', 
                ha='center', va='bottom')
    
    # 2. Pie chart
    ax2 = axes[0, 1]
    ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=plt.cm.Set3(range(len(counts))))
    ax2.set_title(f'Proporción de {cat_col}')
    
    # 3. Boxplot por categoría
    ax3 = axes[1, 0]
    df.boxplot(column=target_col, by=cat_col, ax=ax3)
    ax3.set_title(f'{target_col} por {cat_col}')
    ax3.set_xlabel(cat_col)
    ax3.set_ylabel(target_col)
    plt.sca(ax3)
    plt.xticks(rotation=45)
    
    # 4. Estadísticas por categoría
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_by_cat = df.groupby(cat_col)[target_col].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    
    table_data = []
    for idx, row in stats_by_cat.iterrows():
        table_data.append([idx, f"{row['count']:.0f}", 
                          f"${row['mean']:,.0f}", 
                          f"${row['median']:,.0f}",
                          f"${row['std']:,.0f}"])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Categoría', 'N', 'Media', 'Mediana', 'Desv.Est.'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.15, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Colorear encabezados
    for i in range(5):
        table[(0, i)].set_facecolor('#40E0D0')
        table[(0, i)].set_text_props(weight='bold')
    
    plt.suptitle(f'Análisis de Variable Categórica: {cat_col}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_plots:
        # Save plot with descriptive filename
        plot_filename = f'categorical_analysis_{cat_col.replace(" ", "_").lower()}_vs_{target_col.replace(" ", "_").lower()}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"📊 Análisis categórico de '{cat_col}' vs '{target_col}' guardado en: {plot_path}")
    else:
        plt.show()
