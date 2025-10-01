import os
import pandas as pd
from pathlib import Path
import kagglehub

def load_crop_data(dataset_dir="datasets", force_download=False):
    """
    Descarga y carga el dataset de crop desde Kaggle usando kagglehub.
    
    Args:
        dataset_dir (str): Directorio base donde guardar los datasets. Default: 'datasets'
        force_download (bool): Forzar descarga incluso si ya existe el archivo. Default: False
    
    Returns:
        pandas.DataFrame: Dataset de crop cargado, o None si hay error
        
    Raises:
        FileNotFoundError: Si no se puede encontrar el archivo después de la descarga
        Exception: Para otros errores durante la descarga o carga
    """
    try:
        # Configurar rutas
        data_path = Path(dataset_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar cache de kagglehub
        os.environ['KAGGLEHUB_CACHE'] = str(data_path.absolute())
        
        # Buscar archivos CSV existentes en la cache antes de descargar
        existing_csv_files = list(data_path.rglob("*Crop_recommendation*.csv"))
        if not existing_csv_files:
            # Buscar cualquier archivo CSV que contenga "crop" en el nombre
            existing_csv_files = list(data_path.rglob("*crop*.csv"))
        
        # Verificar si ya existe el archivo y no se fuerza la descarga
        if existing_csv_files and not force_download:
            csv_file = existing_csv_files[0]  # Usar el primer archivo encontrado
            print(f"📁 Cargando datos desde caché local: {csv_file.name}")
            return pd.read_csv(csv_file)
        
        # Descargar dataset desde Kaggle
        print("📥 Descargando dataset desde Kaggle...")
        download_path = kagglehub.dataset_download("madhuraatmarambhagat/crop-recommendation-dataset")
        #download_path = kagglehub.dataset_download("samuelotiattakorah/agriculture-crop-yield")
        print(f"✅ Dataset descargado en: {download_path}")
        
        # Buscar el archivo CSV en la estructura descargada
        csv_files = list(Path(download_path).rglob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No se encontró ningún archivo CSV en el dataset descargado")
        
        # Usar el primer archivo CSV encontrado (o buscar específicamente Crop_recommendation.csv)
        csv_file = None
        for file in csv_files:
            if "Crop_recommendation" in file.name or "crop" in file.name.lower():
                csv_file = file
                break
        
        if csv_file is None:
            csv_file = csv_files[0]  # Usar el primer CSV si no se encuentra uno específico
            print(f"⚠️  Usando archivo: {csv_file.name}")
        
        # Cargar y retornar el dataset
        print(f"📊 Cargando datos desde: {csv_file.name}")
        df = pd.read_csv(csv_file)
        
        print(f"✅ Dataset cargado exitosamente: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        return df
        
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {type(e).__name__}: {e}")
        return None
    
# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos de crop
    crop_data = load_crop_data()
    
    if crop_data is not None:
        print(f"\n📊 Dataset cargado: {crop_data.shape[0]:,} filas × {crop_data.shape[1]} columnas")
        print(f"📋 Columnas: {list(crop_data.columns)}")
        print("\n🔍 Primeras 5 filas:")
        print(crop_data.head())
    else:
        print("❌ No se pudo cargar el dataset")