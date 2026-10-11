
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

def load_and_split_data(data_path: str, random_state: int):
    
    """
    Carga los datos crudos, aplica limpieza estructural (eliminación de duplicados y 
    columnas sin valor predictivo) y divide el conjunto en entrenamiento y prueba 
    mediante estratificación. 
    
    Retorna los subconjuntos (X_train, X_test, y_train, y_test) junto con un 
    diccionario de metadatos (structural_cleaning_params) para garantizar la 
    trazabilidad del preprocesamiento en MLFlow.
    """

    logger.info(f"   * Cargando datos crudos desde: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Inicializamos el diccionario de parámetros estructurales
    structural_cleaning_params = {
        "deduplication_key": "PassengerId",
        "pclass_casted_to": "category"
    }
    
    duplicados = df.duplicated(subset=['PassengerId']).sum()
    structural_cleaning_params["duplicates_removed"] = int(duplicados)
    
    if duplicados > 0:
        logger.info(f"   * Removiendo {duplicados} registros duplicados.")
        df = df.drop_duplicates(subset=['PassengerId'])
    
    cols_to_drop = ['Cabin', 'Name', 'Ticket', 'PassengerId']
    structural_cleaning_params["dropped_columns"] = ", ".join(cols_to_drop)
    
    df = df.drop(columns=cols_to_drop)
    df['Pclass'] = df['Pclass'].astype('category')
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    logger.success("   ✓ Datos aprovisionados y divididos correctamente.")
    # 2. Retornamos el diccionario al final
    return X_train, X_test, y_train, y_test, structural_cleaning_params