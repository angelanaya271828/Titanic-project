
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

def build_preprocessor():

    """
    Construye un ColumnTransformer de scikit-learn que define las 
    transformaciones (Pipeline). Imputa y escala variables numéricas, y aplica 
    imputación modal y codificación One-Hot a variables categóricas.
    
    Retorna el objeto preprocesador no ajustado y un diccionario de metadatos 
    (data_cleaning_params) con las reglas lógicas exactas utilizadas, mitigando 
    la deuda técnica de configuración oculta.
    """

    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    
    # 1. Definimos los parámetros del feature engineering
    data_cleaning_params = {
        "imputation_numeric": "median",
        "scaling_numeric": "RobustScaler",
        "imputation_categorical": "most_frequent",
        "encoding_categorical": "OneHotEncoder(drop='first')"
    }
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # 2. Retornamos el preprocesador y su metadata
    return preprocessor, data_cleaning_params
