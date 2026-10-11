
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score, confusion_matrix)

# Importaciones modulares de tu propio paquete
from titanic_module.dataset import load_and_split_data
from titanic_module.features import build_preprocessor

class ModelOrchestrator:

    """
    Clase orquestadora que gestiona el ciclo de vida del entrenamiento del modelo 
    bajo principios de Programación Orientada a Objetos (POO). Centraliza el estado 
    del pipeline, la inferencia de evaluación y la comunicación con MLFlow.
    """

    def __init__(self, args, structural_params, feature_params):

        """
        Inicializa el orquestador inyectando los hiperparámetros definidos por consola 
        (args) y los diccionarios de metadatos originados en los módulos de datos y 
        características. Prepara el estado para la recolección de métricas y artefactos.
        """

        self.args = args
        self.structural_params = structural_params
        self.feature_params = feature_params
        self.metrics = {}
        self.artifact_path = "confusion_matrix.png"
        self.pipeline = None

    def build_and_train(self, X_train, y_train, preprocessor):
        """
        Ensambla dinámicamente el preprocesador y el algoritmo de clasificación 
        seleccionado (XGBoost, Regresión Logística o Árbol de Decisión) en un Pipeline 
        unificado de scikit-learn. Ejecuta el ajuste (fit) empaquetando transformación 
        y entrenamiento en un solo objeto para prevenir el sesgos.
        """

        logger.info(f"   * Construyendo modelo: {self.args.model_type}")

        if self.args.model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=self.args.n_estimators,
                max_depth=self.args.max_depth,
                learning_rate=self.args.learning_rate,
                subsample=self.args.subsample,
                random_state=self.args.random_state,
                eval_metric='logloss'
            )
        elif self.args.model_type == 'logistic_regression':
            model = LogisticRegression(C=self.args.c_reg, random_state=self.args.random_state)
        elif self.args.model_type == 'decision_tree':
            model = DecisionTreeClassifier(max_depth=self.args.max_depth, random_state=self.args.random_state)
        else:
            raise ValueError("   ✖ Modelo no soportado.")
        
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        logger.info("   * Entrenando el pipeline completo...")
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):

        """
        Infiere predicciones y probabilidades sobre el conjunto de datos de validación 
        (hold-out). Calcula métricas críticas de clasificación (Accuracy, Precision, 
        Recall, F1-Score, ROC-AUC, PR-AUC) y renderiza una matriz de confusión exportada 
        como artefacto visual local (.png).
        """

        logger.info("   * Evaluando predicciones...")
        y_pred = self.pipeline.predict(X_test)
        
        if hasattr(self.pipeline.named_steps['classifier'], "predict_proba"):
            y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred 
        
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['precision'] = precision_score(y_test, y_pred)
        self.metrics['recall'] = recall_score(y_test, y_pred)
        self.metrics['f1_score'] = f1_score(y_test, y_pred)
        self.metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        self.metrics['pr_auc'] = average_precision_score(y_test, y_prob)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusión - {self.args.model_type.upper()}')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.savefig(self.artifact_path)
        plt.close()

    def log_experiment(self):

        """
        Instrumenta el registro del ciclo de vida en MLFlow. Persiste la configuración 
        de limpieza (data pipeline), los hiperparámetros específicos del algoritmo, 
        las métricas de desempeño, los artefactos visuales y el modelo empaquetado 
        (serialización del Pipeline completo) para asegurar reproducibilidad absoluta.
        """
        
        logger.info("   * Instrumentando registro en MLFlow...")
        mlflow.set_experiment("Titanic_Modular_Optimization")
        
        with mlflow.start_run(run_name=f"Run_{self.args.model_type}"):
            mlflow.log_param("model_type", self.args.model_type)
            mlflow.log_param("random_state", self.args.random_state)
            
            if self.args.model_type == 'xgboost':
                mlflow.log_param("n_estimators", self.args.n_estimators)
                mlflow.log_param("max_depth", self.args.max_depth)
                mlflow.log_param("learning_rate", self.args.learning_rate)
                mlflow.log_param("subsample", self.args.subsample)
            elif self.args.model_type == 'decision_tree':
                mlflow.log_param("max_depth", self.args.max_depth)
            elif self.args.model_type == 'logistic_regression':
                mlflow.log_param("C_regularization", self.args.c_reg)
            
            mlflow.log_metrics(self.metrics)
            mlflow.log_artifact(self.artifact_path)
            mlflow.sklearn.log_model(self.pipeline, 
                                    f"{self.args.model_type}_pipeline",
                                    skops_trusted_types=[
                                                        "numpy.dtype", 
                                                        "sklearn.tree._tree.Tree", 
                                                        "xgboost.core.Booster", 
                                                        "xgboost.sklearn.XGBClassifier"
                                                    ])
            
        logger.success("   ✓ Experimento registrado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="*-*-* Ejecución modular de pipeline para Titanic. -*-*-* Orquesta la ingesta de datos, preprocesamiento y entrenamiento, garantizando trazabilidad y reproducibilidad."
    )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/raw/Titanic-Dataset.csv",
        help="Ruta relativa o absoluta al conjunto de datos crudo. Fundamental para el versionado de datos."
    )
    
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="logistic_regression", 
        choices=["xgboost", "logistic_regression", "decision_tree"],
        help="Algoritmo de clasificación a instanciar. Habilita la experimentación dinámica sin modificar el código fuente."
    )
    
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42,
        help="Semilla estática para el split estratificado y la inicialización de pesos. Obligatorio para asegurar la reproducibilidad del experimento."
    )
    
    parser.add_argument(
        "--n_estimators", 
        type=int, 
        default=150,
        help="[XGBoost] Número de árboles secuenciales a construir (boosting rounds)."
    )
    
    parser.add_argument(
        "--max_depth", 
        type=int, 
        default=4,
        help="[XGBoost / Decision Tree] Profundidad máxima del modelo. Controla la complejidad para mitigar el riesgo de sobreajuste (overfitting)."
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.05,
        help="[XGBoost] Tasa de aprendizaje (eta). Escala la contribución de cada árbol para estabilizar la convergencia."
    )
    
    parser.add_argument(
        "--subsample", 
        type=float, 
        default=0.8,
        help="[XGBoost] Fracción del dataset inyectada a cada árbol. Actúa como mecanismo de regularización."
    )
    
    parser.add_argument(
        "--c_reg", 
        type=float, 
        default=1.0,
        help="[Logistic Regression] Inverso de la fuerza de regularización (L2 por defecto). Valores bajos penalizan coeficientes altos."
    )
    
    args = parser.parse_args()
    
    # 1. Lectura y división de los datos desde dataset.py
    X_train, X_test, y_train, y_test, struct_params = load_and_split_data(args.data_path, args.random_state)

    # 2. Desempaquetado del preprocesador y sus parámetros de ingeniería
    preprocessor, feat_params = build_preprocessor()
    
    # 3. Orquestación desde la clase local
    orchestrator = ModelOrchestrator(args, struct_params, feat_params)
    orchestrator.build_and_train(X_train, y_train, preprocessor)
    orchestrator.evaluate(X_test, y_test)
    orchestrator.log_experiment()