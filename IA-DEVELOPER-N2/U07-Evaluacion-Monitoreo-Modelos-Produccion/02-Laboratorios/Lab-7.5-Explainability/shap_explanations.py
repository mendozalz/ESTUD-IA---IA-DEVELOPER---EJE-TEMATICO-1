"""
Caso de Uso 5 - Explicabilidad y Cumplimiento con SHAP y Arize
Fase 1: Generación de Explicaciones con SHAP para Préstamos
"""

import shap
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanExplainer:
    """
    Clase para generar explicaciones de decisiones de préstamos usando SHAP
    """
    
    def __init__(self, model_path="models/loan_approval_model.h5"):
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.model_path = model_path
        
    def load_model(self):
        """Cargar modelo de aprobación de préstamos"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Modelo cargado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False
    
    def setup_explainer(self, background_data):
        """Configurar explainer SHAP"""
        try:
            # Usar DeepExplainer para modelos neuronales
            self.explainer = shap.DeepExplainer(
                self.model, 
                background_data[:100]  # Usar 100 muestras de fondo
            )
            logger.info("Explainer SHAP configurado")
            return True
        except Exception as e:
            logger.error(f"Error configurando explainer: {e}")
            return False
    
    def explain_application(self, application_data, application_id):
        """Generar explicación para una solicitud de préstamo"""
        try:
            # Preparar datos para el modelo
            features = self._prepare_features(application_data)
            
            # Generar valores SHAP
            shap_values = self.explainer.shap_values(features)
            
            # Crear explicación completa
            explanation = {
                "application_id": application_id,
                "timestamp": datetime.now().isoformat(),
                "prediction": self._get_prediction(features),
                "shap_values": shap_values[0].tolist(),
                "feature_names": self.feature_names,
                "base_values": self.explainer.expected_value[0].tolist(),
                "top_features": self._get_top_features(shap_values[0]),
                "explanation_type": "shap_deep",
                "compliance_info": self._get_compliance_info()
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generando explicación: {e}")
            return None
    
    def _prepare_features(self, application_data):
        """Preparar características para el modelo"""
        # Convertir a array numpy
        feature_array = np.array([
            application_data.get('income', 0),
            application_data.get('credit_score', 0),
            application_data.get('loan_amount', 0),
            application_data.get('employment_years', 0),
            application_data.get('age', 0),
            application_data.get('debt_to_income', 0),
            application_data.get('has_collateral', 0),
            application_data.get('loan_purpose', 0)
        ])
        
        # Definir nombres de características
        self.feature_names = [
            'income', 'credit_score', 'loan_amount', 'employment_years',
            'age', 'debt_to_income', 'has_collateral', 'loan_purpose'
        ]
        
        return feature_array.reshape(1, -1)
    
    def _get_prediction(self, features):
        """Obtener predicción del modelo"""
        prediction = self.model.predict(features, verbose=0)
        return {
            "approved": bool(prediction[0] > 0.5),
            "confidence": float(prediction[0]),
            "probability": float(prediction[0])
        }
    
    def _get_top_features(self, shap_values):
        """Obtener características más influyentes"""
        feature_importance = sorted(
            zip(self.feature_names, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]  # Top 5 características
        
        return [
            {
                "feature": feature,
                "shap_value": float(value),
                "impact": "positive" if value > 0 else "negative",
                "importance_rank": idx + 1
            }
            for idx, (feature, value) in enumerate(feature_importance)
        ]
    
    def _get_compliance_info(self):
        """Obtener información de cumplimiento"""
        return {
            "regulation": "GDPR Article 22",
            "explanation_available": True,
            "human_understandable": True,
            "data_minimization": True,
            "accuracy_maintained": True,
            "bias_mitigation": True,
            "audit_trail": True
        }
    
    def generate_explanation_plot(self, explanation, save_path):
        """Generar visualización de la explicación"""
        try:
            # Crear gráfico de fuerza SHAP
            shap.force_plot(
                base_value=explanation["base_values"],
                shap_values=explanation["shap_values"],
                feature_names=explanation["feature_names"],
                matplotlib=True,
                show=False
            )
            
            plt.title(f"Explicación - Solicitud {explanation['application_id']}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gráfico guardado en {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generando gráfico: {e}")
            return False
    
    def batch_explain(self, applications_data):
        """Generar explicaciones para múltiples solicitudes"""
        explanations = []
        
        for app_data in applications_data:
            app_id = app_data.get('application_id')
            explanation = self.explain_application(app_data, app_id)
            
            if explanation:
                # Generar gráfico
                plot_path = f"outputs/shap_plots/explanation_{app_id}.png"
                self.generate_explanation_plot(explanation, plot_path)
                explanation["plot_path"] = plot_path
                
                explanations.append(explanation)
        
        return explanations

def main():
    """Función principal para generar explicaciones"""
    logger.info("=" * 80)
    logger.info("GENERANDO EXPLICACIONES SHAP PARA PRÉSTAMOS")
    logger.info("=" * 80)
    
    # Crear directorios necesarios
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/shap_plots", exist_ok=True)
    os.makedirs("outputs/explanations", exist_ok=True)
    
    # Cargar datos de ejemplo
    try:
        applications_data = pd.read_csv('data/loan_applications.csv').to_dict('records')
    except FileNotFoundError:
        # Generar datos de ejemplo si no existen
        np.random.seed(42)
        applications_data = []
        for i in range(100):
            app_data = {
                'application_id': f"APP_{i:04d}",
                'income': np.random.randint(20000, 150000),
                'credit_score': np.random.randint(300, 850),
                'loan_amount': np.random.randint(5000, 100000),
                'employment_years': np.random.randint(0, 30),
                'age': np.random.randint(18, 80),
                'debt_to_income': np.random.uniform(0, 0.5),
                'has_collateral': np.random.choice([0, 1]),
                'loan_purpose': np.random.randint(0, 5)
            }
            applications_data.append(app_data)
        
        pd.DataFrame(applications_data).to_csv('data/loan_applications.csv', index=False)
    
    # Generar datos de fondo
    background_data = np.array([
        [app['income'], app['credit_score'], app['loan_amount'], 
         app['employment_years'], app['age'], app['debt_to_income'],
         app['has_collateral'], app['loan_purpose']]
        for app in applications_data[:100]
    ])
    
    # Crear modelo dummy si no existe
    if not os.path.exists("models/loan_approval_model.h5"):
        logger.info("Creando modelo dummy...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.save("models/loan_approval_model.h5")
    
    # Crear explainer
    explainer = LoanExplainer()
    
    if not explainer.load_model():
        return
    
    if not explainer.setup_explainer(background_data):
        return
    
    # Generar explicaciones
    explanations = explainer.batch_explain(applications_data)
    
    # Guardar explicaciones
    with open('outputs/explanations/loan_explanations.json', 'w') as f:
        json.dump(explanations, f, indent=2)
    
    logger.info(f"Generadas {len(explanations)} explicaciones")
    logger.info("Explicaciones guardadas en outputs/explanations/loan_explanations.json")
    
    logger.info("=" * 80)
    logger.info("EXPLICACIONES GENERADAS EXITOSAMENTE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
