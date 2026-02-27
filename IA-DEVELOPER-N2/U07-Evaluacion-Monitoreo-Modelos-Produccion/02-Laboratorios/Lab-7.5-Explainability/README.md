# Laboratorio 7.5: Explicabilidad y Cumplimiento con SHAP y Arize

## 🎯 Contexto
Un modelo de aprobación de préstamos debe cumplir con regulaciones de explicabilidad (ej: GDPR). Se implementará explicaciones con SHAP para cada decisión, monitoreo con Arize para auditorías, y documentación automática de decisiones.

## 🎯 Objetivos

- Explicar con SHAP para cada decisión
- Monitorear con Arize para auditorías
- Generar documentación automática para auditorías

## 📋 Marco Lógico del Laboratorio

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Explicaciones SHAP** | Generar explicaciones individuales | Explicaciones para 100% de decisiones | Código SHAP funcionando |
| **Monitoreo Arize** | Implementar tracking de explicaciones | Datos en Arize UI | Configuración de Arize |
| **Documentación Automática** | Generar reportes de auditoría | Reportes generados automáticamente | Scripts de documentación |
| **Cumplimiento GDPR** | Asegurar explicabilidad obligatoria | Explicaciones disponibles por solicitud | Logs de acceso y explicaciones |

## 🛠️ Tecnologías Utilizadas

- **SHAP**: Explicaciones de modelos de ML
- **Arize**: Plataforma de observabilidad de modelos
- **FastAPI**: API para servir el modelo con explicaciones
- **TensorFlow**: Modelo de aprobación de préstamos
- **Pandas**: Manipulación de datos
- **Docker**: Contenerización

## 📁 Estructura del Laboratorio

```
Laboratorio 7.5 - Explicabilidad y Cumplimiento con SHAP y Arize/
├── README.md                           # Esta guía
├── requirements.txt                     # Dependencias Python
├── shap_explanations.py               # Generación de explicaciones SHAP
├── arize_integration.py               # Integración con Arize
├── audit_documentation.py             # Generación de documentación de auditoría
├── loan_approval_api.py               # API con explicaciones integradas
├── docker-compose.yml                  # Orquestación de servicios
├── data/                              # Datos de préstamos
│   ├── loan_applications.csv         # Solicitudes de préstamo
│   ├── loan_features.csv             # Características de préstamos
│   └── reference_explanations.json  # Explicaciones de referencia
├── models/                            # Modelos entrenados
│   └── loan_approval_model.h5        # Modelo de aprobación
├── configs/                           # Configuraciones
│   ├── arize_config.json             # Configuración de Arize
│   └── shap_config.json              # Configuración de SHAP
└── outputs/                           # Resultados generados
    ├── explanations/                   # Explicaciones generadas
    ├── audit_reports/                  # Reportes de auditoría
    ├── shap_plots/                    # Visualizaciones SHAP
    └── compliance_logs/                # Logs de cumplimiento
```

## 🚀 Implementación Paso a Paso

### Paso 1: Configuración del Entorno

```bash
pip install shap==0.42.0 arize==4.0.0 pandas==2.0.0 numpy==1.24.0 tensorflow==2.15.0 fastapi==0.103.0 uvicorn==0.24.0 matplotlib==3.8.0 seaborn==0.12.0 plotly==5.17.0
```

### Paso 2: Generar Explicaciones con SHAP

El script `shap_explanations.py` implementa la generación de explicaciones:

```python
import shap
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging

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
    
    # Cargar datos de ejemplo
    applications_data = pd.read_csv('data/loan_applications.csv').to_dict('records')
    background_data = pd.read_csv('data/loan_features.csv').values
    
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
```

### Paso 3: Integración con Arize

El script `arize_integration.py` configura el monitoreo con Arize:

```python
from arize.api import Client
from arize.pandas.logger import Client as Logger
import pandas as pd
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArizeMonitor:
    """
    Clase para integración con Arize para monitoreo de modelos
    """
    
    def __init__(self, space_key, api_key):
        self.space_key = space_key
        self.api_key = api_key
        self.client = None
        self.logger = None
        
    def setup_arize(self):
        """Configurar cliente de Arize"""
        try:
            # Configurar cliente de Arize
            self.client = Client(space_key=self.space_key, api_key=self.api_key)
            
            # Configurar logger
            self.logger = Logger(
                client=self.client,
                model_id="loan_approval_model",
                model_version="1.0",
                model_type=Logger.ModelTypes.SCORE_CATEGORICAL
            )
            
            logger.info("Arize configurado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error configurando Arize: {e}")
            return False
    
    def log_predictions(self, predictions_data):
        """Registrar predicciones en Arize"""
        try:
            for data in predictions_data:
                self.logger.log(
                    prediction_id=str(data['application_id']),
                    features=data['features'],
                    prediction_label=data['prediction']['approved'],
                    prediction_score=data['prediction']['confidence'],
                    shap_values=data['shap_values'],
                    feature_names=data['feature_names'],
                    timestamp=data['timestamp']
                )
            
            logger.info(f"Registradas {len(predictions_data)} predicciones en Arize")
            return True
            
        except Exception as e:
            logger.error(f"Error registrando predicciones: {e}")
            return False
    
    def log_explanations(self, explanations_data):
        """Registrar explicaciones en Arize"""
        try:
            for exp in explanations_data:
                self.logger.log(
                    prediction_id=str(exp['application_id']),
                    features=exp['features'],
                    prediction_label=exp['prediction']['approved'],
                    prediction_score=exp['prediction']['confidence'],
                    shap_values=exp['shap_values'],
                    feature_names=exp['feature_names'],
                    explanation_data={
                        'top_features': exp['top_features'],
                        'explanation_type': exp['explanation_type'],
                        'compliance_info': exp['compliance_info']
                    },
                    timestamp=exp['timestamp']
                )
            
            logger.info(f"Registradas {len(explanations_data)} explicaciones en Arize")
            return True
            
        except Exception as e:
            logger.error(f"Error registrando explicaciones: {e}")
            return False
    
    def get_fairness_metrics(self, time_period='last_30_days'):
        """Obtener métricas de equidad de Arize"""
        try:
            # En producción, usar API de Arize para obtener métricas
            # Aquí simulamos algunas métricas
            fairness_metrics = {
                "demographic_parity": 0.85,
                "equal_opportunity": 0.82,
                "equalized_odds": 0.80,
                "disparate_impact": 0.12,
                "overall_accuracy": 0.88,
                "false_positive_rate": 0.15,
                "false_negative_rate": 0.08,
                "time_period": time_period,
                "sample_size": 10000
            }
            
            return fairness_metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas de equidad: {e}")
            return None
    
    def generate_compliance_report(self):
        """Generar reporte de cumplimiento"""
        try:
            fairness_metrics = self.get_fairness_metrics()
            
            if not fairness_metrics:
                return None
            
            # Evaluar cumplimiento
            compliance_score = self._calculate_compliance_score(fairness_metrics)
            
            report = {
                "generated_at": datetime.now().isoformat(),
                "model_id": "loan_approval_model",
                "model_version": "1.0",
                "fairness_metrics": fairness_metrics,
                "compliance_score": compliance_score,
                "compliance_status": "compliant" if compliance_score >= 0.8 else "non_compliant",
                "recommendations": self._generate_compliance_recommendations(fairness_metrics),
                "regulatory_frameworks": ["GDPR", "ECOA", "FCRA"],
                "audit_trail_available": True,
                "explanation_transparency": True
            }
            
            # Guardar reporte
            with open('outputs/audit_reports/compliance_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("Reporte de cumplimiento generado")
            return report
            
        except Exception as e:
            logger.error(f"Error generando reporte de cumplimiento: {e}")
            return None
    
    def _calculate_compliance_score(self, fairness_metrics):
        """Calcular score de cumplimiento"""
        scores = []
        
        # Métricas de equidad
        if fairness_metrics.get('demographic_parity', 0) >= 0.8:
            scores.append(1.0)
        else:
            scores.append(fairness_metrics.get('demographic_parity', 0))
        
        if fairness_metrics.get('equal_opportunity', 0) >= 0.8:
            scores.append(1.0)
        else:
            scores.append(fairness_metrics.get('equal_opportunity', 0))
        
        if fairness_metrics.get('disparate_impact', 1) <= 0.2:
            scores.append(1.0)
        else:
            scores.append(1.0 - fairness_metrics.get('disparate_impact', 0))
        
        # Accuracy general
        if fairness_metrics.get('overall_accuracy', 0) >= 0.85:
            scores.append(1.0)
        else:
            scores.append(fairness_metrics.get('overall_accuracy', 0) / 0.85)
        
        return sum(scores) / len(scores)
    
    def _generate_compliance_recommendations(self, fairness_metrics):
        """Generar recomendaciones de cumplimiento"""
        recommendations = []
        
        if fairness_metrics.get('demographic_parity', 0) < 0.8:
            recommendations.append({
                "priority": "high",
                "issue": "Baja paridad demográfica",
                "recommendation": "Implementar técnicas de reponderación para mejorar equidad"
            })
        
        if fairness_metrics.get('disparate_impact', 0) > 0.2:
            recommendations.append({
                "priority": "high",
                "issue": "Alto impacto dispar",
                "recommendation": "Revisar criterios de aprobación para reducir sesgo"
            })
        
        if fairness_metrics.get('false_negative_rate', 0) > 0.1:
            recommendations.append({
                "priority": "medium",
                "issue": "Alta tasa de falsos negativos",
                "recommendation": "Ajustar umbral de decisión para reducir rechazos incorrectos"
            })
        
        return recommendations

def main():
    """Función principal para integración con Arize"""
    logger.info("=" * 80)
    logger.info("INTEGRANDO CON ARIZE PARA MONITOREO DE PRÉSTAMOS")
    logger.info("=" * 80)
    
    # Configurar credenciales (en producción, usar variables de entorno)
    space_key = "YOUR_SPACE_KEY"
    api_key = "YOUR_API_KEY"
    
    # Crear monitor
    monitor = ArizeMonitor(space_key, api_key)
    
    if not monitor.setup_arize():
        return
    
    # Cargar datos de ejemplo
    explanations_data = []
    try:
        with open('outputs/explanations/loan_explanations.json', 'r') as f:
            explanations_data = json.load(f)
    except FileNotFoundError:
        logger.warning("No se encontraron explicaciones. Ejecuta shap_explanations.py primero.")
        return
    
    # Registrar explicaciones en Arize
    monitor.log_explanations(explanations_data)
    
    # Generar reporte de cumplimiento
    compliance_report = monitor.generate_compliance_report()
    
    if compliance_report:
        logger.info(f"Score de cumplimiento: {compliance_report['compliance_score']:.2f}")
        logger.info(f"Estado: {compliance_report['compliance_status']}")
    
    logger.info("=" * 80)
    logger.info("INTEGRACIÓN CON ARIZE COMPLETADA")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
```

## 📊 Resultados Esperados

### Sistema de Explicabilidad Funcional
- **Explicaciones SHAP** generadas para cada decisión
- **Visualizaciones** de las explicaciones
- **Monitoreo Arize** con métricas de equidad
- **Reportes de auditoría** generados automáticamente

### Cumplimiento Regulatorio
- **GDPR**: Explicaciones disponibles por solicitud
- **Transparencia**: Decisiones explicadas en lenguaje comprensible
- **Auditoría**: Trail completo de decisiones
- **Equidad**: Monitoreo de sesgos algorítmicos

### Métricas de Equidad
- **Paridad Demográfica**: Igualdad de resultados entre grupos
- **Oportunidad Igual**: Igualdad de tasas de verdaderos positivos
- **Impacto Dispar**: Comparación de tasas de aprobación
- **Odds Igualizados**: Igualdad de tasas de falsos positivos/negativos

## 📌 Entregables del Laboratorio

| Entregable | Descripción | Formato |
|-------------|-------------|-----------|
| Explicaciones con SHAP | Código para generar explicaciones individuales | shap_explanations.py |
| Integración con Arize | Monitoreo de explicaciones en producción | arize_integration.py |
| Documentación de Auditoría | Generación automática de reportes regulatorios | audit_documentation.py |
| API con Explicaciones | Endpoint con explicaciones integradas | loan_approval_api.py |
| Visualizaciones SHAP | Gráficos de explicaciones | outputs/shap_plots/ |
| Reportes de Cumplimiento | Documentación para auditorías | outputs/audit_reports/ |

## 🎯 Criterios de Evaluación

- **Funcionalidad** (40%): Sistema completo funcionando correctamente
- **Explicabilidad** (25%): Explicaciones claras y comprensibles
- **Cumplimiento** (20%): Requisitos regulatorios cumplidos
- **Monitoreo** (15%): Sistema de auditoría funcionando

## 🚀 Extensión y Mejoras

### Mejoras Sugeridas
1. **Explicaciones Adicionales**: LIME, Counterfactual Explanations
2. **Dashboard de Equidad**: Visualización interactiva de métricas
3. **Alertas de Sesgo**: Notificaciones automáticas de desviaciones
4. **Reportes Automatizados**: Generación programada de informes regulatorios

### Aplicaciones en Producción
1. **Explicaciones en Tiempo Real**: Generación instantánea de explicaciones
2. **Auditoría Continua**: Monitoreo 24/7 de cumplimiento
3. **Integración CRM**: Explicaciones en sistemas de gestión de clientes
4. **Portal de Transparencia**: Auto-servicio para clientes

---

**Duración Estimada**: 10-12 horas  
**Nivel de Dificultad**: Avanzada  
**Prerrequisitos**: Conocimientos de explicabilidad, regulaciones, sistemas de monitoreo
