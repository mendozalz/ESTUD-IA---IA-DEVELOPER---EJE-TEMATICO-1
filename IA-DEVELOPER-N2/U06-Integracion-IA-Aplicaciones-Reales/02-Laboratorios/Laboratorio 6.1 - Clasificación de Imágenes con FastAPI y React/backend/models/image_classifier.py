import tensorflow as tf
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ImageClassifier:
    def __init__(self, model_path: str = "models/image_classifier.h5"):
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.input_shape = (224, 224, 3)
        self.is_model_loaded = False
        
    async def load_model(self) -> bool:
        """Cargar el modelo de clasificación de imágenes"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Para demo, creamos un modelo simple
            # En producción, cargaríamos un modelo pre-entrenado
            self.model = self._create_demo_model()
            
            # Cargar nombres de clases
            self.class_names = [
                'cat', 'dog', 'bird', 'car', 'airplane', 'ship', 'truck',
                'deer', 'horse', 'frog', 'flower', 'house', 'tree', 'person',
                'bicycle', 'motorcycle', 'bus', 'train', 'boat', 'bottle'
            ]
            
            self.is_model_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _create_demo_model(self):
        """Crear un modelo demo para clasificación"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names) if hasattr(self, 'class_names') else 20, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def predict(self, image: np.ndarray, confidence_threshold: float = 0.5, top_k: int = 5) -> Dict[str, Any]:
        """Realizar predicción sobre una imagen"""
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preparar imagen para predicción
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Realizar predicción
            predictions = self.model.predict(image, verbose=0)
            
            # Procesar resultados
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Obtener top-k predicciones
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            top_predictions = []
            
            for idx in top_indices:
                if idx < len(self.class_names):
                    top_predictions.append({
                        "class": self.class_names[idx],
                        "confidence": float(predictions[0][idx])
                    })
            
            # Filtrar por umbral de confianza
            filtered_predictions = [
                pred for pred in top_predictions 
                if pred["confidence"] >= confidence_threshold
            ]
            
            result = {
                "predicted_class": self.class_names[predicted_class_idx] if predicted_class_idx < len(self.class_names) else "unknown",
                "confidence": confidence,
                "top_predictions": filtered_predictions,
                "all_predictions": top_predictions,
                "model_version": "1.0.0"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if not self.is_model_loaded:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": "Image Classifier",
            "model_version": "1.0.0",
            "input_shape": self.input_shape,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "model_path": self.model_path,
            "framework": "TensorFlow",
            "framework_version": tf.__version__
        }
    
    async def get_classes(self) -> List[str]:
        """Obtener lista de clases"""
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded")
        
        return self.class_names
    
    async def evaluate_model(self, test_data_path: str) -> Dict[str, Any]:
        """Evaluar el modelo con datos de prueba"""
        try:
            # Simulación de evaluación
            # En producción, cargaríamos datos reales
            
            metrics = {
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85,
                "confusion_matrix": self._generate_confusion_matrix(),
                "classification_report": self._generate_classification_report(),
                "test_samples": 1000,
                "evaluation_time": datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            raise
    
    def _generate_confusion_matrix(self) -> List[List[int]]:
        """Generar matriz de confusión simulada"""
        n_classes = len(self.class_names)
        matrix = []
        
        for i in range(n_classes):
            row = []
            for j in range(n_classes):
                if i == j:
                    row.append(np.random.randint(80, 100))  # Verdaderos positivos
                else:
                    row.append(np.random.randint(0, 20))   # Falsos positivos/negativos
            matrix.append(row)
        
        return matrix
    
    def _generate_classification_report(self) -> Dict[str, Any]:
        """Generar reporte de clasificación simulado"""
        report = {}
        
        for class_name in self.class_names[:10]:  # Limitar a 10 clases para demo
            report[class_name] = {
                "precision": round(np.random.uniform(0.7, 0.95), 3),
                "recall": round(np.random.uniform(0.7, 0.95), 3),
                "f1-score": round(np.random.uniform(0.7, 0.95), 3),
                "support": np.random.randint(50, 150)
            }
        
        # Agregar promedios
        report["accuracy"] = round(np.random.uniform(0.8, 0.9), 3)
        report["macro avg"] = {
            "precision": round(np.random.uniform(0.8, 0.9), 3),
            "recall": round(np.random.uniform(0.8, 0.9), 3),
            "f1-score": round(np.random.uniform(0.8, 0.9), 3),
            "support": 1000
        }
        
        return report
    
    def is_loaded(self) -> bool:
        """Verificar si el modelo está cargado"""
        return self.is_model_loaded
