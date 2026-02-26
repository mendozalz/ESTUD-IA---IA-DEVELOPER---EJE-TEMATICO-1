from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self) -> bool:
        """Cargar el modelo de análisis de sentimiento"""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Cargar tokenizer y modelo
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Crear pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Verificar si el modelo está cargado"""
        return self.pipeline is not None
    
    def analyze(self, text: str) -> Dict:
        """Analizar sentimiento del texto"""
        if not self.is_loaded():
            if not self.load_model():
                raise RuntimeError("Failed to load sentiment model")
        
        try:
            # Realizar análisis
            results = self.pipeline(text)
            
            # Procesar resultados
            scores = {result['label'].lower(): result['score'] for result in results[0]}
            
            # Determinar sentimiento principal
            sentiment = max(scores, key=scores.get)
            confidence = scores[sentiment]
            
            # Mapear etiquetas
            sentiment_mapping = {
                'positive': 'positivo',
                'negative': 'negativo', 
                'neutral': 'neutral'
            }
            
            mapped_sentiment = sentiment_mapping.get(sentiment, sentiment)
            
            # Calcular métricas adicionales
            polarity = self._calculate_polarity(scores)
            subjectivity = self._calculate_subjectivity(scores)
            
            return {
                'sentiment': mapped_sentiment,
                'confidence': confidence,
                'scores': scores,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'model_name': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    def _calculate_polarity(self, scores: Dict) -> float:
        """Calcular polaridad (-1 a 1)"""
        positive = scores.get('positive', 0)
        negative = scores.get('negative', 0)
        
        if positive + negative == 0:
            return 0.0
        
        return (positive - negative) / (positive + negative)
    
    def _calculate_subjectivity(self, scores: Dict) -> float:
        """Calcular subjetividad (0 a 1)"""
        neutral = scores.get('neutral', 0)
        return 1.0 - neutral
    
    def get_model_info(self) -> Dict:
        """Obtener información del modelo"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_loaded': self.is_loaded(),
            'labels': ['positive', 'negative', 'neutral'] if self.is_loaded() else []
        }
