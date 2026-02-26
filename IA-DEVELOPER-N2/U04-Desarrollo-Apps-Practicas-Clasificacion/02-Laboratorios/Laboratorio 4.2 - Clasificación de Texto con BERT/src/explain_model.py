"""
Explicabilidad del Modelo BERT con SHAP
Laboratorio 4.2 - Clasificación de Texto con BERT y Adaptadores
"""

import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
from bert_adapters import BERTWithAdapters
import pandas as pd
from typing import List, Dict, Tuple
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTExplainer:
    """
    Clase para explicar predicciones del modelo BERT usando SHAP
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Configurar SHAP explainer
        self.explainer = None
        self.background_data = None
        
        logger.info("BERTExplainer inicializado")
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Función de predicción para SHAP
        """
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenizar
                inputs = self.tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predicción
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs, dim=-1)
                probabilities.append(probs.cpu().numpy())
        
        return np.concatenate(probabilities, axis=0)
    
    def setup_explainer(self, background_texts: List[str], sample_size: int = 100):
        """
        Configura el explainer SHAP con datos de fondo
        """
        logger.info(f"Configurando explainer con {len(background_texts)} textos de fondo...")
        
        # Muestrear datos de fondo
        if len(background_texts) > sample_size:
            background_texts = np.random.choice(background_texts, sample_size, replace=False)
        
        self.background_data = background_texts
        
        # Crear explainer
        self.explainer = shap.Explainer(
            self.predict_proba,
            self.background_data,
            algorithm="permutation"
        )
        
        logger.info("Explainer SHAP configurado")
    
    def explain_text(self, text: str, class_names: List[str] = None) -> Dict:
        """
        Explica una predicción individual
        """
        if self.explainer is None:
            raise ValueError("El explainer no está configurado. Llama a setup_explainer() primero.")
        
        # Predicción original
        pred_proba = self.predict_proba([text])[0]
        pred_class = np.argmax(pred_proba)
        
        # Explicación SHAP
        shap_values = self.explainer([text])
        
        # Procesar valores SHAP
        tokenized = self.tokenizer(text, return_tensors='pt')
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0])
        
        # Obtener valores SHAP para la clase predicha
        values = shap_values[0, :, pred_class].values
        
        # Crear explicación detallada
        explanation = {
            'text': text,
            'tokens': tokens,
            'shap_values': values,
            'prediction': {
                'class': pred_class,
                'class_name': class_names[pred_class] if class_names else str(pred_class),
                'probabilities': pred_proba.tolist()
            },
            'token_importance': list(zip(tokens, values))
        }
        
        return explanation
    
    def explain_batch(self, texts: List[str], class_names: List[str] = None) -> List[Dict]:
        """
        Explica múltiples textos
        """
        explanations = []
        
        for text in texts:
            try:
                explanation = self.explain_text(text, class_names)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Error explicando texto: {e}")
                explanations.append(None)
        
        return explanations
    
    def plot_explanation(self, explanation: Dict, max_tokens: int = 20, save_path: str = None):
        """
        Visualiza la explicación de un texto
        """
        tokens = explanation['tokens'][:max_tokens]
        values = explanation['shap_values'][:max_tokens]
        
        # Crear DataFrame para visualización
        df = pd.DataFrame({
            'Token': tokens,
            'SHAP Value': values,
            'Impact': ['Positive' if v > 0 else 'Negative' for v in values]
        })
        
        # Gráfico de barras
        plt.figure(figsize=(12, 6))
        
        # Colores basados en el impacto
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = plt.bar(range(len(tokens)), values, color=colors, alpha=0.7)
        
        # Añadir etiquetas
        plt.xlabel('Tokens')
        plt.ylabel('SHAP Value')
        plt.title(f"Explicación de Predicción\nClase: {explanation['prediction']['class_name']} "
                 f"(Prob: {explanation['prediction']['probabilities'][explanation['prediction']['class']]:.3f})")
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        
        # Añadir línea en y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()
        
        return df
    
    def plot_token_importance(self, explanation: Dict, top_k: int = 10, save_path: str = None):
        """
        Visualiza los tokens más importantes
        """
        tokens = explanation['tokens']
        values = explanation['shap_values']
        
        # Crear DataFrame y ordenar por valor absoluto
        df = pd.DataFrame({
            'Token': tokens,
            'SHAP Value': values,
            'Absolute Value': np.abs(values)
        })
        
        # Filtrar tokens especiales y ordenar
        df = df[~df['Token'].isin(['[CLS]', '[SEP]', '[PAD]'])]
        df = df.sort_values('Absolute Value', ascending=False).head(top_k)
        
        # Gráfico horizontal
        plt.figure(figsize=(10, 6))
        
        colors = ['green' if v > 0 else 'red' for v in df['SHAP Value']]
        
        bars = plt.barh(range(len(df)), df['SHAP Value'], color=colors, alpha=0.7)
        
        plt.yticks(range(len(df)), df['Token'])
        plt.xlabel('SHAP Value')
        plt.title(f'Top {top_k} Tokens más Importantes')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Añadir valores en las barras
        for i, (bar, value) in enumerate(zip(bars, df['SHAP Value'])):
            plt.text(value + (0.01 if value > 0 else -0.01), i, f'{value:.3f}', 
                    va='center', ha='left' if value > 0 else 'right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()
        
        return df
    
    def generate_explanation_report(self, text: str, class_names: List[str] = None, 
                                 save_dir: str = 'explanations') -> Dict:
        """
        Genera un reporte completo de explicación
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Explicación
        explanation = self.explain_text(text, class_names)
        
        # Visualizaciones
        self.plot_explanation(explanation, save_path=os.path.join(save_dir, 'explanation_bars.png'))
        self.plot_token_importance(explanation, save_path=os.path.join(save_dir, 'token_importance.png'))
        
        # Análisis detallado
        tokens = explanation['tokens']
        values = explanation['shap_values']
        
        # Estadísticas
        positive_tokens = [(t, v) for t, v in zip(tokens, values) if v > 0]
        negative_tokens = [(t, v) for t, v in zip(tokens, values) if v < 0]
        
        # Ordenar por magnitud
        positive_tokens.sort(key=lambda x: x[1], reverse=True)
        negative_tokens.sort(key=lambda x: x[1])
        
        report = {
            'text': text,
            'prediction': explanation['prediction'],
            'top_positive_tokens': positive_tokens[:5],
            'top_negative_tokens': negative_tokens[:5],
            'summary': {
                'total_tokens': len(tokens),
                'positive_impact_tokens': len(positive_tokens),
                'negative_impact_tokens': len(negative_tokens),
                'max_positive_impact': max([v for _, v in positive_tokens]) if positive_tokens else 0,
                'max_negative_impact': min([v for _, v in negative_tokens]) if negative_tokens else 0
            },
            'visualizations': {
                'explanation_bars': os.path.join(save_dir, 'explanation_bars.png'),
                'token_importance': os.path.join(save_dir, 'token_importance.png')
            }
        }
        
        # Guardar reporte
        import json
        with open(os.path.join(save_dir, 'explanation_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Reporte de explicación guardado en: {save_dir}")
        
        return report


def main():
    """
    Ejemplo de uso del explicador
    """
    # Cargar modelo y tokenizer (ejemplo)
    try:
        model = BERTWithAdapters.load_model('best_bert_adapters.pth')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Crear explicador
        explainer = BERTExplainer(model, tokenizer)
        
        # Textos de ejemplo
        background_texts = [
            "This product is amazing and works perfectly",
            "Terrible experience, would not recommend",
            "Average quality, nothing special",
            "Excellent customer service and fast delivery",
            "Poor quality, broke after one use"
        ]
        
        test_text = "This product is absolutely amazing! Best purchase I've made this year."
        
        # Configurar explainer
        explainer.setup_explainer(background_texts)
        
        # Generar explicación
        report = explainer.generate_explanation_report(
            test_text, 
            class_names=['Negative', 'Positive'],
            save_dir='explanations/sample'
        )
        
        print("✅ Explicación generada exitosamente")
        print(f"Predicción: {report['prediction']}")
        print(f"Tokens positivos top: {report['top_positive_tokens'][:3]}")
        
    except FileNotFoundError:
        print("Modelo no encontrado. Este es un ejemplo de uso.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
