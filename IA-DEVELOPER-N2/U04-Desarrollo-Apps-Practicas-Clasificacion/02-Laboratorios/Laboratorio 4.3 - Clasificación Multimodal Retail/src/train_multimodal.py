"""
Entrenamiento del Modelo Multimodal
Laboratorio 4.3 - Clasificación Multimodal Retail
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import logging
from datetime import datetime

from data_preprocessing import MultimodalDataPreprocessor
from multimodal_model import MultimodalRetailClassifier, MultimodalTrainer

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalTrainingPipeline:
    """
    Pipeline completo para entrenamiento multimodal
    """
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path) if config_path else self.default_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Directorios
        self.data_dir = self.config['data_dir']
        self.model_dir = self.config['model_dir']
        self.results_dir = self.config['results_dir']
        
        # Crear directorios
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Componentes
        self.preprocessor = None
        self.model = None
        self.trainer = None
        self.class_names = None
        
        logger.info(f"Pipeline inicializado en device: {self.device}")
    
    def default_config(self):
        """
        Configuración por defecto
        """
        return {
            'data_dir': 'data',
            'model_dir': 'models',
            'results_dir': 'results',
            'csv_path': 'data/products.csv',
            'image_dir': 'data/images',
            'batch_size': 16,
            'num_epochs': 20,
            'learning_rate': 1e-4,
            'num_classes': 10,
            'image_model': 'efficientnet_b3',
            'text_model': 'distilbert-base-uncased',
            'fusion_dim': 512,
            'dropout_rate': 0.3,
            'test_size': 0.2,
            'val_size': 0.1
        }
    
    def load_config(self, config_path: str):
        """
        Carga configuración desde archivo JSON
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def prepare_data(self):
        """
        Prepara los datos para entrenamiento
        """
        logger.info("🔄 Preparando datos...")
        
        # Inicializar preprocesador
        self.preprocessor = MultimodalDataPreprocessor(
            image_size=(300, 300),
            max_text_length=128
        )
        
        # Cargar datos
        df = self.preprocessor.load_data_from_csv(
            self.config['csv_path'],
            self.config['image_dir']
        )
        
        # Analizar dataset
        stats = self.preprocessor.analyze_dataset(df)
        
        # Crear datasets
        train_dataset, val_dataset, test_dataset, self.class_names = \
            self.preprocessor.create_datasets(df)
        
        # Crear dataloaders
        train_loader, val_loader, test_loader = self.preprocessor.create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=self.config['batch_size']
        )
        
        # Guardar preprocesador
        self.preprocessor.save_preprocessor(
            os.path.join(self.model_dir, 'preprocessor_state.json')
        )
        
        # Guardar nombres de clases
        with open(os.path.join(self.model_dir, 'class_names.json'), 'w') as f:
            json.dump(self.class_names, f, indent=2)
        
        logger.info(f"✅ Datos preparados. Clases: {self.class_names}")
        
        return train_loader, val_loader, test_loader
    
    def build_model(self):
        """
        Construye el modelo multimodal
        """
        logger.info("🏗️ Construyendo modelo multimodal...")
        
        self.model = MultimodalRetailClassifier(
            num_classes=self.config['num_classes'],
            image_model=self.config['image_model'],
            text_model=self.config['text_model'],
            fusion_dim=self.config['fusion_dim'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Congelar encoders inicialmente
        self.model.freeze_encoders()
        
        # Mostrar parámetros
        self.model.get_trainable_parameters()
        
        logger.info("✅ Modelo construido")
    
    def compute_class_weights(self, train_loader):
        """
        Calcula pesos para clases desbalanceadas
        """
        logger.info("📊 Calculando pesos de clases...")
        
        # Extraer todas las etiquetas del entrenamiento
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['label'].numpy())
        
        # Calcular pesos
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        logger.info(f"Pesos de clases: {class_weights}")
        
        return class_weights
    
    def train_model(self, train_loader, val_loader):
        """
        Entrena el modelo
        """
        logger.info("🚀 Iniciando entrenamiento...")
        
        # Calcular pesos de clases
        class_weights = self.compute_class_weights(train_loader)
        
        # Inicializar trainer
        self.trainer = MultimodalTrainer(self.model, self.device)
        
        # Entrenar
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['num_epochs'],
            lr=self.config['learning_rate'],
            class_weights=class_weights
        )
        
        # Guardar historial
        with open(os.path.join(self.results_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Visualizar entrenamiento
        self.plot_training_history(history)
        
        logger.info("✅ Entrenamiento completado")
        
        return history
    
    def evaluate_model(self, test_loader):
        """
        Evalúa el modelo en el conjunto de prueba
        """
        logger.info("📈 Evaluando modelo...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']
                
                outputs = self.model(images, input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Métricas
        report = classification_report(
            all_labels, all_predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Matriz de confusión
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Guardar resultados
        results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': report['accuracy'],
            'macro_avg': report['macro avg'],
            'weighted_avg': report['weighted avg']
        }
        
        with open(os.path.join(self.results_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Visualizar resultados
        self.plot_confusion_matrix(cm)
        self.plot_classification_report(report)
        
        logger.info(f"✅ Evaluación completada. Accuracy: {results['accuracy']:.4f}")
        
        return results
    
    def plot_training_history(self, history):
        """
        Visualiza el historial de entrenamiento
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        train_accs = [h['train_accuracy'] for h in history]
        val_accs = [h['val_accuracy'] for h in history]
        
        # Loss
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', marker='o')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', marker='o')
        axes[0, 0].set_title('Loss por Época')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, train_accs, label='Train Accuracy', marker='o')
        axes[0, 1].plot(epochs, val_accs, label='Val Accuracy', marker='o')
        axes[0, 1].set_title('Accuracy por Época')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Diferencia de accuracy
        acc_diff = np.array(val_accs) - np.array(train_accs)
        axes[1, 0].plot(epochs, acc_diff, marker='o', color='green')
        axes[1, 0].set_title('Val Accuracy - Train Accuracy')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Diferencia')
        axes[1, 0].grid(True)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Learning curve
        axes[1, 1].plot(epochs, train_losses, label='Train Loss', marker='o')
        axes[1, 1].plot(epochs, val_losses, label='Val Loss', marker='o')
        axes[1, 1].set_title('Learning Curve')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """
        Visualiza la matriz de confusión
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_report(self, report):
        """
        Visualiza el reporte de clasificación
        """
        # Extraer métricas por clase
        classes = []
        precision = []
        recall = []
        f1_score = []
        
        for class_name in self.class_names:
            if class_name in report:
                classes.append(class_name)
                precision.append(report[class_name]['precision'])
                recall.append(report[class_name]['recall'])
                f1_score.append(report[class_name]['f1-score'])
        
        # Crear gráfico
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Clases')
        ax.set_ylabel('Score')
        ax.set_title('Métricas por Clase')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'classification_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self):
        """
        Ejecuta el pipeline completo
        """
        logger.info("🎯 Iniciando pipeline completo de entrenamiento multimodal")
        
        try:
            # Preparar datos
            train_loader, val_loader, test_loader = self.prepare_data()
            
            # Construir modelo
            self.build_model()
            
            # Entrenar modelo
            history = self.train_model(train_loader, val_loader)
            
            # Evaluar modelo
            results = self.evaluate_model(test_loader)
            
            logger.info("🎉 Pipeline completado exitosamente!")
            logger.info(f"📁 Resultados guardados en: {self.results_dir}")
            
            return history, results
            
        except Exception as e:
            logger.error(f"❌ Error en el pipeline: {str(e)}")
            raise


def main():
    """
    Función principal
    """
    # Crear pipeline
    pipeline = MultimodalTrainingPipeline()
    
    # Ejecutar pipeline completo
    try:
        history, results = pipeline.run_complete_pipeline()
        
        print("\n🎊 Entrenamiento completado!")
        print(f"📊 Accuracy final: {results['accuracy']:.4f}")
        print(f"📁 Resultados guardados en: {pipeline.results_dir}")
        
    except FileNotFoundError:
        print("❌ Archivos de datos no encontrados. Asegúrate de tener:")
        print("   - data/products.csv")
        print("   - data/images/ (con las imágenes)")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
