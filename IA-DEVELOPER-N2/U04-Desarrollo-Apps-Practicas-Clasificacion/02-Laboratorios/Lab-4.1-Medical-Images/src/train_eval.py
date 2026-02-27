"""
Entrenamiento y Evaluación del Modelo Híbrido
Laboratorio 4.1 - Clasificación de Imágenes Médicas con EfficientNetV3 y ViT
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from data_loader import MedicalImageDataLoader
from hybrid_model import HybridMedicalClassifier
import json

class ModelTrainer:
    """
    Clase para entrenar y evaluar el modelo híbrido
    """
    
    def __init__(self, model_save_dir='models'):
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.history = None
        self.classifier = None
    
    def train_model(self, data_dir, epochs=50, batch_size=32, validation_split=0.2):
        """
        Entrena el modelo híbrido
        """
        print("🚀 Iniciando entrenamiento del modelo híbrido...")
        
        # Inicializar componentes
        loader = MedicalImageDataLoader(batch_size=batch_size)
        self.classifier = HybridMedicalClassifier()
        
        # Crear generadores de datos
        train_gen, val_gen = loader.create_data_generators(data_dir, validation_split)
        
        # Construir y compilar modelo
        model = self.classifier.build_model()
        model = self.classifier.compile_model()
        
        # Obtener pesos de clases si hay desbalance
        class_weights = loader.get_class_weights(data_dir)
        print(f"Pesos de clases: {class_weights}")
        
        # Configurar callbacks
        callbacks = self.classifier.get_callbacks(
            model_path=os.path.join(self.model_save_dir, 'best_model.h5'),
            patience=10
        )
        
        # Entrenar modelo
        print(f"📊 Entrenando por {epochs} épocas...")
        self.history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Guardar historial
        self.save_training_history()
        
        return model, train_gen, val_gen
    
    def evaluate_model(self, model, val_gen, save_results=True):
        """
        Evalúa el modelo y genera métricas detalladas
        """
        print("📈 Evaluando modelo...")
        
        # Predicciones
        y_pred_proba = model.predict(val_gen)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = val_gen.classes
        
        # Métricas básicas
        loss, accuracy, precision, recall, auc_score = model.evaluate(val_gen, verbose=0)
        
        print(f"\n🎯 Resultados de Evaluación:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc_score:.4f}")
        
        # Reporte de clasificación
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))
        
        # Guardar resultados
        if save_results:
            results = {
                'loss': float(loss),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'auc': float(auc_score),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            with open(os.path.join(self.model_save_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
        
        # Visualizaciones
        self.plot_training_history()
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_pred_proba)
        
        return results
    
    def plot_training_history(self):
        """
        Visualiza el historial de entrenamiento
        """
        if self.history is None:
            print("No hay historial de entrenamiento disponible.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Accuracy por Época')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Loss por Época')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Precision por Época')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Recall por Época')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Visualiza la matriz de confusión
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'])
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.savefig(os.path.join(self.model_save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """
        Visualiza la curva ROC
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.model_save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_history(self):
        """
        Guarda el historial de entrenamiento
        """
        if self.history is not None:
            history_dict = {key: [float(x) for x in values] 
                          for key, values in self.history.history.items()}
            
            with open(os.path.join(self.model_save_dir, 'training_history.json'), 'w') as f:
                json.dump(history_dict, f, indent=2)
    
    def load_model(self, model_path):
        """
        Carga un modelo entrenado
        """
        self.classifier = HybridMedicalClassifier()
        self.classifier.model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo cargado desde {model_path}")
        return self.classifier.model


def main():
    """
    Función principal para ejecutar el entrenamiento y evaluación
    """
    # Configuración
    DATA_DIR = 'data/train'
    MODEL_SAVE_DIR = 'models'
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Crear directorios
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Inicializar trainer
    trainer = ModelTrainer(MODEL_SAVE_DIR)
    
    # Entrenar modelo
    try:
        model, train_gen, val_gen = trainer.train_model(
            data_dir=DATA_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        # Evaluar modelo
        results = trainer.evaluate_model(model, val_gen)
        
        print("\n🎉 Entrenamiento y evaluación completados!")
        print(f"📁 Resultados guardados en: {MODEL_SAVE_DIR}")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise


if __name__ == "__main__":
    main()
