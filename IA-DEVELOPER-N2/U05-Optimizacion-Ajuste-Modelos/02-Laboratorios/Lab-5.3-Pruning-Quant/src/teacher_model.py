"""
Laboratorio 5.3 - Modelo Teacher (ResNet50V2)
===============================================

Este script implementa el modelo teacher grande que servirá como base
para las técnicas de pruning, quantization y distillation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import time

class TeacherModel:
    """
    Clase para crear y entrenar el modelo teacher (ResNet50V2)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 32, 3), 
                 num_classes: int = 10):
        """
        Inicializa el modelo teacher
        
        Args:
            input_shape: Forma de entrada de las imágenes
            num_classes: Número de clases
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self) -> tf.keras.Model:
        """
        Construye el modelo teacher usando ResNet50V2
        
        Returns:
            Modelo TensorFlow compilado
        """
        print("🏗️ Construyendo modelo teacher (ResNet50V2)...")
        
        # Cargar ResNet50V2 pre-entrenado (sin top layer)
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        # Congelar capas base inicialmente
        base_model.trainable = False
        
        # Agregar capas de clasificación
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Data augmentation
        x = tf.keras.layers.RandomFlip('horizontal')(inputs)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
        
        # Preprocesamiento para ResNet50V2
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        
        # Pasar por el modelo base
        x = base_model(x, training=False)
        
        # Capas de clasificación
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Crear modelo
        self.model = tf.keras.Model(inputs, outputs, name='teacher_model')
        
        print(f"✅ Modelo teacher construido:")
        self.model.summary()
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compila el modelo con optimizador y métricas
        
        Args:
            learning_rate: Tasa de aprendizaje
        """
        print("⚙️ Compilando modelo teacher...")
        
        # Optimizador con learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Métricas
        metrics = [
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        print("✅ Modelo compilado exitosamente")
    
    def train_model(self, train_dataset: tf.data.Dataset, 
                   val_dataset: tf.data.Dataset,
                   epochs: int = 50,
                   fine_tune_epochs: int = 20) -> Dict[str, Any]:
        """
        Entrena el modelo teacher en dos fases
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            epochs: Épocas de entrenamiento inicial
            fine_tune_epochs: Épocas de fine-tuning
            
        Returns:
            Diccionario con historial de entrenamiento
        """
        print("🎓 Iniciando entrenamiento del modelo teacher...")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'teacher_model_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Fase 1: Entrenamiento con capas base congeladas
        print(f"\n📚 Fase 1: Entrenamiento inicial ({epochs} épocas)")
        
        start_time = time.time()
        history_phase1 = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        phase1_time = time.time() - start_time
        
        # Fase 2: Fine-tuning
        print(f"\n🔧 Fase 2: Fine-tuning ({fine_tune_epochs} épocas)")
        
        # Descongelar capas base
        base_model = self.model.layers[4]  # ResNet50V2 base
        base_model.trainable = True
        
        # Recompilar con learning rate más bajo
        self.compile_model(learning_rate=1e-4)
        
        start_time = time.time()
        history_phase2 = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            verbose=1
        )
        phase2_time = time.time() - start_time
        
        # Combinar historiales
        self.history = {
            'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
            'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
            'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
            'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
            'top_5_accuracy': history_phase1.history['top_5_accuracy'] + history_phase2.history['top_5_accuracy'],
            'val_top_5_accuracy': history_phase1.history['val_top_5_accuracy'] + history_phase2.history['val_top_5_accuracy']
        }
        
        # Evaluar modelo final
        print("\n📊 Evaluación final del modelo teacher:")
        final_metrics = self.model.evaluate(val_dataset, verbose=1)
        
        results = {
            'history': self.history,
            'final_metrics': final_metrics,
            'training_time': phase1_time + phase2_time,
            'phase1_time': phase1_time,
            'phase2_time': phase2_time
        }
        
        print(f"✅ Entrenamiento completado:")
        print(f"   - Accuracy final: {final_metrics[1]:.4f}")
        print(f"   - Top-5 Accuracy: {final_metrics[2]:.4f}")
        print(f"   - Tiempo total: {results['training_time']:.2f} segundos")
        
        return results
    
    def evaluate_model(self, test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Evalúa el modelo en el dataset de prueba
        
        Args:
            test_dataset: Dataset de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        print("🧪 Evaluando modelo teacher en dataset de prueba...")
        
        # Evaluar
        metrics = self.model.evaluate(test_dataset, verbose=1)
        
        # Predicciones para análisis adicional
        predictions = []
        true_labels = []
        
        for batch_images, batch_labels in test_dataset:
            batch_preds = self.model.predict(batch_images, verbose=0)
            predictions.extend(batch_preds)
            true_labels.extend(batch_labels.numpy())
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Calcular métricas adicionales
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(true_labels, axis=1)
        
        # Accuracy por clase
        class_accuracy = {}
        for i in range(self.num_classes):
            mask = true_classes == i
            if np.sum(mask) > 0:
                class_acc = np.mean(pred_classes[mask] == true_classes[mask])
                class_accuracy[i] = class_acc
        
        results = {
            'test_loss': metrics[0],
            'test_accuracy': metrics[1],
            'test_top_5_accuracy': metrics[2],
            'class_accuracy': class_accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        print(f"✅ Evaluación completada:")
        print(f"   - Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"   - Test Top-5 Accuracy: {results['test_top_5_accuracy']:.4f}")
        
        return results
    
    def plot_training_history(self):
        """
        Visualiza el historial de entrenamiento
        """
        if self.history is None:
            print("⚠️ No hay historial de entrenamiento disponible")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history['loss'], label='Training')
        axes[0, 1].plot(self.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-5 Accuracy
        axes[1, 0].plot(self.history['top_5_accuracy'], label='Training')
        axes[1, 0].plot(self.history['val_top_5_accuracy'], label='Validation')
        axes[1, 0].set_title('Top-5 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-5 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning curves (accuracy vs loss)
        axes[1, 1].plot(self.history['accuracy'], self.history['loss'], 'b-', alpha=0.5)
        axes[1, 1].plot(self.history['val_accuracy'], self.history['val_loss'], 'r-', alpha=0.5)
        axes[1, 1].set_title('Learning Curves')
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend(['Training', 'Validation'])
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str = 'teacher_model_final.h5'):
        """
        Guarda el modelo entrenado
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        print(f"💾 Guardando modelo teacher en {filepath}...")
        self.model.save(filepath)
        print("✅ Modelo guardado exitosamente")
    
    def load_model(self, filepath: str = 'teacher_model_final.h5'):
        """
        Carga un modelo entrenado
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        print(f"📂 Cargando modelo teacher desde {filepath}...")
        self.model = tf.keras.models.load_model(filepath)
        print("✅ Modelo cargado exitosamente")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada del modelo
        
        Returns:
            Diccionario con información del modelo
        """
        if self.model is None:
            print("⚠️ No hay modelo construido")
            return {}
        
        # Contar parámetros
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) 
                              for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Calcular tamaño estimado
        param_size = total_params * 4  # 4 bytes por float32
        model_size_mb = param_size / (1024 * 1024)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'model_size_mb': model_size_mb,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'num_layers': len(self.model.layers)
        }
        
        print(f"📊 Información del modelo teacher:")
        print(f"   - Parámetros totales: {total_params:,}")
        print(f"   - Parámetros entrenables: {trainable_params:,}")
        print(f"   - Tamaño estimado: {model_size_mb:.2f} MB")
        print(f"   - Capas: {info['num_layers']}")
        
        return info


def main():
    """
    Función principal para probar el modelo teacher
    """
    print("🚀 Iniciando prueba del modelo teacher")
    
    # Crear datos de prueba (simulados)
    from load_cifar10 import CIFAR10DataLoader
    
    loader = CIFAR10DataLoader(batch_size=32)
    datasets = loader.create_datasets()
    
    # Crear y entrenar modelo teacher
    teacher = TeacherModel()
    teacher.build_model()
    teacher.compile_model()
    
    # Entrenar (con menos épocas para prueba)
    results = teacher.train_model(
        datasets['train_dataset'],
        datasets['val_dataset'],
        epochs=5,  # Reducido para prueba
        fine_tune_epochs=3
    )
    
    # Evaluar
    eval_results = teacher.evaluate_model(datasets['test_dataset'])
    
    # Visualizar resultados
    teacher.plot_training_history()
    
    # Guardar modelo
    teacher.save_model()
    
    print("✅ Prueba completada exitosamente")


if __name__ == "__main__":
    main()
