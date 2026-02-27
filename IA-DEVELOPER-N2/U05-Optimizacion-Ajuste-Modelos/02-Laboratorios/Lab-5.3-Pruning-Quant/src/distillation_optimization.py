"""
Laboratorio 5.3 - Knowledge Distillation Optimization
=====================================================

Este script implementa técnicas de knowledge distillation para
crear un modelo estudiante más pequeño a partir de un teacher.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
import time
import tempfile

class DistillationOptimizer:
    """
    Clase para aplicar técnicas de knowledge distillation
    """
    
    def __init__(self, teacher_model: tf.keras.Model, 
                 temperature: float = 3.0, alpha: float = 0.5):
        """
        Inicializa el optimizador de distillation
        
        Args:
            teacher_model: Modelo teacher entrenado
            temperature: Temperatura para soft targets
            alpha: Peso para balancear loss (teacher vs student)
        """
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.student_model = None
        self.distillation_history = None
        
    def create_student_model(self, input_shape: Tuple[int, int, int] = (32, 32, 3),
                           num_classes: int = 10) -> tf.keras.Model:
        """
        Crea un modelo estudiante más pequeño y eficiente
        
        Args:
            input_shape: Forma de entrada
            num_classes: Número de clases
            
        Returns:
            Modelo estudiante
        """
        print("🎓 Creando modelo estudiante...")
        
        # Arquitectura más pequeña que el teacher
        inputs = tf.keras.Input(shape=input_shape)
        
        # Data augmentation
        x = tf.keras.layers.RandomFlip('horizontal')(inputs)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        
        # Bloque convolacional 1
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        # Bloque convolacional 2
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        # Bloque convolacional 3
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        
        # Clasificación
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.student_model = tf.keras.Model(inputs, outputs, name='student_model')
        
        print(f"✅ Modelo estudiante creado:")
        self.student_model.summary()
        
        return self.student_model
    
    def distillation_loss(self, y_true, y_pred, teacher_pred):
        """
        Calcula la loss de distillation
        
        Args:
            y_true: Labels verdaderos
            y_pred: Predicciones del estudiante
            teacher_pred: Predicciones del teacher
            
        Returns:
            Loss combinada
        """
        # Loss con labels verdaderos (hard targets)
        student_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Loss con soft targets del teacher
        teacher_soft = tf.nn.softmax(teacher_pred / self.temperature)
        student_soft = tf.nn.softmax(y_pred / self.temperature)
        
        distillation_loss = tf.keras.losses.categorical_crossentropy(
            teacher_soft, student_soft
        )
        
        # Combinar losses
        total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        return total_loss
    
    def train_student_with_distillation(self, train_dataset: tf.data.Dataset,
                                       val_dataset: tf.data.Dataset,
                                       epochs: int = 20) -> Dict[str, Any]:
        """
        Entrena el modelo estudiante con distillation
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            epochs: Número de épocas
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("🎓 Iniciando entrenamiento con Knowledge Distillation...")
        
        if self.student_model is None:
            print("⚠️ No hay modelo estudiante creado")
            return {}
        
        # Compilar modelo con custom loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Métricas
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        val_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        # Historial
        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Entrenamiento personalizado
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss = 0
            train_accuracy.reset_states()
            
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                # Obtener predicciones del teacher
                teacher_pred = self.teacher_model(x_batch, training=False)
                
                with tf.GradientTape() as tape:
                    # Predicciones del estudiante
                    student_pred = self.student_model(x_batch, training=True)
                    
                    # Calcular loss
                    loss = self.distillation_loss(y_batch, student_pred, teacher_pred)
                
                # Calcular gradientes
                gradients = tape.gradient(loss, self.student_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
                
                # Actualizar métricas
                train_loss += loss
                train_accuracy.update_state(y_batch, student_pred)
                
                if step % 100 == 0:
                    print(f"Step {step}: loss={loss:.4f}, acc={train_accuracy.result():.4f}")
            
            # Validation
            val_loss = 0
            val_accuracy.reset_states()
            
            for x_batch, y_batch in val_dataset:
                teacher_pred = self.teacher_model(x_batch, training=False)
                student_pred = self.student_model(x_batch, training=False)
                
                loss = self.distillation_loss(y_batch, student_pred, teacher_pred)
                val_loss += loss
                val_accuracy.update_state(y_batch, student_pred)
            
            # Guardar historial
            avg_train_loss = train_loss / len(train_dataset)
            avg_val_loss = val_loss / len(val_dataset)
            
            history['loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['accuracy'].append(train_accuracy.result())
            history['val_accuracy'].append(val_accuracy.result())
            
            print(f"Epoch {epoch + 1}: "
                  f"train_loss={avg_train_loss:.4f}, train_acc={train_accuracy.result():.4f}, "
                  f"val_loss={avg_val_loss:.4f}, val_acc={val_accuracy.result():.4f}")
            
            # Early stopping
            if len(history['val_accuracy']) > 5:
                recent_acc = history['val_accuracy'][-5:]
                if max(recent_acc) == recent_acc[0]:  # No mejora
                    print("Early stopping triggered")
                    break
        
        self.distillation_history = history
        
        print("✅ Entrenamiento con distillation completado")
        
        return {
            'history': history,
            'final_train_accuracy': history['accuracy'][-1],
            'final_val_accuracy': history['val_accuracy'][-1]
        }
    
    def train_student_baseline(self, train_dataset: tf.data.Dataset,
                              val_dataset: tf.data.Dataset,
                              epochs: int = 20) -> Dict[str, Any]:
        """
        Entrena el modelo estudiante sin distillation (baseline)
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            epochs: Número de épocas
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print("🎓 Entrenando modelo estudiante baseline (sin distillation)...")
        
        if self.student_model is None:
            print("⚠️ No hay modelo estudiante creado")
            return {}
        
        # Compilar modelo normalmente
        self.student_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Entrenar
        start_time = time.time()
        history = self.student_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        print(f"✅ Entrenamiento baseline completado en {training_time:.2f} segundos")
        
        return {
            'history': history.history,
            'training_time': training_time
        }
    
    def compare_models(self, test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Compara teacher, student con distillation y student baseline
        
        Args:
            test_dataset: Dataset de prueba
            
        Returns:
            Diccionario con comparación de modelos
        """
        print("📊 Comparando modelos...")
        
        # Evaluar teacher
        print("🔍 Evaluando modelo teacher...")
        teacher_metrics = self.teacher_model.evaluate(test_dataset, verbose=1)
        
        # Evaluar student con distillation
        print("🔍 Evaluando modelo estudiante (con distillation)...")
        student_dist_metrics = self.student_model.evaluate(test_dataset, verbose=1)
        
        # Crear y entrenar baseline student
        print("🔍 Creando y entrenando baseline student...")
        baseline_student = tf.keras.models.clone_model(self.student_model)
        baseline_student.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar baseline brevemente
        baseline_history = baseline_student.fit(
            test_dataset.take(10),  # Solo para prueba
            epochs=2,
            verbose=0
        )
        
        print("🔍 Evaluando baseline student...")
        baseline_metrics = baseline_student.evaluate(test_dataset, verbose=1)
        
        # Calcular información de modelos
        teacher_params = self.teacher_model.count_params()
        student_params = self.student_model.count_params()
        baseline_params = baseline_student.count_params()
        
        def get_model_size(model):
            with tempfile.NamedTemporaryFile() as tmp:
                model.save(tmp.name, save_format='h5')
                size = tmp.tell()
            return size / (1024 * 1024)  # MB
        
        teacher_size = get_model_size(self.teacher_model)
        student_size = get_model_size(self.student_model)
        baseline_size = get_model_size(baseline_student)
        
        comparison = {
            'teacher_model': {
                'accuracy': teacher_metrics[1],
                'loss': teacher_metrics[0],
                'parameters': teacher_params,
                'size_mb': teacher_size
            },
            'student_distilled': {
                'accuracy': student_dist_metrics[1],
                'loss': student_dist_metrics[0],
                'parameters': student_params,
                'size_mb': student_size
            },
            'student_baseline': {
                'accuracy': baseline_metrics[1],
                'loss': baseline_metrics[0],
                'parameters': baseline_params,
                'size_mb': baseline_size
            },
            'improvements': {
                'parameter_reduction_vs_teacher': (teacher_params - student_params) / teacher_params * 100,
                'size_reduction_vs_teacher': (teacher_size - student_size) / teacher_size * 100,
                'distillation_vs_baseline_accuracy': student_dist_metrics[1] - baseline_metrics[1],
                'teacher_vs_student_accuracy': teacher_metrics[1] - student_dist_metrics[1]
            }
        }
        
        print("✅ Comparación completada:")
        print(f"   - Reducción de parámetros vs Teacher: {comparison['improvements']['parameter_reduction_vs_teacher']:.1f}%")
        print(f"   - Reducción de tamaño vs Teacher: {comparison['improvements']['size_reduction_vs_teacher']:.1f}%")
        print(f"   - Mejora distillation vs baseline: {comparison['improvements']['distillation_vs_baseline_accuracy']:.4f}")
        print(f"   - Pérdida vs teacher: {comparison['improvements']['teacher_vs_student_accuracy']:.4f}")
        
        return comparison
    
    def analyze_knowledge_transfer(self, test_dataset: tf.data.Dataset,
                                 num_samples: int = 100) -> Dict[str, Any]:
        """
        Analiza la transferencia de conocimiento entre teacher y student
        
        Args:
            test_dataset: Dataset de prueba
            num_samples: Número de muestras para análisis
            
        Returns:
            Diccionario con análisis de transferencia
        """
        print("🔍 Analizando transferencia de conocimiento...")
        
        # Obtener predicciones
        teacher_predictions = []
        student_predictions = []
        true_labels = []
        
        samples_processed = 0
        
        for batch_images, batch_labels in test_dataset:
            if samples_processed >= num_samples:
                break
            
            batch_size = batch_images.shape[0]
            samples_to_take = min(batch_size, num_samples - samples_processed)
            
            # Predicciones
            teacher_pred = self.teacher_model.predict(batch_images[:samples_to_take], verbose=0)
            student_pred = self.student_model.predict(batch_images[:samples_to_take], verbose=0)
            
            teacher_predictions.extend(teacher_pred)
            student_predictions.extend(student_pred)
            true_labels.extend(batch_labels[:samples_to_take])
            
            samples_processed += samples_to_take
        
        # Convertir a arrays
        teacher_predictions = np.array(teacher_predictions)
        student_predictions = np.array(student_predictions)
        true_labels = np.array(true_labels)
        
        # Calcular métricas de transferencia
        teacher_classes = np.argmax(teacher_predictions, axis=1)
        student_classes = np.argmax(student_predictions, axis=1)
        true_classes = np.argmax(true_labels, axis=1)
        
        # Agreement entre teacher y student
        agreement = np.mean(teacher_classes == student_classes)
        
        # Accuracy de cada modelo
        teacher_accuracy = np.mean(teacher_classes == true_classes)
        student_accuracy = np.mean(student_classes == true_classes)
        
        # Confidence analysis
        teacher_confidence = np.max(teacher_predictions, axis=1)
        student_confidence = np.max(student_predictions, axis=1)
        
        # Correlación de predicciones
        correlation_matrix = np.corrcoef(teacher_predictions.flatten(), 
                                        student_predictions.flatten())
        correlation = correlation_matrix[0, 1]
        
        # Cases where student corrects teacher
        teacher_wrong = teacher_classes != true_classes
        student_correct = student_classes == true_classes
        corrections = np.sum(teacher_wrong & student_correct)
        correction_rate = corrections / np.sum(teacher_wrong) if np.sum(teacher_wrong) > 0 else 0
        
        analysis = {
            'agreement_rate': agreement,
            'teacher_accuracy': teacher_accuracy,
            'student_accuracy': student_accuracy,
            'teacher_confidence_mean': np.mean(teacher_confidence),
            'student_confidence_mean': np.mean(student_confidence),
            'prediction_correlation': correlation,
            'teacher_to_student_corrections': corrections,
            'correction_rate': correction_rate,
            'num_samples': len(true_labels)
        }
        
        print("✅ Análisis de transferencia completado:")
        print(f"   - Agreement rate: {agreement:.3f}")
        print(f"   - Teacher accuracy: {teacher_accuracy:.3f}")
        print(f"   - Student accuracy: {student_accuracy:.3f}")
        print(f"   - Correlation: {correlation:.3f}")
        print(f"   - Student corrections: {corrections}/{np.sum(teacher_wrong)} ({correction_rate:.1%})")
        
        return analysis
    
    def plot_distillation_results(self, comparison_results: Dict[str, Any],
                                transfer_analysis: Dict[str, Any]):
        """
        Visualiza resultados de distillation
        
        Args:
            comparison_results: Resultados de comparación
            transfer_analysis: Análisis de transferencia
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Accuracy comparison
        models = ['Teacher', 'Student (Distilled)', 'Student (Baseline)']
        accuracies = [
            comparison_results['teacher_model']['accuracy'],
            comparison_results['student_distilled']['accuracy'],
            comparison_results['student_baseline']['accuracy']
        ]
        
        bars = axes[0, 0].bar(models, accuracies, color=['blue', 'green', 'orange'])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Añadir valores en barras
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # Parameter count
        params = [
            comparison_results['teacher_model']['parameters'],
            comparison_results['student_distilled']['parameters'],
            comparison_results['student_baseline']['parameters']
        ]
        
        axes[0, 1].bar(models, params, color=['blue', 'green', 'orange'])
        axes[0, 1].set_title('Parameter Count')
        axes[0, 1].set_ylabel('Parameters')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Model size
        sizes = [
            comparison_results['teacher_model']['size_mb'],
            comparison_results['student_distilled']['size_mb'],
            comparison_results['student_baseline']['size_mb']
        ]
        
        axes[0, 2].bar(models, sizes, color=['blue', 'green', 'orange'])
        axes[0, 2].set_title('Model Size')
        axes[0, 2].set_ylabel('Size (MB)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Training history (si existe)
        if self.distillation_history:
            epochs = range(1, len(self.distillation_history['accuracy']) + 1)
            
            axes[1, 0].plot(epochs, self.distillation_history['accuracy'], 'b-', label='Train')
            axes[1, 0].plot(epochs, self.distillation_history['val_accuracy'], 'r-', label='Val')
            axes[1, 0].set_title('Distillation Training History')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Transfer metrics
        if transfer_analysis:
            metrics = ['Agreement', 'Teacher Acc', 'Student Acc', 'Correlation']
            values = [
                transfer_analysis['agreement_rate'],
                transfer_analysis['teacher_accuracy'],
                transfer_analysis['student_accuracy'],
                transfer_analysis['prediction_correlation']
            ]
            
            axes[1, 1].bar(metrics, values, color=['purple', 'blue', 'green', 'orange'])
            axes[1, 1].set_title('Knowledge Transfer Metrics')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylim(0, 1)
        
        # Confidence comparison
        if transfer_analysis:
            confidences = [
                transfer_analysis['teacher_confidence_mean'],
                transfer_analysis['student_confidence_mean']
            ]
            
            axes[1, 2].bar(['Teacher', 'Student'], confidences, color=['blue', 'green'])
            axes[1, 2].set_title('Average Prediction Confidence')
            axes[1, 2].set_ylabel('Confidence')
            axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def save_student_model(self, filepath: str = 'student_model_distilled.h5'):
        """
        Guarda el modelo estudiante entrenado
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        if self.student_model is None:
            print("⚠️ No hay modelo estudiante para guardar")
            return
        
        print(f"💾 Guardando modelo estudiante en {filepath}...")
        self.student_model.save(filepath)
        print("✅ Modelo guardado exitosamente")


def main():
    """
    Función principal para probar knowledge distillation
    """
    print("🚀 Iniciando prueba de knowledge distillation")
    
    # Cargar modelo teacher
    from teacher_model import TeacherModel
    from load_cifar10 import CIFAR10DataLoader
    
    # Crear datos
    loader = CIFAR10DataLoader(batch_size=32)
    datasets = loader.create_datasets()
    
    # Crear y entrenar modelo teacher simplificado
    teacher = TeacherModel()
    teacher.build_model()
    teacher.compile_model()
    
    # Entrenar brevemente
    teacher.train_model(
        datasets['train_dataset'],
        datasets['val_dataset'],
        epochs=2,
        fine_tune_epochs=1
    )
    
    # Aplicar knowledge distillation
    distiller = DistillationOptimizer(teacher.model)
    student_model = distiller.create_student_model()
    
    # Entrenar con distillation
    distillation_results = distiller.train_student_with_distillation(
        datasets['train_dataset'],
        datasets['val_dataset'],
        epochs=5
    )
    
    # Comparar modelos
    comparison = distiller.compare_models(datasets['test_dataset'])
    
    # Analizar transferencia de conocimiento
    transfer_analysis = distiller.analyze_knowledge_transfer(datasets['test_dataset'])
    
    # Visualizar resultados
    distiller.plot_distillation_results(comparison, transfer_analysis)
    
    # Guardar modelo
    distiller.save_student_model()
    
    print("✅ Prueba de knowledge distillation completada exitosamente")


if __name__ == "__main__":
    main()
