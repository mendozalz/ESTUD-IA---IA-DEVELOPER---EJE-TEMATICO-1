# -*- coding: utf-8 -*-
"""
Detector de Fraudes Multimodal - Versión Simplificada
Sin dependencias pesadas para evitar errores de instalación
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

class SimpleMultimodalFusion(layers.Layer):
    """Capa simplificada para fusión multimodal"""
    
    def __init__(self, fusion_units=64, **kwargs):
        super().__init__(**kwargs)
        self.fusion_units = fusion_units
        
    def build(self, input_shape):
        # Asumimos 3 modalidades: texto, imagen, grafo
        self.text_dense = layers.Dense(self.fusion_units, activation='relu')
        self.image_dense = layers.Dense(self.fusion_units, activation='relu')
        self.graph_dense = layers.Dense(self.fusion_units, activation='relu')
        
        # Capa de atención para fusionar
        self.attention = layers.Dense(3, activation='softmax')  # 3 modalidades
        
    def call(self, inputs):
        text_features, image_features, graph_features = inputs
        
        # Procesar cada modalidad
        text_out = self.text_dense(text_features)
        image_out = self.image_dense(image_features)
        graph_out = self.graph_dense(graph_features)
        
        # Concatenar características
        combined = tf.concat([text_out, image_out, graph_out], axis=-1)
        
        # Aplicar atención ponderada
        attention_weights = self.attention(tf.reduce_mean(combined, axis=-1, keepdims=True))
        
        # Fusionar con pesos de atención
        fused = (attention_weights[:, 0:1] * text_out + 
                 attention_weights[:, 1:2] * image_out + 
                 attention_weights[:, 2:3] * graph_out)
        
        return fused

class FraudDetectionModel:
    """Modelo completo de detección de fraudes multimodal"""
    
    def __init__(self):
        self.model = None
        self.fusion_layer = SimpleMultimodalFusionLayer()
        
    def build_model(self, text_shape=(100,), image_shape=(224, 224, 3), graph_shape=(50,)):
        """Construir el modelo multimodal"""
        
        # Input para cada modalidad
        text_input = layers.Input(shape=text_shape, name='text_input')
        image_input = layers.Input(shape=image_shape, name='image_input')
        graph_input = layers.Input(shape=graph_shape, name='graph_input')
        
        # Procesamiento individual
        text_features = layers.Dense(32, activation='relu')(text_input)
        text_features = layers.Dropout(0.2)(text_features)
        
        image_features = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        image_features = layers.MaxPooling2D((2, 2))(image_features)
        image_features = layers.Flatten()(image_features)
        image_features = layers.Dense(32, activation='relu')(image_features)
        
        graph_features = layers.Dense(32, activation='relu')(graph_input)
        graph_features = layers.Dropout(0.2)(graph_features)
        
        # Fusión multimodal
        fused_features = self.fusion_layer([text_features, image_features, graph_features])
        
        # Clasificación final
        x = layers.Dense(64, activation='relu')(fused_features)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid', name='fraud_output')(x)
        
        # Crear modelo
        model = Model(
            inputs=[text_input, image_input, graph_input],
            outputs=output,
            name='multimodal_fraud_detector'
        )
        
        self.model = model
        return model
    
    def compile_model(self):
        """Compilar el modelo"""
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generar datos sintéticos para demostración"""
        print(f"Generando {num_samples} muestras sintéticas...")
        
        # Datos de texto (descripciones de transacciones)
        text_data = np.random.randn(num_samples, 100)
        
        # Datos de imagen (simulación de imágenes de cheques)
        image_data = np.random.randn(num_samples, 224, 224, 3)
        
        # Datos de grafo (patrones de transacciones)
        graph_data = np.random.randn(num_samples, 50)
        
        # Labels (0: normal, 1: fraude)
        labels = np.random.randint(0, 2, num_samples)
        
        return [text_data, image_data, graph_data], labels
    
    def train_model(self, epochs=10):
        """Entrenar el modelo"""
        print("Iniciando entrenamiento del detector de fraudes...")
        
        # Generar datos
        (text_data, image_data, graph_data), labels = self.generate_synthetic_data(500)
        
        # Entrenar
        history = self.model.fit(
            [text_data, image_data, graph_data],
            labels,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self):
        """Evaluar el modelo"""
        print("Evaluando modelo...")
        
        # Generar datos de prueba
        (text_data, image_data, graph_data), labels = self.generate_synthetic_data(100)
        
        # Evaluar
        results = self.model.evaluate(
            [text_data, image_data, graph_data],
            labels,
            verbose=0
        )
        
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
        
        return results
    
    def predict_fraud(self, text_sample, image_sample, graph_sample):
        """Predecir si es fraude"""
        # Preparar datos
        text_data = np.array([text_sample])
        image_data = np.array([image_sample])
        graph_data = np.array([graph_sample])
        
        # Predecir
        prediction = self.model.predict(
            [text_data, image_data, graph_data],
            verbose=0
        )
        
        fraud_probability = prediction[0][0]
        is_fraud = fraud_probability > 0.5
        
        return {
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(is_fraud),
            'prediction': 'FRAUDE' if is_fraud else 'NORMAL'
        }
    
    def visualize_training_history(self, history):
        """Visualizar historial de entrenamiento"""
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Precision
        plt.subplot(1, 3, 3)
        plt.plot(history.history['precision'], label='Train')
        plt.plot(history.history['val_precision'], label='Validation')
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='multimodal_fraud_detector.h5'):
        """Guardar el modelo"""
        self.model.save(filename)
        print(f"Modelo guardado como {filename}")

def main():
    """Función principal de demostración"""
    print("=" * 60)
    print("DETECTOR DE FRAUDES MULTIMODAL")
    print("=" * 60)
    
    # Crear detector
    detector = FraudDetectionModel()
    
    # Construir modelo
    print("\n1. Construyendo modelo multimodal...")
    model = detector.build_model()
    detector.compile_model()
    model.summary()
    
    # Entrenar
    print("\n2. Entrenando modelo...")
    history = detector.train_model(epochs=5)
    
    # Evaluar
    print("\n3. Evaluando modelo...")
    results = detector.evaluate_model()
    
    # Visualizar
    print("\n4. Generando visualizaciones...")
    detector.visualize_training_history(history)
    
    # Probar predicción
    print("\n5. Probando predicción...")
    test_text = np.random.randn(100)
    test_image = np.random.randn(224, 224, 3)
    test_graph = np.random.randn(50)
    
    prediction = detector.predict_fraud(test_text, test_image, test_graph)
    print(f"   Probabilidad de fraude: {prediction['fraud_probability']:.4f}")
    print(f"   Predicción: {prediction['prediction']}")
    
    # Guardar modelo
    print("\n6. Guardando modelo...")
    detector.save_model()
    
    print("\n" + "=" * 60)
    print("DETECTOR MULTIMODAL COMPLETADO")
    print("=" * 60)
    print("🎯 RESULTADOS:")
    print(f"   • Accuracy final: {results[1]:.4f}")
    print(f"   • Precision: {results[2]:.4f}")
    print(f"   • Recall: {results[3]:.4f}")
    print(f"   • Modelo guardado: multimodal_fraud_detector.h5")
    print("=" * 60)

if __name__ == "__main__":
    main()
