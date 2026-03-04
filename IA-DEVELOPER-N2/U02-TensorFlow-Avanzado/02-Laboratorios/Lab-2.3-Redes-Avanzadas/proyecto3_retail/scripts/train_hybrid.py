# -*- coding: utf-8 -*-
"""
Proyecto 3: Automatización en Retail
Entrenamiento de Red Híbrida (CNN + RNN) para recomendación de productos
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import cv2

class HybridProductRecommender:
    """Clase para sistema de recomendación híbrido (CNN + RNN)"""
    
    def __init__(self, vocab_size=10000, max_seq_length=100, image_size=(224, 224)):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.history = None
        
    def load_product_data(self, csv_path="data/processed/products.csv"):
        """Cargar datos de productos"""
        
        # Datos simulados si no existe el archivo
        if not os.path.exists(csv_path):
            print("📝 Generando datos simulados de productos...")
            
            # Categorías de productos
            categories = ['Electrónica', 'Ropa', 'Hogar', 'Deportes', 'Libros']
            
            # Descripciones por categoría
            descriptions = {
                'Electrónica': [
                    'Smartphone con pantalla de 6 pulgadas y cámara de 48MP',
                    'Laptop ultraligera con procesador de última generación',
                    'Auriculares inalámbricos con cancelación de ruido',
                    'Tablet de 10 pulgadas ideal para trabajo y entretenimiento',
                    'Smartwatch con monitor de actividad y GPS'
                ],
                'Ropa': [
                    'Camiseta de algodón orgánico talla M color azul',
                    'Pantalón vaquero slim fit para uso casual',
                    'Chaqueta impermeable con capucha y cremallera',
                    'Vestido elegante para ocasiones especiales',
                    'Zapatillas deportivas cómodas para correr'
                ],
                'Hogar': [
                    'Juego de sartenes antiadherentes 3 piezas',
                    'Lámpara LED inteligente con control remoto',
                    'Set de toallas de baño de alta calidad',
                    'Olla de cocción lenta programable',
                    'Organizador de cocina con múltiples compartimentos'
                ],
                'Deportes': [
                    'Bicicleta estática con monitor de ritmo cardíaco',
                    'Set de mancuernas ajustables de 5 a 25 kg',
                    'Pelota de yoga profesional con bomba incluida',
                    'Cinta de correr plegable con inclinación ajustable',
                    'Botella de agua térmica para deportes'
                ],
                'Libros': [
                    'Novela de ciencia ficción galardonada internacionalmente',
                    'Guía práctica de programación Python para principiantes',
                    'Libro de cocina recetas saludables y fáciles',
                    'Biografía de líder empresarial inspirador',
                    'Manual de meditación y mindfulness'
                ]
            }
            
            # Generar productos
            products = []
            product_id = 1
            
            for category in categories:
                for i in range(100):  # 100 productos por categoría
                    desc_idx = i % len(descriptions[category])
                    products.append({
                        'id': product_id,
                        'name': f'Producto {product_id}',
                        'category': category,
                        'description': descriptions[category][desc_idx],
                        'price': np.random.uniform(10, 500),
                        'rating': np.random.uniform(3.0, 5.0),
                        'image_path': f'data/products/{category.lower()}_{product_id}.jpg'
                    })
                    product_id += 1
            
            df = pd.DataFrame(products)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
        else:
            df = pd.read_csv(csv_path)
            
        print(f"📊 Dataset cargado: {len(df)} productos")
        print(f"   • Categorías: {df['category'].nunique()}")
        print(f"   • Productos por categoría: {df.groupby('category').size().to_dict()}")
        
        return df
    
    def preprocess_text_data(self, df):
        """Preprocesar datos de texto"""
        
        # Inicializar tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['description'])
        
        # Convertir textos a secuencias
        sequences = self.tokenizer.texts_to_sequences(df['description'])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_seq_length)
        
        return padded_sequences
    
    def preprocess_image_data(self, df):
        """Preprocesar datos de imágenes (simulado)"""
        
        # En un caso real, cargaríamos las imágenes reales
        # Por ahora, generamos imágenes simuladas
        num_products = len(df)
        images = np.random.rand(num_products, *self.image_size, 3)
        
        # Simular diferentes características por categoría
        categories = df['category'].unique()
        for i, category in enumerate(categories):
            mask = df['category'] == category
            # Añadir "patrones" diferentes por categoría
            images[mask] = images[mask] * (0.5 + i * 0.1)
        
        return images
    
    def preprocess_labels(self, df):
        """Preprocesar etiquetas (categorías)"""
        
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(df['category'])
        
        # Convertir a one-hot encoding
        num_classes = len(self.label_encoder.classes_)
        labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        return labels_one_hot, num_classes
    
    def build_hybrid_model(self, num_classes):
        """Construir modelo híbrido (CNN + RNN)"""
        
        # Rama de imágenes (CNN)
        image_input = layers.Input(shape=(*self.image_size, 3), name='image_input')
        
        # Bloques convolucionales
        x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        image_features = layers.Dense(256, activation='relu')(x)
        
        # Rama de texto (RNN)
        text_input = layers.Input(shape=(self.max_seq_length,), name='text_input')
        y = layers.Embedding(self.vocab_size, 128)(text_input)
        y = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(y)
        y = layers.Bidirectional(layers.LSTM(32))(y)
        y = layers.Dense(128, activation='relu')(y)
        y = layers.Dropout(0.3)(y)
        text_features = layers.Dense(256, activation='relu')(y)
        
        # Combinar características
        combined = layers.Concatenate()([image_features, text_features])
        
        # Capas densas finales
        z = layers.Dense(512, activation='relu')(combined)
        z = layers.Dropout(0.5)(z)
        z = layers.Dense(256, activation='relu')(z)
        z = layers.Dropout(0.3)(z)
        output = layers.Dense(num_classes, activation='softmax')(z)
        
        # Crear modelo
        model = models.Model(
            inputs=[image_input, text_input],
            outputs=output,
            name='HybridProductRecommender'
        )
        
        # Compilar
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        self.model = model
        return model
    
    def train_model(self, images, texts, labels, epochs=20):
        """Entrenar el modelo híbrido"""
        
        # Dividir datos
        (train_images, test_images, 
         train_texts, test_texts, 
         train_labels, test_labels) = train_test_split(
            images, texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            tf.keras.callbacks.ModelCheckpoint(
                'models/hybrid_best_model.h5', save_best_only=True
            )
        ]
        
        # Entrenamiento
        history = self.model.fit(
            [train_images, train_texts],
            train_labels,
            validation_data=([test_images, test_texts], test_labels),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        return history, (test_images, test_texts, test_labels)
    
    def evaluate_model(self, test_images, test_texts, test_labels):
        """Evaluar el modelo"""
        
        results = self.model.evaluate(
            [test_images, test_texts], test_labels, verbose=0
        )
        
        metrics_names = self.model.metrics_names
        
        print("\n📊 Resultados de Evaluación:")
        for name, value in zip(metrics_names, results):
            print(f"   {name}: {value:.4f}")
        
        return dict(zip(metrics_names, results))
    
    def plot_training_history(self):
        """Visualizar historial de entrenamiento"""
        
        if not self.history:
            print("⚠️ No hay historial de entrenamiento disponible")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Top-3 Accuracy
        axes[1, 0].plot(self.history.history['top_k_categorical_accuracy'], label='Train')
        axes[1, 0].plot(self.history.history['val_top_k_categorical_accuracy'], label='Validation')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        
        # Learning Rate (si está disponible)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNo disponible', 
                            ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('notebooks/hybrid_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def predict_category(self, image, text, top_k=3):
        """Predecir categoría para un nuevo producto"""
        
        if not self.model:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train_model primero.")
        
        # Preprocesar imagen (simulado)
        if image is None:
            image = np.random.rand(1, *self.image_size, 3)
        else:
            image = np.expand_dims(image, axis=0)
        
        # Preprocesar texto
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_seq_length)
        
        # Predecir
        predictions = self.model.predict([image, padded_sequence], verbose=0)
        
        # Obtener top-k predicciones
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_categories = self.label_encoder.inverse_transform(top_indices)
        top_probabilities = predictions[0][top_indices]
        
        results = []
        for i, (category, prob) in enumerate(zip(top_categories, top_probabilities)):
            results.append({
                'rank': i + 1,
                'category': category,
                'probability': float(prob)
            })
        
        return results
    
    def save_model(self, path='models/hybrid_model.h5'):
        """Guardar modelo y componentes"""
        
        # Guardar modelo
        self.model.save(path)
        
        # Guardar tokenizer y label encoder
        import joblib
        joblib.dump(self.tokenizer, 'models/hybrid_tokenizer.pkl')
        joblib.dump(self.label_encoder, 'models/hybrid_label_encoder.pkl')
        
        print(f"💾 Modelo guardado como '{path}'")
        print(f"💾 Tokenizer guardado como 'models/hybrid_tokenizer.pkl'")
        print(f"💾 Label encoder guardado como 'models/hybrid_label_encoder.pkl'")

def main():
    """Función principal de entrenamiento"""
    
    print("🛍️ Iniciando entrenamiento de Red Híbrida para recomendación de productos")
    print("=" * 70)
    
    # Crear directorios necesarios
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    # 1. Inicializar recomendador
    print("\n🔧 Inicializando sistema de recomendación híbrido...")
    recommender = HybridProductRecommender()
    
    # 2. Cargar datos
    print("\n📁 Cargando datos de productos...")
    df = recommender.load_product_data()
    
    # 3. Preprocesar datos
    print("\n🔄 Preprocesando datos...")
    texts = recommender.preprocess_text_data(df)
    images = recommender.preprocess_image_data(df)
    labels, num_classes = recommender.preprocess_labels(df)
    
    print(f"   • Textos: {texts.shape}")
    print(f"   • Imágenes: {images.shape}")
    print(f"   • Etiquetas: {labels.shape}")
    print(f"   • Clases: {num_classes}")
    
    # 4. Construir modelo
    print("\n🏗️ Construyendo modelo híbrido...")
    model = recommender.build_hybrid_model(num_classes)
    model.summary()
    
    # 5. Entrenar modelo
    print("\n🎯 Entrenando modelo...")
    history, test_data = recommender.train_model(images, texts, labels, epochs=15)
    
    # 6. Evaluar modelo
    print("\n📈 Evaluando modelo...")
    results = recommender.evaluate_model(*test_data)
    
    # 7. Visualizar resultados
    print("\n📊 Generando visualizaciones...")
    recommender.plot_training_history()
    
    # 8. Guardar modelo
    print("\n💾 Guardando modelo...")
    recommender.save_model()
    
    # 9. Probar predicciones
    print("\n🧪 Probando predicciones...")
    test_products = [
        {
            'description': 'Smartphone Android con cámara de alta resolución y batería de larga duración',
            'category_expected': 'Electrónica'
        },
        {
            'description': 'Camiseta deportiva transpirable perfecta para ejercicio',
            'category_expected': 'Deportes'
        },
        {
            'description': 'Set de cuchillos de cocina profesional con bloque de madera',
            'category_expected': 'Hogar'
        }
    ]
    
    for i, product in enumerate(test_products, 1):
        predictions = recommender.predict_category(None, product['description'], top_k=3)
        print(f"\n   Producto {i}: {product['description'][:50]}...")
        print(f"   Categoría esperada: {product['category_expected']}")
        print(f"   Predicciones:")
        for pred in predictions:
            print(f"      {pred['rank']}. {pred['category']} ({pred['probability']:.4f})")
    
    # 10. Resumen final
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"📊 Métricas Finales:")
    print(f"   • Accuracy: {results['accuracy']:.4f}")
    print(f"   • Top-3 Accuracy: {results['top_k_categorical_accuracy']:.4f}")
    print(f"   • Loss: {results['loss']:.4f}")
    print(f"📁 Archivos generados:")
    print(f"   • models/hybrid_model.h5 - Modelo entrenado")
    print(f"   • models/hybrid_best_model.h5 - Mejor modelo durante entrenamiento")
    print(f"   • models/hybrid_tokenizer.pkl - Tokenizer guardado")
    print(f"   • models/hybrid_label_encoder.pkl - Label encoder guardado")
    print(f"   • notebooks/hybrid_training_history.png - Gráficos de entrenamiento")
    print("=" * 70)

if __name__ == "__main__":
    main()
