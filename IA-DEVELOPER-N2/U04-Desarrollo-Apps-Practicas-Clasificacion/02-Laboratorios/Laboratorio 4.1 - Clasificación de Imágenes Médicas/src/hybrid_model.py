"""
Modelo Híbrido: EfficientNetV3 + Vision Transformers
Laboratorio 4.1 - Clasificación de Imágenes Médicas con EfficientNetV3 y ViT
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.applications import EfficientNetV3Small
import numpy as np

class AttentionBlock(layers.Layer):
    """
    Bloque de atención simplificado inspirado en Vision Transformers
    """
    
    def __init__(self, units, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.query_dense = layers.Dense(units)
        self.key_dense = layers.Dense(units)
        self.value_dense = layers.Dense(units)
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=units // num_heads
        )
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(units * 4, activation='gelu'),
            layers.Dense(units)
        ])
    
    def call(self, inputs, training=None):
        # Multi-head self-attention
        attn_output = self.multi_head_attention(inputs, inputs)
        out1 = self.norm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
        return config


class HybridMedicalClassifier:
    """
    Clasificador híbrido que combina EfficientNetV3 con bloques de atención tipo ViT
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self, dropout_rate=0.3, fine_tune_at=100):
        """
        Construye el modelo híbrido EfficientNetV3 + ViT
        """
        # Base EfficientNetV3 pre-entrenado
        efficientnet_base = EfficientNetV3Small(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling=None
        )
        
        # Congelar capas iniciales
        for layer in efficientnet_base.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Entrada del modelo
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        # Pasar por EfficientNet
        x = efficientnet_base(inputs)
        
        # Obtener dimensiones espaciales
        _, h, w, c = x.shape
        
        # Reshape para atención (similar a ViT patches)
        x = layers.Reshape((h * w, c))(x)
        
        # Añadir bloques de atención
        x = AttentionBlock(units=256, num_heads=8)(x)
        x = AttentionBlock(units=256, num_heads=8)(x)
        
        # Global average pooling sobre la dimensión de secuencia
        x = layers.GlobalAveragePooling1D()(x)
        
        # Capas densas finales
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Capa de salida
        if self.num_classes == 1:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Crear modelo
        self.model = models.Model(inputs=inputs, outputs=outputs, name='HybridMedicalClassifier')
        
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """
        Compila el modelo con optimizador y métricas
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido construido. Llama a build_model() primero.")
        
        # Optimizador con learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Métricas para clasificación médica
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
        
        if self.num_classes == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return self.model
    
    def get_callbacks(self, model_path='best_model.h5', patience=5):
        """
        Configura callbacks para entrenamiento
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_auc',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def summary(self):
        """
        Muestra el resumen del modelo
        """
        if self.model is not None:
            self.model.summary()
        else:
            print("El modelo no ha sido construido aún.")
    
    def predict_with_confidence(self, image, threshold=0.5):
        """
        Realiza predicción con confianza
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido construido o cargado.")
        
        prediction = self.model.predict(image)
        confidence = np.max(prediction)
        
        if self.num_classes == 1:
            class_pred = 'positive' if prediction[0][0] > threshold else 'negative'
            return {
                'prediction': class_pred,
                'confidence': float(confidence),
                'raw_score': float(prediction[0][0])
            }
        else:
            class_idx = np.argmax(prediction)
            return {
                'prediction': int(class_idx),
                'confidence': float(confidence),
                'probabilities': prediction[0].tolist()
            }


if __name__ == "__main__":
    # Ejemplo de uso
    classifier = HybridMedicalClassifier(input_shape=(224, 224, 3), num_classes=1)
    
    # Construir modelo
    model = classifier.build_model()
    model = classifier.compile_model()
    
    # Mostrar resumen
    classifier.summary()
    
    # Probar predicción
    dummy_input = np.random.random((1, 224, 224, 3))
    try:
        result = classifier.predict_with_confidence(dummy_input)
        print("Predicción de prueba:", result)
    except Exception as e:
        print(f"Error en predicción: {e}")
