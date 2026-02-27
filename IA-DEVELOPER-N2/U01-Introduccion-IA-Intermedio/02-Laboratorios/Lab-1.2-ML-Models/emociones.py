import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import warnings
warnings.filterwarnings('ignore')

class ReconocedorEmociones:
    def __init__(self):
        self.modelo = None
        self.datos_entrenamiento = None
        self.datos_validacion = None
        self.datos_prueba = None
        self.etiquetas = ['feliz', 'triste', 'enojado', 'sorprendido', 'neutral']
        self.historial = None
        
    def generar_datos_sinteticos(self, n_muestras=1000, img_size=48):
        """Genera datos sintéticos de imágenes faciales para diferentes emociones"""
        print("😊 Generando datos sintéticos de emociones...")
        
        np.random.seed(42)
        
        # Generar imágenes aleatorias para cada emoción
        imagenes = []
        etiquetas_numericas = []
        
        for i, emocion in enumerate(self.etiquetas):
            # Generar imágenes para esta emoción
            for _ in range(n_muestras // len(self.etiquetas)):
                # Crear imagen base con ruido
                img = np.random.rand(img_size, img_size, 3) * 255
                
                # Añadir patrones característicos para cada emoción
                if emocion == 'feliz':
                    # Añadir curva de sonrisa
                    img[30:35, 15:33] = [255, 200, 0]  # Amarillo para sonrisa
                elif emocion == 'triste':
                    # Añadir curva hacia abajo
                    img[35:40, 15:33] = [0, 100, 200]  # Azul para tristeza
                elif emocion == 'enojado':
                    # Añadir cejas fruncidas
                    img[10:15, 10:20] = [200, 0, 0]  # Rojo para enojo
                    img[10:15, 28:38] = [200, 0, 0]
                elif emocion == 'sorprendido':
                    # Ojos grandes y boca ovalada
                    img[10:20, 10:20] = [255, 255, 255]  # Blanco para ojos
                    img[10:20, 28:38] = [255, 255, 255]
                    img[30:40, 20:28] = [255, 200, 0]  # Amarillo para boca
                # neutral no necesita patrón especial
                
                imagenes.append(img)
                etiquetas_numericas.append(i)
        
        # Convertir a arrays numpy
        imagenes = np.array(imagenes, dtype=np.float32) / 255.0
        etiquetas_numericas = np.array(etiquetas_numericas)
        
        # One-hot encoding para etiquetas
        etiquetas_onehot = tf.keras.utils.to_categorical(etiquetas_numericas, len(self.etiquetas))
        
        print(f"✅ Datos generados: {len(imagenes)} imágenes de {img_size}x{img_size}")
        print(f"📊 Distribución: {dict(zip(self.etiquetas, np.bincount(etiquetas_numericas)))}")
        
        return imagenes, etiquetas_onehot, etiquetas_numericas
    
    def preparar_datos(self, imagenes, etiquetas_onehot, test_size=0.2, val_size=0.2):
        """Divide y prepara los datos para entrenamiento"""
        print("📂 Preparando datos para entrenamiento...")
        
        # Dividir en entrenamiento+validación y prueba
        total_samples = len(imagenes)
        test_samples = int(total_samples * test_size)
        val_samples = int(total_samples * (1 - test_size) * val_size)
        
        # Mezclar datos
        indices = np.random.permutation(total_samples)
        imagenes = imagenes[indices]
        etiquetas_onehot = etiquetas_onehot[indices]
        
        # Dividir
        X_test = imagenes[:test_samples]
        y_test = etiquetas_onehot[:test_samples]
        
        X_temp = imagenes[test_samples:]
        y_temp = etiquetas_onehot[test_samples:]
        
        X_val = X_temp[:val_samples]
        y_val = y_temp[:val_samples]
        
        X_train = X_temp[val_samples:]
        y_train = y_temp[val_samples:]
        
        print(f"📊 División de datos:")
        print(f"   Entrenamiento: {len(X_train)} muestras")
        print(f"   Validación: {len(X_val)} muestras")
        print(f"   Prueba: {len(X_test)} muestras")
        
        # Data augmentation para entrenamiento
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        datagen.fit(X_train)
        
        self.datos_entrenamiento = (X_train, y_train, datagen)
        self.datos_validacion = (X_val, y_val)
        self.datos_prueba = (X_test, y_test)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def construir_modelo(self, input_shape=(48, 48, 3)):
        """Construye el modelo CNN para reconocimiento de emociones"""
        print("🏗️  Construyendo modelo CNN...")
        
        self.modelo = Sequential([
            # Primera capa convolucional
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Segunda capa convolucional
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Tercera capa convolucional
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Capas densas
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(self.etiquetas), activation='softmax')
        ])
        
        self.modelo.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modelo construido:")
        self.modelo.summary()
        
    def entrenar(self, epocas=50, batch_size=32):
        """Entrena el modelo CNN"""
        print("🎯 Entrenando modelo CNN...")
        
        if self.datos_entrenamiento is None:
            print("❌ Datos no preparados")
            return
        
        X_train, y_train, datagen = self.datos_entrenamiento
        X_val, y_val = self.datos_validacion
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        
        # Entrenamiento
        self.historial = self.modelo.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epocas,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("✅ Entrenamiento completado")
        
    def visualizar_entrenamiento(self):
        """Visualiza el progreso del entrenamiento"""
        if self.historial is None:
            print("❌ No hay historial de entrenamiento")
            return
            
        plt.figure(figsize=(12, 4))
        
        # Gráfico de accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.historial.history['accuracy'], label='Entrenamiento')
        plt.plot(self.historial.history['val_accuracy'], label='Validación')
        plt.title('Accuracy durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(self.historial.history['loss'], label='Entrenamiento')
        plt.plot(self.historial.history['val_loss'], label='Validación')
        plt.title('Pérdida durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def evaluar_modelo(self):
        """Evalúa el modelo con datos de prueba"""
        if self.modelo is None or self.datos_prueba is None:
            print("❌ Modelo o datos no disponibles")
            return
        
        X_test, y_test = self.datos_prueba
        
        # Evaluar
        perdida, accuracy = self.modelo.evaluate(X_test, y_test, verbose=0)
        
        print(f"📊 Evaluación del modelo:")
        print(f"   Pérdida: {perdida:.4f}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Predicciones
        y_pred = self.modelo.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Accuracy por clase
        print("\n📊 Accuracy por clase:")
        for i, etiqueta in enumerate(self.etiquetas):
            mask = y_true_classes == i
            if np.sum(mask) > 0:
                acc = np.mean(y_pred_classes[mask] == y_true_classes[mask])
                print(f"   {etiqueta}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Mostrar algunas predicciones
        print("\n🧪 Ejemplos de predicciones:")
        indices_muestra = np.random.choice(len(X_test), 5, replace=False)
        
        for idx in indices_muestra:
            img = X_test[idx]
            true_label = self.etiquetas[y_true_classes[idx]]
            pred_label = self.etiquetas[y_pred_classes[idx]]
            confidence = np.max(y_pred[idx])
            
            print(f"   Real: {true_label:10} | Predicho: {pred_label:10} | Confianza: {confidence:.3f}")
    
    def detectar_caras(self, imagen):
        """Detecta caras en una imagen usando OpenCV"""
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Cargar clasificador de caras (usando uno básico)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detectar caras
        caras = face_cascade.detectMultiScale(gris, 1.1, 4)
        
        return caras
    
    def predecir_emocion(self, imagen):
        """Predice la emoción en una imagen"""
        if self.modelo is None:
            print("❌ Modelo no entrenado")
            return None, None
        
        # Preprocesar imagen
        if len(imagen.shape) == 3:
            img = cv2.resize(imagen, (48, 48))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
        else:
            img = np.expand_dims(imagen, axis=0)
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)
        
        # Predecir
        prediccion = self.modelo.predict(img)
        clase_predicha = np.argmax(prediccion)
        confianza = np.max(prediccion)
        
        emocion = self.etiquetas[clase_predicha]
        
        return emocion, confianza

def demo_reconocimiento_emociones():
    """Demostración completa del reconocimiento de emociones"""
    print("=" * 60)
    print("😊 SISTEMA DE RECONOCIMIENTO DE EMOCIONES")
    print("=" * 60)
    
    # Crear instancia
    reconocedor = ReconocedorEmociones()
    
    # Generar datos
    imagenes, etiquetas_onehot, etiquetas_numericas = reconocedor.generar_datos_sinteticos(n_muestras=1000)
    
    # Preparar datos
    X_train, y_train, X_val, y_val, X_test, y_test = reconocedor.preparar_datos(imagenes, etiquetas_onehot)
    
    # Construir modelo
    reconocedor.construir_modelo(input_shape=(48, 48, 3))
    
    # Entrenar
    reconocedor.entrenar(epocas=30, batch_size=32)
    
    # Visualizar entrenamiento
    reconocedor.visualizar_entrenamiento()
    
    # Evaluar modelo
    reconocedor.evaluar_modelo()
    
    print("\n" + "=" * 60)
    print("🎉 DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)

if __name__ == "__main__":
    demo_reconocimiento_emociones()
