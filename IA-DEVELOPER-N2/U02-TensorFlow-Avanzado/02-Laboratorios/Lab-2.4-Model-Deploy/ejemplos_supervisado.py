import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification, load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class EjemplosSupervisado:
    def __init__(self):
        self.ejemplos = {
            'Dense': self.ejemplo_dense,
            'CNN': self.ejemplo_cnn,
            'LSTM': self.ejemplo_lstm
        }
    
    def ejemplo_dense(self):
        """Ejemplo 1: Red Densa para Clasificación"""
        print("=" * 60)
        print("🔗 EJEMPLO 1: RED DENSA - CLASIFICACIÓN MULTICLASE")
        print("=" * 60)
        
        # Generar datos sintéticos
        print("📊 Generando datos sintéticos...")
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=3,
            n_informative=15,
            random_state=42
        )
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Construir red densa
        print("🏗️  Construyendo red densa...")
        modelo = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        # Compilar
        modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modelo construido:")
        modelo.summary()
        
        # Entrenar
        print("\n🎯 Entrenando modelo...")
        historia = modelo.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluar
        print("\n📊 Evaluando modelo...")
        perdida, accuracy = modelo.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Predicciones
        y_pred = np.argmax(modelo.predict(X_test_scaled), axis=1)
        print("\n📋 Reporte de clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Visualizar entrenamiento
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(historia.history['loss'], label='Entrenamiento')
        plt.plot(historia.history['val_loss'], label='Validación')
        plt.title('Pérdida - Red Densa')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(historia.history['accuracy'], label='Entrenamiento')
        plt.plot(historia.history['val_accuracy'], label='Validación')
        plt.title('Accuracy - Red Densa')
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return modelo, accuracy
    
    def ejemplo_cnn(self):
        """Ejemplo 2: CNN para Clasificación de Imágenes"""
        print("\n" + "=" * 60)
        print("🖼️  EJEMPLO 2: CNN - CLASIFICACIÓN DE IMÁGENES")
        print("=" * 60)
        
        # Cargar MNIST
        print("📊 Cargando dataset MNIST...")
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        # Preprocesar
        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # One-hot encoding
        y_train_cat = keras.utils.to_categorical(y_train, 10)
        y_test_cat = keras.utils.to_categorical(y_test, 10)
        
        print(f"Datos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
        
        # Construir CNN
        print("🏗️  Construyendo CNN...")
        modelo = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compilar
        modelo.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modelo construido:")
        modelo.summary()
        
        # Entrenar
        print("\n🎯 Entrenando CNN...")
        historia = modelo.fit(
            X_train, y_train_cat,
            epochs=10,
            batch_size=128,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluar
        print("\n📊 Evaluando CNN...")
        perdida, accuracy = modelo.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Visualizar algunas predicciones
        print("\n🔮 Ejemplos de predicciones:")
        indices = np.random.choice(len(X_test), 5, replace=False)
        
        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(indices):
            plt.subplot(1, 5, i+1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            
            # Predicción
            prediccion = np.argmax(modelo.predict(X_test[idx:idx+1], verbose=0))
            real = y_test[idx]
            
            color = 'green' if prediccion == real else 'red'
            plt.title(f'Pred: {prediccion}, Real: {real}', color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return modelo, accuracy
    
    def ejemplo_lstm(self):
        """Ejemplo 3: LSTM para Predicción de Series Temporales"""
        print("\n" + "=" * 60)
        print("🧠 EJEMPLO 3: LSTM - PREDICCIÓN DE SERIES TEMPORALES")
        print("=" * 60)
        
        # Generar datos sintéticos (seno + ruido)
        print("📊 Generando serie temporal sintética...")
        np.random.seed(42)
        tiempo = np.linspace(0, 100, 1000)
        senal = np.sin(tiempo * 0.1) + np.random.normal(0, 0.1, 1000)
        
        # Crear secuencias
        def crear_secuencias(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        seq_length = 50
        X, y = crear_secuencias(senal, seq_length)
        
        # Dividir datos
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Reshape para LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        print(f"Secuencias: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
        
        # Construir LSTM
        print("🏗️  Construyendo LSTM...")
        modelo = keras.Sequential([
            keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])
        
        # Compilar
        modelo.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Modelo construido:")
        modelo.summary()
        
        # Entrenar
        print("\n🎯 Entrenando LSTM...")
        historia = modelo.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluar
        print("\n📊 Evaluando LSTM...")
        perdida, mae = modelo.evaluate(X_test, y_test, verbose=0)
        print(f"MAE: {mae:.4f}")
        
        # Predicciones
        y_pred = modelo.predict(X_test)
        
        # Visualizar resultados
        plt.figure(figsize=(15, 8))
        
        # Serie completa
        plt.subplot(2, 1, 1)
        plt.plot(tiempo, senal, 'b-', alpha=0.7, label='Señal original')
        plt.title('Serie Temporal Completa')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        
        # Predicciones vs reales
        plt.subplot(2, 1, 2)
        tiempo_test = tiempo[split+seq_length:split+seq_length+len(y_test)]
        plt.plot(tiempo_test, y_test, 'b-', label='Real', alpha=0.7)
        plt.plot(tiempo_test, y_pred, 'r-', label='Predicción', alpha=0.7)
        plt.title('Predicciones LSTM vs Real')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return modelo, mae
    
    def ejecutar_todos(self):
        """Ejecuta todos los ejemplos supervisados"""
        print("🎯 EJECUTANDO EJEMPLOS DE APRENDIZAJE SUPERVISADO")
        print("=" * 80)
        
        resultados = {}
        
        for nombre, funcion in self.ejemplos.items():
            print(f"\n🔄 Ejecutando ejemplo: {nombre}")
            modelo, metrica = funcion()
            resultados[nombre] = {'modelo': modelo, 'metrica': metrica}
            print(f"✅ {nombre} completado - Métrica: {metrica:.4f}")
        
        # Resumen
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE RESULTADOS - APRENDIZAJE SUPERVISADO")
        print("=" * 80)
        
        for nombre, resultado in resultados.items():
            print(f"{nombre:10}: {resultado['metrica']:.4f}")
        
        return resultados

def demo_ejemplos_supervisado():
    """Demostración completa de ejemplos supervisados"""
    print("=" * 80)
    print("🎯 LABORATORIO 4 - EJEMPLOS SUPERVISADOS")
    print("=" * 80)
    
    # Crear y ejecutar ejemplos
    ejemplos = EjemplosSupervisado()
    resultados = ejemplos.ejecutar_todos()
    
    print("\n" + "=" * 80)
    print("🎉 EJEMPLOS SUPERVISADOS COMPLETADOS")
    print("=" * 80)

if __name__ == "__main__":
    demo_ejemplos_supervisado()
