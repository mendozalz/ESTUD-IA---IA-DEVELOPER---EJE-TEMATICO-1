import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ConversorTemperatura:
    def __init__(self):
        self.modelo = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.historial = None
        
    def preparar_datos(self, n_muestras=1000):
        """Genera datos sintéticos para conversión de temperaturas"""
        print("🌡️  Generando datos de conversión de temperaturas...")
        
        np.random.seed(42)
        
        # Generar temperaturas en diferentes rangos
        celsius = np.random.uniform(-50, 150, n_muestras)
        fahrenheit = celsius * 9/5 + 32
        kelvin = celsius + 273.15
        
        # Crear DataFrame
        datos = pd.DataFrame({
            'celsius': celsius,
            'fahrenheit': fahrenheit,
            'kelvin': kelvin
        })
        
        # Seleccionar conversión (ej: Celsius a Fahrenheit)
        X = datos[['celsius']].values
        y = datos[['fahrenheit']].values
        
        # Normalizar datos
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        print(f"✅ Datos preparados: {len(X)} muestras")
        print(f"📊 Rango Celsius: {X.min():.1f}° a {X.max():.1f}°")
        print(f"📊 Rango Fahrenheit: {y.min():.1f}° a {y.max():.1f}°")
        
        return X_scaled, y_scaled, X, y
    
    def construir_modelo(self):
        """Construye el modelo de red neuronal"""
        print("🏗️  Construyendo modelo de red neuronal...")
        
        self.modelo = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        self.modelo.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Modelo construido:")
        self.modelo.summary()
        
    def entrenar(self, X_scaled, y_scaled, epocas=100, batch_size=32):
        """Entrena el modelo"""
        print("🎯 Entrenando modelo...")
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.historial = self.modelo.fit(
            X_scaled, y_scaled,
            epochs=epocas,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        print("✅ Entrenamiento completado")
        
    def visualizar_entrenamiento(self):
        """Visualiza el progreso del entrenamiento"""
        if self.historial is None:
            print("❌ No hay historial de entrenamiento")
            return
            
        plt.figure(figsize=(12, 4))
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(self.historial.history['loss'], label='Entrenamiento')
        plt.plot(self.historial.history['val_loss'], label='Validación')
        plt.title('Pérdida durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (MSE)')
        plt.legend()
        plt.grid(True)
        
        # Gráfico de MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.historial.history['mae'], label='Entrenamiento')
        plt.plot(self.historial.history['val_mae'], label='Validación')
        plt.title('Error absoluto medio durante entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def convertir_temperatura(self, celsius):
        """Convierte temperatura usando el modelo entrenado"""
        if self.modelo is None:
            print("❌ Modelo no entrenado")
            return None
            
        # Normalizar entrada
        celsius_scaled = self.scaler_X.transform([[celsius]])
        
        # Predecir
        fahrenheit_scaled = self.modelo.predict(celsius_scaled)
        
        # Desnormalizar salida
        fahrenheit = self.scaler_y.inverse_transform(fahrenheit_scaled)
        
        return fahrenheit[0][0]
    
    def evaluar_modelo(self, X_test, y_test):
        """Evalúa el modelo con datos de prueba"""
        if self.modelo is None:
            print("❌ Modelo no entrenado")
            return
            
        # Normalizar datos de prueba
        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Evaluar
        perdida, mae = self.modelo.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        
        # Desnormalizar MAE
        mae_desnormalizado = mae * (self.scaler_y.data_max_[0] - self.scaler_y.data_min_[0])
        
        print(f"📊 Evaluación del modelo:")
        print(f"   Pérdida (MSE): {perdida:.6f}")
        print(f"   Error absoluto medio: {mae_desnormalizado:.4f}°F")
        
        # Probar algunas conversiones
        print("\n🧪 Pruebas de conversión:")
        temperaturas_prueba = [0, 25, 100, -40]
        
        for temp_c in temperaturas_prueba:
            prediccion = self.convertir_temperatura(temp_c)
            formula_real = temp_c * 9/5 + 32
            error = abs(prediccion - formula_real)
            error_relativo = (error / abs(formula_real)) * 100
            
            print(f"   {temp_c:3.0f}°C → {prediccion:6.2f}°F (real: {formula_real:6.2f}°F, error: {error:.2f}°F, {error_relativo:.1f}%)")

def demo_conversor_temperatura():
    """Demostración completa del conversor de temperaturas"""
    print("=" * 60)
    print("🌡️  CONVERSOR DE TEMPERATURAS CON IA")
    print("=" * 60)
    
    # Crear instancia
    conversor = ConversorTemperatura()
    
    # Preparar datos
    X_scaled, y_scaled, X, y = conversor.preparar_datos(n_muestras=1000)
    
    # Construir modelo
    conversor.construir_modelo()
    
    # Entrenar
    conversor.entrenar(X_scaled, y_scaled, epocas=100, batch_size=32)
    
    # Visualizar entrenamiento
    conversor.visualizar_entrenamiento()
    
    # Evaluar modelo
    conversor.evaluar_modelo(X, y)
    
    print("\n" + "=" * 60)
    print("🎉 DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)

if __name__ == "__main__":
    demo_conversor_temperatura()
