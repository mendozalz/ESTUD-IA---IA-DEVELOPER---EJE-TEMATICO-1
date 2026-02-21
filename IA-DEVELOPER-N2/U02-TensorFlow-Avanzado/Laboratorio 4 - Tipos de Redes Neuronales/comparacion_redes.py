import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
import warnings
warnings.filterwarnings('ignore')

class ComparadorRedes:
    def __init__(self):
        self.metricas = {}
        self.tiempos_entrenamiento = {}
        self.tiempos_inferencia = {}
        
    def crear_dataset_comun(self, n_samples=1000):
        """Crea un dataset común para comparación"""
        print("📊 Creando dataset común para comparación...")
        
        # Dataset para clasificación
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_classes=3,
            n_informative=15,
            random_state=42
        )
        
        # Dataset para secuencias (para RNN/LSTM)
        def crear_secuencias(n_seq=100, seq_len=50):
            secuencias = []
            for _ in range(n_seq):
                # Generar secuencia con patrón
                t = np.linspace(0, 10, seq_len)
                senal = np.sin(t) + 0.1 * np.random.normal(0, 1, seq_len)
                secuencias.append(senal)
            return np.array(secuencias)
        
        X_seq = crear_secuencias(n_samples // 10, 50)
        y_seq = np.random.randint(0, 3, n_samples // 10)
        
        # Dataset para imágenes (para CNN)
        X_img = np.random.rand(n_samples // 5, 28, 28, 1)
        y_img = np.random.randint(0, 3, n_samples // 5)
        
        print(f"✅ Datasets creados:")
        print(f"   Tabular: {X.shape}")
        print(f"   Secuencias: {X_seq.shape}")
        print(f"   Imágenes: {X_img.shape}")
        
        return (X, y), (X_seq, y_seq), (X_img, y_img)
    
    def evaluar_red_densa(self, X, y):
        """Evalúa Red Neuronal Densa"""
        print("\n🔗 Evaluando Red Neuronal Densa...")
        
        # Preparar datos
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Construir modelo
        modelo = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        historia = modelo.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)
        tiempo_entrenamiento = time.time() - start_time
        
        # Medir tiempo de inferencia
        start_time = time.time()
        y_pred = np.argmax(modelo.predict(X_test_scaled, verbose=0), axis=1)
        tiempo_inferencia = time.time() - start_time
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        
        # Contar parámetros
        n_parametros = modelo.count_params()
        
        # Guardar resultados
        self.metricas['Dense'] = {
            'accuracy': accuracy,
            'parametros': n_parametros,
            'dataset_size': len(X),
            'epochs': 20
        }
        self.tiempos_entrenamiento['Dense'] = tiempo_entrenamiento
        self.tiempos_inferencia['Dense'] = tiempo_inferencia
        
        print(f"✅ Dense - Accuracy: {accuracy:.4f}, Params: {n_parametros:,}")
        print(f"   Tiempo entrenamiento: {tiempo_entrenamiento:.2f}s")
        print(f"   Tiempo inferencia: {tiempo_inferencia:.4f}s")
        
        return accuracy, tiempo_entrenamiento, n_parametros
    
    def evaluar_cnn(self, X_img, y_img):
        """Evalúa Red Convolucional"""
        print("\n🖼️  Evaluando Red Convolucional...")
        
        # Preparar datos
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_img, y_img, test_size=0.2, random_state=42)
        
        # Construir CNN
        modelo = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        historia = modelo.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        tiempo_entrenamiento = time.time() - start_time
        
        # Medir tiempo de inferencia
        start_time = time.time()
        y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
        tiempo_inferencia = time.time() - start_time
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        n_parametros = modelo.count_params()
        
        # Guardar resultados
        self.metricas['CNN'] = {
            'accuracy': accuracy,
            'parametros': n_parametros,
            'dataset_size': len(X_img),
            'epochs': 10
        }
        self.tiempos_entrenamiento['CNN'] = tiempo_entrenamiento
        self.tiempos_inferencia['CNN'] = tiempo_inferencia
        
        print(f"✅ CNN - Accuracy: {accuracy:.4f}, Params: {n_parametros:,}")
        print(f"   Tiempo entrenamiento: {tiempo_entrenamiento:.2f}s")
        print(f"   Tiempo inferencia: {tiempo_inferencia:.4f}s")
        
        return accuracy, tiempo_entrenamiento, n_parametros
    
    def evaluar_lstm(self, X_seq, y_seq):
        """Evalúa LSTM"""
        print("\n🧠 Evaluando LSTM...")
        
        # Preparar datos
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
        
        # Construir LSTM
        modelo = keras.Sequential([
            keras.layers.LSTM(32, activation='relu', input_shape=(X_seq.shape[1], 1), return_sequences=True),
            keras.layers.LSTM(16, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        historia = modelo.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)
        tiempo_entrenamiento = time.time() - start_time
        
        # Medir tiempo de inferencia
        start_time = time.time()
        y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
        tiempo_inferencia = time.time() - start_time
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        n_parametros = modelo.count_params()
        
        # Guardar resultados
        self.metricas['LSTM'] = {
            'accuracy': accuracy,
            'parametros': n_parametros,
            'dataset_size': len(X_seq),
            'epochs': 15
        }
        self.tiempos_entrenamiento['LSTM'] = tiempo_entrenamiento
        self.tiempos_inferencia['LSTM'] = tiempo_inferencia
        
        print(f"✅ LSTM - Accuracy: {accuracy:.4f}, Params: {n_parametros:,}")
        print(f"   Tiempo entrenamiento: {tiempo_entrenamiento:.2f}s")
        print(f"   Tiempo inferencia: {tiempo_inferencia:.4f}s")
        
        return accuracy, tiempo_entrenamiento, n_parametros
    
    def evaluar_autoencoder(self, X):
        """Evalúa Autoencoder"""
        print("\n🗜️  Evaluando Autoencoder...")
        
        # Preparar datos
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Construir Autoencoder
        input_dim = X.shape[1]
        encoding_dim = 10
        
        # Encoder
        input_layer = keras.layers.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(32, activation='relu')(input_layer)
        encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = keras.layers.Dense(32, activation='relu')(encoded)
        decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        historia = autoencoder.fit(X_train_scaled, X_train_scaled, epochs=30, batch_size=32, verbose=0)
        tiempo_entrenamiento = time.time() - start_time
        
        # Medir tiempo de inferencia
        start_time = time.time()
        X_reconstruido = autoencoder.predict(X_test_scaled, verbose=0)
        tiempo_inferencia = time.time() - start_time
        
        # Calcular métricas
        mse = np.mean(np.square(X_test_scaled - X_reconstruido))
        n_parametros = autoencoder.count_params()
        
        # Guardar resultados
        self.metricas['Autoencoder'] = {
            'mse': mse,
            'parametros': n_parametros,
            'dataset_size': len(X),
            'epochs': 30
        }
        self.tiempos_entrenamiento['Autoencoder'] = tiempo_entrenamiento
        self.tiempos_inferencia['Autoencoder'] = tiempo_inferencia
        
        print(f"✅ Autoencoder - MSE: {mse:.6f}, Params: {n_parametros:,}")
        print(f"   Tiempo entrenamiento: {tiempo_entrenamiento:.2f}s")
        print(f"   Tiempo inferencia: {tiempo_inferencia:.4f}s")
        
        return mse, tiempo_entrenamiento, n_parametros
    
    def crear_tabla_comparativa(self):
        """Crea tabla comparativa completa"""
        print("\n📊 Creando tabla comparativa...")
        
        # Preparar datos para tabla
        tabla_datos = []
        
        for red, metricas in self.metricas.items():
            fila = {
                'Red Neuronal': red,
                'Parámetros': metricas.get('parametros', 0),
                'Tiempo Entrenamiento (s)': self.tiempos_entrenamiento.get(red, 0),
                'Tiempo Inferencia (s)': self.tiempos_inferencia.get(red, 0),
                'Dataset Size': metricas.get('dataset_size', 0),
                'Épocas': metricas.get('epochs', 0)
            }
            
            # Añadir métricas específicas
            if 'accuracy' in metricas:
                fila['Accuracy'] = metricas['accuracy']
            if 'mse' in metricas:
                fila['MSE'] = metricas['mse']
            
            tabla_datos.append(fila)
        
        df = pd.DataFrame(tabla_datos)
        
        # Calcular eficiencia
        df['Parámetros/1000'] = df['Parámetros'] / 1000
        df['Eficiencia (Acc/Params)'] = df.get('Accuracy', 0) / (df['Parámetros/1000'] + 1)
        
        return df
    
    def visualizar_comparacion(self):
        """Visualiza comparación de redes"""
        print("\n📈 Generando visualizaciones...")
        
        if not self.metricas:
            print("❌ No hay métricas para visualizar")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comparación de Tipos de Redes Neuronales', fontsize=16)
        
        redes = list(self.metricas.keys())
        
        # 1. Accuracy (si aplica)
        accuracies = []
        for red in redes:
            if 'accuracy' in self.metricas[red]:
                accuracies.append(self.metricas[red]['accuracy'])
            else:
                accuracies.append(0)
        
        axes[0, 0].bar(redes, accuracies, color='skyblue')
        axes[0, 0].set_title('Accuracy por Tipo de Red')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Número de parámetros
        parametros = [self.metricas[red].get('parametros', 0) for red in redes]
        axes[0, 1].bar(redes, parametros, color='lightcoral')
        axes[0, 1].set_title('Número de Parámetros')
        axes[0, 1].set_ylabel('Parámetros')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Tiempo de entrenamiento
        tiempos_ent = [self.tiempos_entrenamiento.get(red, 0) for red in redes]
        axes[0, 2].bar(redes, tiempos_ent, color='lightgreen')
        axes[0, 2].set_title('Tiempo de Entrenamiento')
        axes[0, 2].set_ylabel('Tiempo (s)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Tiempo de inferencia
        tiempos_inf = [self.tiempos_inferencia.get(red, 0) for red in redes]
        axes[1, 0].bar(redes, tiempos_inf, color='gold')
        axes[1, 0].set_title('Tiempo de Inferencia')
        axes[1, 0].set_ylabel('Tiempo (s)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Eficiencia (Accuracy/Parámetros)
        eficiencia = []
        for red in redes:
            acc = self.metricas[red].get('accuracy', 0)
            params = self.metricas[red].get('parametros', 1)
            eficiencia.append(acc / (params / 1000 + 1))
        
        axes[1, 1].bar(redes, eficiencia, color='mediumpurple')
        axes[1, 1].set_title('Eficiencia (Accuracy/Parámetros)')
        axes[1, 1].set_ylabel('Eficiencia')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. MSE (para autoencoder)
        mses = []
        for red in redes:
            if 'mse' in self.metricas[red]:
                mses.append(self.metricas[red]['mse'])
            else:
                mses.append(0)
        
        axes[1, 2].bar(redes, mses, color='orange')
        axes[1, 2].set_title('MSE (Reconstrucción)')
        axes[1, 2].set_ylabel('MSE')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generar_recomendaciones(self):
        """Genera recomendaciones basadas en resultados"""
        print("\n💡 Generando recomendaciones...")
        
        if not self.metricas:
            print("❌ No hay métricas para generar recomendaciones")
            return
        
        recomendaciones = []
        
        # Mejor accuracy
        mejor_acc = max([(red, metrics['accuracy']) for red, metrics in self.metricas.items() if 'accuracy' in metrics], 
                       key=lambda x: x[1], default=(None, 0))
        if mejor_acc[0]:
            recomendaciones.append(f"🏆 Mejor Accuracy: {mejor_acc[0]} ({mejor_acc[1]:.4f})")
        
        # Más rápida en inferencia
        mas_rapida = min(self.tiempos_inferencia.items(), key=lambda x: x[1])
        recomendaciones.append(f"⚡ Más Rápida en Inferencia: {mas_rapida[0]} ({mas_rapida[1]:.4f}s)")
        
        # Menos parámetros
        menos_params = min([(red, metrics['parametros']) for red, metrics in self.metricas.items()], 
                          key=lambda x: x[1])
        recomendaciones.append(f"🪉 Menos Parámetros: {menos_params[0]} ({menos_params[1]:,})")
        
        # Mejor MSE
        mejor_mse = min([(red, metrics['mse']) for red, metrics in self.metricas.items() if 'mse' in metrics], 
                       key=lambda x: x[1], default=(None, float('inf')))
        if mejor_mse[0]:
            recomendaciones.append(f"🎯 Mejor Reconstrucción: {mejor_mse[0]} (MSE: {mejor_mse[1]:.6f})")
        
        # Recomendaciones por caso de uso
        print("\n📋 Recomendaciones por Caso de Uso:")
        print("   📊 Datos Tabulares: Red Densa (mejor para características estructuradas)")
        print("   🖼️  Imágenes: CNN (especializada en datos espaciales)")
        print("   🔄 Secuencias: LSTM/RNN (memoria temporal)")
        print("   🗜️  Reducción Dimensional: Autoencoder (aprendizaje no supervisado)")
        print("   ⚡ Aplicaciones en Tiempo Real: Red con menos parámetros y rápida inferencia")
        
        return recomendaciones
    
    def ejecutar_comparacion_completa(self):
        """Ejecuta comparación completa de todas las redes"""
        print("🔄 EJECUTANDO COMPARACIÓN COMPLETA DE REDES NEURONALES")
        print("=" * 80)
        
        # Crear datasets
        (X_tab, y_tab), (X_seq, y_seq), (X_img, y_img) = self.crear_dataset_comun(1000)
        
        # Evaluar cada tipo de red
        print("\n🔍 Evaluando diferentes tipos de redes...")
        
        # Red Densa
        self.evaluar_red_densa(X_tab, y_tab)
        
        # CNN
        self.evaluar_cnn(X_img, y_img)
        
        # LSTM
        self.evaluar_lstm(X_seq, y_seq)
        
        # Autoencoder
        self.evaluar_autoencoder(X_tab)
        
        # Crear tabla comparativa
        tabla = self.crear_tabla_comparativa()
        
        # Visualizar resultados
        self.visualizar_comparacion()
        
        # Generar recomendaciones
        recomendaciones = self.generar_recomendaciones()
        
        print("\n" + "=" * 80)
        print("📊 TABLA COMPARATIVA COMPLETA")
        print("=" * 80)
        print(tabla.to_string(index=False))
        
        print("\n💡 RECOMENDACIONES:")
        for rec in recomendaciones:
            print(f"   {rec}")
        
        return tabla, recomendaciones

def demo_comparacion_redes():
    """Demostración completa de comparación de redes"""
    print("=" * 80)
    print("🔬 LABORATORIO 4 - COMPARACIÓN DE REDES NEURONALES")
    print("=" * 80)
    
    # Crear comparador y ejecutar análisis
    comparador = ComparadorRedes()
    tabla, recomendaciones = comparador.ejecutar_comparacion_completa()
    
    print("\n" + "=" * 80)
    print("🎉 COMPARACIÓN COMPLETADA")
    print("=" * 80)
    
    return comparador, tabla, recomendaciones

if __name__ == "__main__":
    demo_comparacion_redes()
