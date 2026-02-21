import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class EjemplosNoSupervisado:
    def __init__(self):
        self.ejemplos = {
            'Autoencoder': self.ejemplo_autoencoder,
            'GAN': self.ejemplo_gan,
            'Transformer': self.ejemplo_transformer_clustering
        }
    
    def ejemplo_autoencoder(self):
        """Ejemplo 1: Autoencoder para Reducción Dimensional"""
        print("=" * 60)
        print("🗜️  EJEMPLO 1: AUTOENCODER - REDUCCIÓN DIMENSIONAL")
        print("=" * 60)
        
        # Generar datos de alta dimensionalidad
        print("📊 Generando datos de alta dimensionalidad...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 100
        
        # Crear datos con estructura latente
        latent_dim = 10
        latent_data = np.random.randn(n_samples, latent_dim)
        
        # Proyectar a alta dimensionalidad
        projection_matrix = np.random.randn(latent_dim, n_features)
        X_high_dim = latent_data @ projection_matrix + np.random.normal(0, 0.1, (n_samples, n_features))
        
        # Escalar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_high_dim)
        
        print(f"Datos originales: {X_scaled.shape}")
        
        # Construir autoencoder
        print("🏗️  Construyendo Autoencoder...")
        
        # Encoder
        input_layer = keras.layers.Input(shape=(n_features,))
        encoded = keras.layers.Dense(64, activation='relu')(input_layer)
        encoded = keras.layers.Dense(32, activation='relu')(encoded)
        bottleneck = keras.layers.Dense(latent_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = keras.layers.Dense(32, activation='relu')(bottleneck)
        decoded = keras.layers.Dense(64, activation='relu')(decoded)
        output_layer = keras.layers.Dense(n_features, activation='linear')(decoded)
        
        # Modelo completo
        autoencoder = keras.Model(input_layer, output_layer)
        
        # Modelo encoder (para reducción)
        encoder = keras.Model(input_layer, bottleneck)
        
        # Compilar
        autoencoder.compile(optimizer='adam', loss='mse')
        
        print("✅ Autoencoder construido:")
        autoencoder.summary()
        
        # Entrenar
        print("\n🎯 Entrenando Autoencoder...")
        historia = autoencoder.fit(
            X_scaled, X_scaled,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Obtener representación latente
        X_latent = encoder.predict(X_scaled)
        
        print(f"Representación latente: {X_latent.shape}")
        
        # Evaluar reconstrucción
        X_reconstruido = autoencoder.predict(X_scaled)
        mse_reconstruccion = np.mean(np.square(X_scaled - X_reconstruido))
        print(f"MSE reconstrucción: {mse_reconstruccion:.6f}")
        
        # Visualizar
        plt.figure(figsize=(15, 10))
        
        # Pérdida durante entrenamiento
        plt.subplot(2, 3, 1)
        plt.plot(historia.history['loss'], label='Entrenamiento')
        plt.plot(historia.history['val_loss'], label='Validación')
        plt.title('Pérdida - Autoencoder')
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.legend()
        
        # Comparación original vs reconstruido (primeras 2 dimensiones)
        plt.subplot(2, 3, 2)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6, label='Original')
        plt.scatter(X_reconstruido[:, 0], X_reconstruido[:, 1], alpha=0.6, label='Reconstruido')
        plt.title('Original vs Reconstruido (Dims 1-2)')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.legend()
        
        # Representación latente (primeras 2 dimensiones)
        plt.subplot(2, 3, 3)
        plt.scatter(X_latent[:, 0], X_latent[:, 1], alpha=0.6, c=range(len(X_latent)), cmap='viridis')
        plt.title('Representación Latente (Dims 1-2)')
        plt.xlabel('Dimensión Latente 1')
        plt.ylabel('Dimensión Latente 2')
        plt.colorbar()
        
        # Distribución de dimensiones originales
        plt.subplot(2, 3, 4)
        plt.hist(X_scaled.flatten(), bins=50, alpha=0.7)
        plt.title('Distribución Datos Originales')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        
        # Distribución de representación latente
        plt.subplot(2, 3, 5)
        plt.hist(X_latent.flatten(), bins=50, alpha=0.7, color='orange')
        plt.title('Distribución Representación Latente')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        
        # Correlación entre original y reconstruido
        plt.subplot(2, 3, 6)
        correlaciones = []
        for i in range(min(10, n_features)):
            corr = np.corrcoef(X_scaled[:, i], X_reconstruido[:, i])[0, 1]
            correlaciones.append(corr)
        
        plt.bar(range(len(correlaciones)), correlaciones)
        plt.title('Correlación Original-Reconstruido')
        plt.xlabel('Dimensión')
        plt.ylabel('Correlación')
        
        plt.tight_layout()
        plt.show()
        
        return autoencoder, encoder, mse_reconstruccion
    
    def ejemplo_gan(self):
        """Ejemplo 2: GAN para Generación de Datos Sintéticos"""
        print("\n" + "=" * 60)
        print("⚔️  EJEMPLO 2: GAN - GENERACIÓN DE DATOS SINTÉTICOS")
        print("=" * 60)
        
        # Generar datos reales (2D blobs)
        print("📊 Generando datos reales...")
        X_real, _ = make_blobs(
            n_samples=1000,
            centers=3,
            n_features=2,
            cluster_std=1.0,
            random_state=42
        )
        
        # Escalar datos
        scaler = StandardScaler()
        X_real_scaled = scaler.fit_transform(X_real)
        
        print(f"Datos reales: {X_real_scaled.shape}")
        
        # Parámetros GAN
        latent_dim = 10
        data_dim = 2
        
        # Construir Generador
        print("🏗️  Construyendo GAN...")
        generator = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(latent_dim,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(data_dim, activation='linear')
        ])
        
        # Construir Discriminador
        discriminator = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(data_dim,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilar discriminador
        discriminator.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Construir GAN combinado
        discriminator.trainable = False
        gan_input = keras.layers.Input(shape=(latent_dim,))
        generated_data = generator(gan_input)
        gan_output = discriminator(generated_data)
        gan = keras.Model(gan_input, gan_output)
        gan.compile(optimizer='adam', loss='binary_crossentropy')
        
        print("✅ GAN construido")
        
        # Entrenar GAN
        print("\n🎯 Entrenando GAN...")
        batch_size = 32
        epochs = 200
        
        perdidas_g = []
        perdidas_d = []
        
        for epoch in range(epochs):
            # Entrenar discriminador
            idx = np.random.randint(0, X_real_scaled.shape[0], batch_size)
            real_data = X_real_scaled[idx]
            
            # Generar datos falsos
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_data = generator.predict(noise, verbose=0)
            
            # Etiquetas
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            
            # Entrenar discriminador
            d_loss_real = discriminator.train_on_batch(real_data, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            # Entrenar generador
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            misleading_labels = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, misleading_labels)
            
            perdidas_g.append(g_loss)
            perdidas_d.append(d_loss)
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
        
        # Generar datos sintéticos
        print("\n🎨 Generando datos sintéticos...")
        n_generated = 1000
        noise = np.random.normal(0, 1, (n_generated, latent_dim))
        X_generated = generator.predict(noise, verbose=0)
        
        # Desescalar para visualización
        X_real_orig = scaler.inverse_transform(X_real_scaled)
        X_generated_orig = scaler.inverse_transform(X_generated)
        
        # Visualizar resultados
        plt.figure(figsize=(15, 10))
        
        # Pérdidas durante entrenamiento
        plt.subplot(2, 3, 1)
        plt.plot(perdidas_g, label='Generador')
        plt.plot(perdidas_d, label='Discriminador')
        plt.title('Pérdidas durante Entrenamiento GAN')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        # Datos reales
        plt.subplot(2, 3, 2)
        plt.scatter(X_real_orig[:, 0], X_real_orig[:, 1], alpha=0.6, label='Datos Reales')
        plt.title('Datos Reales')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.legend()
        
        # Datos generados
        plt.subplot(2, 3, 3)
        plt.scatter(X_generated_orig[:, 0], X_generated_orig[:, 1], alpha=0.6, color='red', label='Datos Generados')
        plt.title('Datos Generados por GAN')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.legend()
        
        # Comparación
        plt.subplot(2, 3, 4)
        plt.scatter(X_real_orig[:, 0], X_real_orig[:, 1], alpha=0.6, label='Reales')
        plt.scatter(X_generated_orig[:, 0], X_generated_orig[:, 1], alpha=0.6, color='red', label='Generados')
        plt.title('Comparación: Reales vs Generados')
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.legend()
        
        # Distribuciones
        plt.subplot(2, 3, 5)
        plt.hist(X_real_orig[:, 0], bins=30, alpha=0.7, label='Reales Dim 1')
        plt.hist(X_generated_orig[:, 0], bins=30, alpha=0.7, color='red', label='Generados Dim 1')
        plt.title('Distribución Dimensión 1')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.legend()
        
        plt.subplot(2, 3, 6)
        plt.hist(X_real_orig[:, 1], bins=30, alpha=0.7, label='Reales Dim 2')
        plt.hist(X_generated_orig[:, 1], bins=30, alpha=0.7, color='red', label='Generados Dim 2')
        plt.title('Distribución Dimensión 2')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return generator, discriminator, perdidas_g[-1], perdidas_d[-1]
    
    def ejemplo_transformer_clustering(self):
        """Ejemplo 3: Transformer para Clustering no Supervisado"""
        print("\n" + "=" * 60)
        print("🤖 EJEMPLO 3: TRANSFORMER - CLUSTERING NO SUPERVISADO")
        print("=" * 60)
        
        # Generar datos secuenciales
        print("📊 Generando datos secuenciales...")
        np.random.seed(42)
        
        # Crear secuencias con patrones diferentes
        def crear_secuencia_patron(patron, longitud=20, ruido=0.1):
            secuencia = []
            for i in range(longitud):
                if patron == 'seno':
                    valor = np.sin(i * 0.3) + np.random.normal(0, ruido)
                elif patron == 'lineal':
                    valor = i * 0.1 + np.random.normal(0, ruido)
                elif patron == 'cuadrada':
                    valor = 1.0 if (i // 5) % 2 == 0 else -1.0
                    valor += np.random.normal(0, ruido)
                secuencia.append(valor)
            return np.array(secuencia)
        
        # Generar múltiples secuencias
        secuencias = []
        etiquetas_true = []
        
        for _ in range(100):
            patron = np.random.choice(['seno', 'lineal', 'cuadrada'])
            secuencia = crear_secuencia_patron(patron)
            secuencias.append(secuencia)
            etiquetas_true.append(['seno', 'lineal', 'cuadrada'].index(patron))
        
        X = np.array(secuencias)
        y_true = np.array(etiquetas_true)
        
        print(f"Secuencias generadas: {X.shape}")
        
        # Construir Transformer encoder
        print("🏗️  Construyendo Transformer Encoder...")
        
        class TransformerEncoder(keras.Model):
            def __init__(self, num_heads=4, ff_dim=32, **kwargs):
                super().__init__(**kwargs)
                self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)
                self.ffn = keras.Sequential([
                    keras.layers.Dense(ff_dim, activation='relu'),
                    keras.layers.Dense(ff_dim)
                ])
                self.layernorm1 = keras.layers.LayerNormalization()
                self.layernorm2 = keras.layers.LayerNormalization()
                self.dropout1 = keras.layers.Dropout(0.1)
                self.dropout2 = keras.layers.Dropout(0.1)
            
            def call(self, inputs, training=None):
                # Self-attention
                attn_output = self.attention(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                
                # Feed-forward
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)
        
        # Modelo completo
        seq_length = X.shape[1]
        feature_dim = 1
        
        inputs = keras.layers.Input(shape=(seq_length, feature_dim))
        
        # Proyección inicial
        x = keras.layers.Dense(32)(inputs)
        
        # Transformer encoder layers
        for _ in range(2):
            x = TransformerEncoder()(x)
        
        # Global average pooling para obtener embedding
        embedding = keras.layers.GlobalAveragePooling1D()(x)
        
        # Modelo para obtener embeddings
        encoder_model = keras.Model(inputs, embedding)
        
        # Compilar (para pre-entrenamiento auto-supervisado)
        encoder_model.compile(optimizer='adam', loss='mse')
        
        print("✅ Transformer construido:")
        encoder_model.summary()
        
        # Pre-entrenamiento auto-supervisado (masking)
        print("\n🎯 Pre-entrenando Transformer...")
        
        # Crear datos enmascarados para pre-entrenamiento
        X_masked = X.copy()
        mask_positions = np.random.random(X.shape) < 0.15  # 15% de enmascaramiento
        X_masked[mask_positions] = 0
        
        # Entrenar para reconstruir
        historia = encoder_model.fit(
            X_masked, X,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        
        # Obtener embeddings
        embeddings = encoder_model.predict(X)
        
        print(f"Embeddings obtenidos: {embeddings.shape}")
        
        # Clustering con K-means sobre embeddings
        from sklearn.cluster import KMeans
        
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Evaluar clustering
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        # Visualizar resultados
        plt.figure(figsize=(15, 10))
        
        # Pérdida durante pre-entrenamiento
        plt.subplot(2, 3, 1)
        plt.plot(historia.history['loss'], label='Entrenamiento')
        plt.plot(historia.history['val_loss'], label='Validación')
        plt.title('Pérdida - Transformer Pre-entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        # Embeddings (2D con PCA)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.subplot(2, 3, 2)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Embeddings (Colores: Etiquetas Reales)')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        
        plt.subplot(2, 3, 3)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Embeddings (Colores: Clusters)')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        
        # Ejemplos de secuencias por cluster
        for i in range(n_clusters):
            plt.subplot(2, 3, 4+i)
            indices_cluster = np.where(cluster_labels == i)[0][:3]  # Primeras 3 secuencias del cluster
            for idx in indices_cluster:
                plt.plot(X[idx], alpha=0.7, label=f'Sec {idx}')
            plt.title(f'Cluster {i} - Ejemplos')
            plt.xlabel('Tiempo')
            plt.ylabel('Valor')
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return encoder_model, silhouette_avg
    
    def ejecutar_todos(self):
        """Ejecuta todos los ejemplos no supervisados"""
        print("🎯 EJECUTANDO EJEMPLOS DE APRENDIZAJE NO SUPERVISADO")
        print("=" * 80)
        
        resultados = {}
        
        for nombre, funcion in self.ejemplos.items():
            print(f"\n🔄 Ejecutando ejemplo: {nombre}")
            resultado = funcion()
            if nombre == 'Autoencoder':
                autoencoder, encoder, mse = resultado
                resultados[nombre] = {'metrica': mse, 'modelo': autoencoder}
            elif nombre == 'GAN':
                generator, discriminator, g_loss, d_loss = resultado
                resultados[nombre] = {'g_loss': g_loss, 'd_loss': d_loss, 'generator': generator}
            elif nombre == 'Transformer':
                encoder, silhouette = resultado
                resultados[nombre] = {'metrica': silhouette, 'modelo': encoder}
            print(f"✅ {nombre} completado")
        
        # Resumen
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE RESULTADOS - APRENDIZAJE NO SUPERVISADO")
        print("=" * 80)
        
        for nombre, resultado in resultados.items():
            if 'metrica' in resultado:
                print(f"{nombre:12}: {resultado['metrica']:.4f}")
            else:
                print(f"{nombre:12}: G_loss={resultado['g_loss']:.4f}, D_loss={resultado['d_loss']:.4f}")
        
        return resultados

def demo_ejemplos_no_supervisado():
    """Demostración completa de ejemplos no supervisados"""
    print("=" * 80)
    print("🎯 LABORATORIO 4 - EJEMPLOS NO SUPERVISADOS")
    print("=" * 80)
    
    # Crear y ejecutar ejemplos
    ejemplos = EjemplosNoSupervisado()
    resultados = ejemplos.ejecutar_todos()
    
    print("\n" + "=" * 80)
    print("🎉 EJEMPLOS NO SUPERVISADOS COMPLETADOS")
    print("=" * 80)

if __name__ == "__main__":
    demo_ejemplos_no_supervisado()
