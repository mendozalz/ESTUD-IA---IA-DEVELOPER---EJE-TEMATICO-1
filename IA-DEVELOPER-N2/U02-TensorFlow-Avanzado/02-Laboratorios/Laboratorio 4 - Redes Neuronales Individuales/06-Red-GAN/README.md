# Red Neuronal Generative Adversarial Network (GAN)

## 📖 Teoría

### Definición
Las Redes Generativas Adversariales (GAN - Generative Adversarial Networks) son arquitecturas de aprendizaje no supervisado que consisten en dos redes neuronales que compiten entre sí: un generador que crea datos sintéticos y un discriminador que distingue entre datos reales y falsos.

### Características Principales
- **Arquitectura adversarial**: Dos redes compiten en un juego de suma cero
- **Generador**: Crea datos sintéticos que parecen reales
- **Discriminador**: Clasifica datos como reales o falsos
- **Aprendizaje no supervisado**: No requiere etiquetas
- **Generación de datos**: Puede crear nuevos datos similares al dataset original

### Componentes
1. **Generador**: Red que crea datos sintéticos a partir de ruido aleatorio
2. **Discriminador**: Red que clasifica datos como reales o generados
3. **Función de pérdida adversarial**: Minimax game entre las dos redes
4. **Ruido latente**: Vector aleatorio de entrada para el generador
5. **Optimización alternada**: Entrenamiento por turnos de ambas redes

### Ventajas
- **Generación de datos**: Puede crear nuevos datos realistas
- **Aprendizaje no supervisado**: No requiere etiquetas
- **Calidad alta**: Puede generar datos de alta fidelidad
- **Flexibilidad**: Aplicable a imágenes, texto, audio, etc.

### Limitaciones
- **Entrenamiento inestable**: Difícil de converger
- **Mode collapse**: El generador puede producir datos limitados
- **Requiere muchos datos**: Necesita datasets grandes
- **Computacionalmente intensivo**: Requiere mucho poder de cómputo

### Casos de Uso
- Generación de imágenes
- Data augmentation
- Super-resolución
- Traducción de imágenes
- Generación de texto

## 🎯 Ejercicio Práctico: Generación de Imágenes de Dígitos

### Tipo de Aprendizaje: **NO SUPERVISADO**
- **Por qué**: No usa etiquetas, aprende de datos no etiquetados
- **Datos**: Imágenes MNIST sin etiquetas para aprender distribución

### Implementación Completa

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("🎨 RED GAN - GENERACIÓN DE DÍGITOS MNIST")
print("=" * 60)

# --- 1. Cargar y preprocesar datos ---
print("📊 Cargando y preprocesando datos...")
mnist = tf.keras.datasets.mnist
(x_train, _), (_, _) = mnist.load_data()

# Normalizar a [-1, 1] para GAN
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)  # Añadir canal

print(f"   Datos de entrenamiento: {x_train.shape}")
print(f"   Rango de valores: [{x_train.min():.2f}, {x_train.max():.2f}]")

# --- 2. Definir el Generador ---
print("\n🏗️ Construyendo el Generador...")
def build_generator(latent_dim):
    model = models.Sequential([
        # Capa densa inicial
        layers.Dense(7 * 7 * 128, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        # Reshape a 7x7x128
        layers.Reshape((7, 7, 128)),
        
        # Upsampling a 14x14
        layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        # Upsampling a 28x28
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.Activation('tanh')  # Salida en [-1, 1]
    ])
    
    return model

# --- 3. Definir el Discriminador ---
print("🔍 Construyendo el Discriminador...")
def build_discriminator():
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1)  # Salida sin activación (logits)
    ])
    
    return model

# --- 4. Crear modelos ---
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Mostrar arquitecturas
print("\nArquitectura del Generador:")
generator.summary()

print("\nArquitectura del Discriminador:")
discriminator.summary()

# --- 5. Definir funciones de pérdida y optimizadores ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --- 6. Definir paso de entrenamiento ---
print("\n🚀 Iniciando entrenamiento...")
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# Semilla para visualización consistente
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# --- 7. Función para generar y guardar imágenes ---
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.suptitle(f'Imágenes Generadas - Época {epoch}')
    plt.tight_layout()
    plt.show()

# --- 8. Entrenamiento ---
def train(dataset, epochs):
    gen_losses = []
    disc_losses = []
    
    for epoch in range(epochs):
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        num_batches = 0
        
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss
            num_batches += 1
        
        # Promedios por época
        avg_gen_loss = epoch_gen_loss / num_batches
        avg_disc_loss = epoch_disc_loss / num_batches
        
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        print(f"Época {epoch + 1}/{epochs} - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
        
        # Generar imágenes cada 10 épocas
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
    
    return gen_losses, disc_losses

# --- 9. Preparar dataset y entrenar ---
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

gen_losses, disc_losses = train(train_dataset, EPOCHS)

# --- 10. Visualizar resultados finales ---
print("\n📊 Visualizando resultados finales...")

# Generar imágenes finales
generate_and_save_images(generator, EPOCHS, seed)

# Graficar pérdidas
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(gen_losses, label='Generador')
plt.plot(disc_losses, label='Discriminador')
plt.title('Pérdidas durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# --- 11. Análisis de diversidad ---
plt.subplot(1, 2, 2)
# Generar múltiples lotes para analizar diversidad
noise_samples = tf.random.normal([100, noise_dim])
generated_samples = generator(noise_samples, training=False)

# Calcular diversidad (varianza promedio)
diversity = tf.math.reduce_variance(generated_samples)
print(f"   Diversidad de imágenes generadas: {diversity:.4f}")

# Mostrar histograma de valores generados
plt.hist(generated_samples.numpy().flatten(), bins=50, alpha=0.7, density=True)
plt.title('Distribución de Valores Generados')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 12. Evaluar calidad ---
print("\n🎯 Evaluando calidad de imágenes generadas...")
# Generar imágenes para evaluación
eval_noise = tf.random.normal([1000, noise_dim])
eval_images = generator(eval_noise, training=False)

# Calcular estadísticas básicas
mean_intensity = tf.reduce_mean(tf.abs(eval_images))
std_intensity = tf.math.reduce_std(eval_images)

print(f"   Intensidad promedio: {mean_intensity:.4f}")
print(f"   Desviación estándar: {std_intensity:.4f}")
print(f"   Rango de valores: [{tf.reduce_min(eval_images):.4f}, {tf.reduce_max(eval_images):.4f}]")

# --- 13. Guardar modelos ---
generator.save('gan_generator_mnist.h5')
discriminator.save('gan_discriminator_mnist.h5')
print("\n💾 Modelos guardados:")
print("   • 'gan_generator_mnist.h5'")
print("   • 'gan_discriminator_mnist.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Épocas entrenadas: {EPOCHS}")
print(f"   • Pérdida final generador: {gen_losses[-1]:.4f}")
print(f"   • Pérdida final discriminador: {disc_losses[-1]:.4f}")
print(f"   • Total de parámetros generador: {generator.count_params():,}")
print(f"   • Total de parámetros discriminador: {discriminator.count_params():,}")
print("=" * 60)
```

## 📊 Resultados Esperados

```
🎨 RED GAN - GENERACIÓN DE DÍGITOS MNIST
============================================================
📊 Cargando y preprocesando datos...
   Datos de entrenamiento: (60000, 28, 28, 1)
   Rango de valores: [-1.00, 1.00]

🏗️ Construyendo el Generador...
🔍 Construyendo el Discriminador...

Arquitectura del Generador:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 6272)              627200    
 batch_normalization (BatchN (None, 6272)              25088     
 leaky_re_lu (LeakyReLU)     (None, 6272)              0         
 reshape (Reshape)           (None, 7, 7, 128)         0         
 conv2d_transpose (Conv2DTra (None, 7, 7, 64)          204800    
 batch_normalization_1 (Batc (None, 7, 7, 64)          256       
 leaky_re_lu_1 (LeakyReLU)   (None, 7, 7, 64)          0         
 conv2d_transpose_1 (Conv2DT (None, 14, 14, 64)       20480     
 batch_normalization_2 (Batc (None, 14, 14, 64)       256       
 leaky_re_lu_2 (LeakyReLU)   (None, 14, 14, 64)       0         
 conv2d_transpose_2 (Conv2DT (None, 28, 28, 1)        1600      
 activation (Activation)     (None, 28, 28, 1)        0         
=================================================================
Total params: 879,680
Trainable params: 879,552
Non-trainable params: 128
_________________________________________________________________

🚀 Iniciando entrenamiento...
Época 1/50 - Gen Loss: 2.3456, Disc Loss: 1.2345
Época 10/50 - Gen Loss: 1.8765, Disc Loss: 0.9876
...
Época 50/50 - Gen Loss: 0.8765, Disc Loss: 0.6543

📊 Visualizando resultados finales...

🎯 Evaluando calidad de imágenes generadas...
   Intensidad promedio: 0.2345
   Desviación estándar: 0.5678
   Rango de valores: [-0.9876, 0.9876]

✅ EJERCICIO COMPLETADO
============================================================
🎯 RESULTADOS:
   • Épocas entrenadas: 50
   • Pérdida final generador: 0.8765
   • Pérdida final discriminador: 0.6543
   • Total de parámetros generador: 879,680
   • Total de parámetros discriminador: 340,000
============================================================
```

## 🎓 Análisis de Aprendizaje

### Identificación del Paradigma
**✅ APRENDIZAJE NO SUPERVISADO**
- **Sin etiquetas**: No usa etiquetas de las imágenes MNIST
- **Aprendizaje adversarial**: Aprende la distribución de datos reales
- **Objetivo**: Generar datos que parezcan reales
- **Métrica**: Calidad visual y diversidad de imágenes generadas

### Ventajas de las GAN
- **Generación realista**: Puede crear datos muy realistas
- **Aprendizaje no supervisado**: No requiere etiquetas
- **Data augmentation**: Puede generar datos adicionales
- **Flexibilidad**: Aplicable a diferentes tipos de datos

### Limitaciones
- **Entrenamiento inestable**: Difícil de converger correctamente
- **Mode collapse**: El generador puede producir datos limitados
- **Computacionalmente intensivo**: Requiere mucho poder de cómputo
- **Difícil de evaluar**: No hay métricas objetivas claras

## 🚀 Adaptación al Proyecto Integrador

### Ideas para Adaptación
1. **Dominio específico**: Adaptar para generar datos del proyecto
2. **Conditional GAN**: Añadir condiciones para generar tipos específicos
3. **Data augmentation**: Usar para aumentar datasets limitados

### Ejemplo de Adaptación
```python
# Adaptar GAN para datos específicos del proyecto
def crear_gan_proyecto(input_shape, latent_dim=100):
    """
    Adapta GAN para generar datos específicos del proyecto
    """
    def build_generator():
        model = models.Sequential([
            layers.Dense(128, input_shape=(latent_dim,)),
            layers.LeakyReLU(),
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dense(np.prod(input_shape), activation='tanh'),
            layers.Reshape(input_shape)
        ])
        return model
    
    def build_discriminator():
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(512),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Dense(1)
        ])
        return model
    
    return build_generator(), build_discriminator()
```

## 📝 Ejercicios Adicionales

### Ejercicio 1: Supervisado (Conditional GAN)
```python
# Convertir GAN a Conditional GAN para generación supervisada
def crear_conditional_gan(num_classes, latent_dim=100):
    """Crear GAN condicional para generar clases específicas"""
    
    # Generador condicional
    def build_conditional_generator():
        noise = layers.Input(shape=(latent_dim,))
        label = layers.Input(shape=(1,))
        
        # Embedding de etiqueta
        label_embedding = layers.Embedding(num_classes, latent_dim)(label)
        label_embedding = layers.Flatten()(label_embedding)
        
        # Combinar ruido y etiqueta
        combined = layers.Concatenate()([noise, label_embedding])
        
        # Red generadora
        x = layers.Dense(256)(combined)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(784, activation='tanh')(x)
        x = layers.Reshape((28, 28, 1))(x)
        
        model = models.Model([noise, label], x)
        return model
    
    return build_conditional_generator()
```

### Ejercicio 2: Reforzado (GAN-RL)
```python
# Usar GAN en contexto de aprendizaje por refuerzo
class GANEnvironment:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.current_state = None
    
    def reset(self):
        """Reiniciar entorno"""
        noise = tf.random.normal([1, 100])
        self.current_state = self.generator(noise, training=False)
        return self.current_state
    
    def step(self, action):
        """Realizar acción y obtener recompensa"""
        # Acción: modificar imagen generada
        modified_state = self.current_state + action
        
        # Recompensa: qué tan realista es la imagen
        reward = self.discriminator(modified_state, training=False)[0, 0]
        
        # Nuevo estado
        self.current_state = modified_state
        
        return self.current_state, reward, False, {}
```

## 🔗 Recursos Adicionales

- [Documentación TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [GAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
- [Conditional GAN](https://www.tensorflow.org/tutorials/generative/cgan)
- [StyleGAN Guide](https://www.tensorflow.org/tutorials/generative/stylegan)
