import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("RED GAN - GENERACION DE DIGITOS MNIST")
print("=" * 60)

# --- 1. Cargar y preprocesar datos ---
print("Cargando y preprocesando datos...")
mnist = tf.keras.datasets.mnist
(x_train, _), (_, _) = mnist.load_data()

# Normalizar a [-1, 1] para GAN
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)  # Añadir canal

print(f"   Datos de entrenamiento: {x_train.shape}")
print(f"   Rango de valores: [{x_train.min():.2f}, {x_train.max():.2f}]")

# --- 2. Definir el Generador ---
print("\nConstruyendo el Generador...")
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
print("Construyendo el Discriminador...")
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
print("\nIniciando entrenamiento...")
BATCH_SIZE = 256
EPOCHS = 30  # Reducido para demostración
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
    
    plt.suptitle(f'Imagenes Generadas - Epoca {epoch}')
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
        
        print(f"Epoca {epoch + 1}/{epochs} - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
        
        # Generar imágenes cada 10 épocas
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
    
    return gen_losses, disc_losses

# --- 9. Preparar dataset y entrenar ---
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

gen_losses, disc_losses = train(train_dataset, EPOCHS)

# --- 10. Visualizar resultados finales ---
print("\nVisualizando resultados finales...")

# Generar imágenes finales
generate_and_save_images(generator, EPOCHS, seed)

# Graficar pérdidas
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(gen_losses, label='Generador')
plt.plot(disc_losses, label='Discriminador')
plt.title('Perdidas durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Perdida')
plt.legend()
plt.grid(True)

# --- 11. Análisis de diversidad ---
plt.subplot(1, 2, 2)
# Generar múltiples lotes para analizar diversidad
noise_samples = tf.random.normal([100, noise_dim])
generated_samples = generator(noise_samples, training=False)

# Calcular diversidad (varianza promedio)
diversity = tf.math.reduce_variance(generated_samples)
print(f"   Diversidad de imagenes generadas: {diversity:.4f}")

# Mostrar histograma de valores generados
plt.hist(generated_samples.numpy().flatten(), bins=50, alpha=0.7, density=True)
plt.title('Distribucion de Valores Generados')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 12. Evaluar calidad ---
print("\nEvaluando calidad de imagenes generadas...")
# Generar imágenes para evaluación
eval_noise = tf.random.normal([1000, noise_dim])
eval_images = generator(eval_noise, training=False)

# Calcular estadísticas básicas
mean_intensity = tf.reduce_mean(tf.abs(eval_images))
std_intensity = tf.math.reduce_std(eval_images)

print(f"   Intensidad promedio: {mean_intensity:.4f}")
print(f"   Desviacion estandar: {std_intensity:.4f}")
print(f"   Rango de valores: [{tf.reduce_min(eval_images):.4f}, {tf.reduce_max(eval_images):.4f}]")

# --- 13. Guardar modelos ---
generator.save('gan_generator_mnist.h5')
discriminator.save('gan_discriminator_mnist.h5')
print("\nModelos guardados:")
print("   • 'gan_generator_mnist.h5'")
print("   • 'gan_discriminator_mnist.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Epocas entrenadas: {EPOCHS}")
print(f"   • Perdida final generador: {gen_losses[-1]:.4f}")
print(f"   • Perdida final discriminador: {disc_losses[-1]:.4f}")
print(f"   • Total de parametros generador: {generator.count_params():,}")
print(f"   • Total de parametros discriminador: {discriminator.count_params():,}")
print("=" * 60)
