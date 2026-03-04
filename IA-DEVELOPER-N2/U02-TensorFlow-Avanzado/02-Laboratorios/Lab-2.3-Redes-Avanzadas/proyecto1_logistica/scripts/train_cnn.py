# -*- coding: utf-8 -*-
"""
Proyecto 1: Automatización en Logística
Entrenamiento de CNN para detección de daños en paquetes
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

def create_data_generators(data_dir="data/processed", img_size=(224, 224), batch_size=32):
    """Crear generadores de datos para entrenamiento y validación"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, val_generator

def build_cnn_model(input_shape=(224, 224, 3)):
    """Construir modelo CNN para detección de daños"""
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(model, train_generator, val_generator, epochs=20):
    """Entrenar el modelo CNN"""
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            'models/cnn_best_model.h5', save_best_only=True
        )
    ]
    
    # Entrenamiento
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """Visualizar historial de entrenamiento"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('notebooks/cnn_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_model(model, val_generator):
    """Evaluar el modelo en datos de validación"""
    
    results = model.evaluate(val_generator, verbose=0)
    metrics_names = model.metrics_names
    
    print("\n📊 Resultados de Evaluación:")
    for name, value in zip(metrics_names, results):
        print(f"   {name}: {value:.4f}")
    
    return dict(zip(metrics_names, results))

def main():
    """Función principal de entrenamiento"""
    
    print("🚀 Iniciando entrenamiento de CNN para detección de daños en paquetes")
    print("=" * 60)
    
    # Crear directorios necesarios
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    # 1. Preparar datos
    print("\n📁 Preparando datos...")
    train_gen, val_gen = create_data_generators()
    print(f"   Clases: {train_gen.class_indices}")
    print(f"   Muestras de entrenamiento: {len(train_gen)} batches")
    print(f"   Muestras de validación: {len(val_gen)} batches")
    
    # 2. Construir modelo
    print("\n🏗️ Construyendo modelo CNN...")
    model = build_cnn_model()
    model.summary()
    
    # 3. Entrenar modelo
    print("\n🎯 Entrenando modelo...")
    history = train_model(model, train_gen, val_gen, epochs=20)
    
    # 4. Evaluar modelo
    print("\n📈 Evaluando modelo...")
    results = evaluate_model(model, val_gen)
    
    # 5. Visualizar resultados
    print("\n📊 Generando visualizaciones...")
    plot_training_history(history)
    
    # 6. Guardar modelo final
    print("\n💾 Guardando modelo...")
    model.save('models/cnn_model.h5')
    print("   Modelo guardado como 'models/cnn_model.h5'")
    
    # 7. Resumen final
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"📊 Métricas Finales:")
    print(f"   • Accuracy: {results['accuracy']:.4f}")
    print(f"   • Precision: {results['precision']:.4f}")
    print(f"   • Recall: {results['recall']:.4f}")
    print(f"   • Loss: {results['loss']:.4f}")
    print(f"📁 Archivos generados:")
    print(f"   • models/cnn_model.h5 - Modelo entrenado")
    print(f"   • models/cnn_best_model.h5 - Mejor modelo durante entrenamiento")
    print(f"   • notebooks/cnn_training_history.png - Gráficos de entrenamiento")
    print("=" * 60)

if __name__ == "__main__":
    main()
