import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import json

def main():
    """Función principal para ejecutar el ejercicio completo"""
    
    print("🏭 EJERCICIO 1: DETECCIÓN DE DEFECTOS EN PCB")
    print("=" * 60)
    
    # 1. Generar datos sintéticos (si no se tienen datos reales)
    print("📊 Generando datos sintéticos...")
    generate_synthetic_pcb_defects(num_images=1000)
    
    # 2. Preparar datasets
    print("🔄 Preparando datasets...")
    image_paths = []
    labels = []
    
    # Cargar rutas de imágenes
    for filename in os.listdir("synthetic_pcb"):
        if filename.endswith('.jpg'):
            image_paths.append(f"synthetic_pcb/{filename}")
            labels.append(1 if 'defect' in filename else 0)
    
    # Dividir en entrenamiento/validación
    split = int(0.8 * len(image_paths))
    train_paths, val_paths = image_paths[:split], image_paths[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    
    train_dataset = create_dataset(train_paths, train_labels, batch_size=32)
    val_dataset = create_dataset(val_paths, val_labels, batch_size=32)
    
    print(f"   Imágenes de entrenamiento: {len(train_paths)}")
    print(f"   Imágenes de validación: {len(val_paths)}")
    
    # 3. Entrenar modelo
    print("\n🚀 Entrenando modelo...")
    history = train_pcb_model(train_dataset, val_dataset, epochs=10)
    
    # 4. Evaluar modelo
    print("\n📈 Evaluando modelo...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(val_dataset)
    print(f"   Precisión: {test_acc:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    
    # 5. Convertir a TFLite
    print("\n🔄 Convirtiendo a TensorFlow Lite...")
    convert_to_tflite('best_pcb_model.h5')
    
    # 6. Probar despliegue
    print("\n🎯 Probando despliegue en edge device...")
    predictor = deploy_on_edge_device()
    
    # Probar con una imagen de prueba
    test_image = "synthetic_pcb/defect_0.jpg" if os.path.exists("synthetic_pcb/defect_0.jpg") else val_paths[0]
    prediction = predictor(test_image)
    print(f"   Predicción para {test_image}: {prediction}")
    
    print("\n✅ EJERCICIO COMPLETADO")
    print("=" * 60)
    print("🎯 RESULTADOS:")
    print(f"   • Precisión final: {test_acc:.4f}")
    print(f"   • Precision: {test_precision:.4f}")
    print(f"   • Recall: {test_recall:.4f}")
    print(f"   • Modelo TFLite: pcb_model.tflite")
    print(f"   • ROI estimado: 15-20% reducción en desperdicios")
    print("=" * 60)

def load_and_preprocess_image(path, label, img_size=(224, 224)):
    """Cargar y preprocesar imagen con aumento de datos"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    
    # Aumento de datos
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def create_dataset(image_paths, labels, batch_size=32, img_size=(224, 224)):
    """Crear dataset optimizado con tf.data"""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image(path, label, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def generate_synthetic_pcb_defects(num_images=1000, output_dir="synthetic_pcb"):
    """Generar imágenes sintéticas de PCB con defectos"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        # Crear imagen base (simulación de PCB)
        img = np.random.randint(50, 100, (224, 224, 3), dtype=np.uint8)
        
        # Añadir líneas de circuito
        for _ in range(np.random.randint(5, 15)):
            x1, y1 = np.random.randint(0, 224, 2)
            x2, y2 = np.random.randint(0, 224, 2)
            cv2.line(img, (x1, y1), (x2, y2), (150, 150, 150), 1)
        
        # Decidir si tiene defecto
        has_defect = np.random.random() > 0.5
        
        if has_defect:
            defect_type = np.random.choice(['crack', 'missing_component', 'bad_solder'])
            
            if defect_type == 'crack':
                # Añadir grieta
                x, y = np.random.randint(0, 224, 2)
                length = np.random.randint(10, 50)
                angle = np.random.uniform(0, 2*np.pi)
                end_x = int(x + length * np.cos(angle))
                end_y = int(y + length * np.sin(angle))
                cv2.line(img, (x, y), (end_x, end_y), (0, 0, 0), 2)
                
            elif defect_type == 'missing_component':
                # Añadir círculo vacío (componente faltante)
                x, y = np.random.randint(20, 204, 2)
                cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
                
            elif defect_type == 'bad_solder':
                # Añadir soldadura incorrecta
                x, y = np.random.randint(10, 214, 2)
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        
        # Guardar imagen
        filename = f"{'defect' if has_defect else 'good'}_{i}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), img)
    
    print(f"Generadas {num_images} imágenes sintéticas en {output_dir}")

def build_pcb_defect_model(input_shape=(224, 224, 3)):
    """Construir modelo CNN con transfer learning"""
    # Cargar EfficientNet pre-entrenado
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False  # Congelar capas base
    
    # Añadir capas personalizadas
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Clasificación binaria
    ])
    
    return model

def train_pcb_model(train_dataset, val_dataset, epochs=20):
    """Entrenar modelo con callbacks avanzados"""
    
    # Construir y compilar modelo
    global model
    model = build_pcb_defect_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, 
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_pcb_model.h5', 
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.1, 
            patience=3,
            monitor='val_loss'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def convert_to_tflite(model_path, tflite_path='pcb_model.tflite'):
    """Convertir modelo a TensorFlow Lite para edge devices"""
    
    # Cargar modelo entrenado
    model = tf.keras.models.load_model(model_path)
    
    # Convertir a TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizaciones
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Cuantización para reducir tamaño
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # Guardar modelo
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Modelo guardado como {tflite_path}")
    print(f"Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    return tflite_model

def deploy_on_edge_device():
    """Ejemplo de despliegue en Raspberry Pi/Jetson Nano"""
    
    # Cargar modelo TFLite
    interpreter = tf.lite.Interpreter(model_path='pcb_model.tflite')
    interpreter.allocate_tensors()
    
    # Obtener detalles de entrada/salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    def predict_image(image_path):
        """Predecir si una imagen tiene defectos"""
        # Cargar y preprocesar imagen
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Realizar predicción
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        return "Defecto detectado" if output[0][0] > 0.5 else "Sin defectos"
    
    return predict_image

if __name__ == "__main__":
    main()
