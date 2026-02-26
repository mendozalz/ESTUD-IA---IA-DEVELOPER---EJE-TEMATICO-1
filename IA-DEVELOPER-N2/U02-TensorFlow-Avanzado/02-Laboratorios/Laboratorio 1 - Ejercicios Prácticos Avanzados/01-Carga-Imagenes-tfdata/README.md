# Ejercicio 1: Carga de Imágenes con tf.data

## 🏭 Caso de Uso: Detección de Defectos en Placas de Circuito Impreso (PCB)

### Contexto Empresarial
En la industria electrónica, la detección automática de defectos en Placas de Circuito Impreso (PCB) reduce costos y mejora la calidad. Las inspecciones manuales son lentas y propensas a errores humanos.

### Problema a Resolver
- **Detección**: Identificar defectos como grietas, soldaduras incorrectas, componentes faltantes
- **Velocidad**: Procesamiento en tiempo real en línea de producción
- **Precisión**: Minimizar falsos positivos y negativos
- **ROI**: Reducción del 15-20% en desperdicios

## 🎯 Objetivos de Aprendizaje

1. **Manejo avanzado de tf.data**:
   - Carga eficiente de imágenes con prefetching y paralelización
   - Aumento de datos (ImageDataGenerator) para mejorar generalización

2. **Construcción de modelos CNN**:
   - Uso de arquitecturas como EfficientNet o ResNet con transfer learning
   - Optimización para edge devices

3. **Despliegue en edge devices**:
   - Conversión a TensorFlow Lite para implementación en dispositivos embebidos
   - Integración con sistemas industriales

## 📊 Dataset

### Opción 1: Dataset Público - KolektorSDD2
- **Descripción**: Defectos en superficies industriales
- **Descarga**: [KolektorSDD2 Dataset](https://www.kaggle.com/datasets/barkhakahvi/kolektorsdd2)
- **Tamaño**: ~1000 imágenes de alta resolución

### Opción 2: Datos Sintéticos (para desarrollo)
- **Generación**: Usar OpenCV para simular defectos
- **Ventajas**: Control total sobre tipos y cantidad de defectos

## 🛠️ Implementación

### Paso 1: Configuración del Entorno
```python
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
```

### Paso 2: Funciones de Preprocesamiento
```python
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
```

### Paso 3: Generación de Datos Sintéticos
```python
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
```

### Paso 4: Construcción del Modelo
```python
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

# Compilar modelo
model = build_pcb_defect_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()
```

### Paso 5: Entrenamiento con Callbacks Avanzados
```python
def train_pcb_model(train_dataset, val_dataset, epochs=20):
    """Entrenar modelo con callbacks avanzados"""
    
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
```

### Paso 6: Conversión a TensorFlow Lite
```python
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
```

### Paso 7: Despliegue en Edge Device
```python
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
```

## 🧪 Ejecución Completa

```python
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

if __name__ == "__main__":
    main()
```

## 📊 Métricas de Evaluación

### Métricas de Clasificación
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Verdaderos positivos / (Verdaderos positivos + Falsos positivos)
- **Recall**: Verdaderos positivos / (Verdaderos positivos + Falsos negativos)
- **F1-Score**: Media armónica de precision y recall

### Métricas de Producción
- **Latencia**: Tiempo de inferencia por imagen (<100ms)
- **Throughput**: Imágenes procesadas por segundo (>30 fps)
- **Tamaño del modelo**: Espacio en disco (<5MB para TFLite)

## 🚀 Desafíos Adicionales

### Desafío 1: Mejora de Datos
- Implementar aumento de datos más realista
- Usar GANs para generar más datos de defectos
- Balancear clases con técnicas avanzadas

### Desafío 2: Optimización
- Implementar pruning para reducir tamaño del modelo
- Usar knowledge distillation
- Optimizar para hardware específico (Coral TPU)

### Desafío 3: Integración Industrial
- Conectar con sistema MES (Manufacturing Execution System)
- Implementar dashboard de monitoreo en tiempo real
- Añadir sistema de alertas por correo/mensaje

## 📝 Entrega Requerida

1. **Código fuente** completo y funcional
2. **Informe técnico** con:
   - Arquitectura del modelo
   - Resultados experimentales
   - Análisis de ROI
3. **Prototipo desplegado** en Docker
4. **Video demostrativo** del sistema en acción

## 🔗 Recursos Adicionales

- [TensorFlow Data Guide](https://www.tensorflow.org/guide/data)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [OpenCV Computer Vision](https://opencv.org/)
