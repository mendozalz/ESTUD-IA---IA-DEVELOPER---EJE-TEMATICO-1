# Laboratorio 4.5: Clasificación de Productos en Retail con Vision Transformers (ViT) y Quantization

## 🎯 Objetivo del Laboratorio
Desarrollar un sistema de clasificación de productos en estantes de retail usando Vision Transformers (ViT) con optimización para edge devices mediante quantization.

## 📋 Metodología de Marco Lógico

### **Jerarquía de Objetivos**
- **Fin**: Automatizar la gestión de inventario en tiendas retail
- **Propósito**: Implementar clasificador ViT con >92% de accuracy en edge
- **Componentes**: Modelo ViT, quantization, inferencia edge, monitoreo
- **Actividades**: Transfer learning, optimización, despliegue edge, validación

### **Indicadores de Verificación**
| Indicador | Meta | Medio de Verificación |
|-----------|------|------------------------|
| Accuracy en edge | >92% | Tests en dispositivo |
| Tamaño modelo quantizado | <5MB | Archivo .tflite |
| Inferencia en edge | <50ms | Logs del dispositivo |
| Consumo energético | <1W | Mediciones hardware |

## 🛠️ Tecnologías y Herramientas

### **Principal**
- TensorFlow 2.15.0
- Vision Transformers (ViT)
- TensorFlow Lite
- TensorFlow Model Optimization
- OpenCV 4.9.0

### **Edge Deployment**
- Raspberry Pi 4 / Jetson Nano
- TensorFlow Lite Interpreter
- Edge TPU (opcional)
- Docker Edge

## 📁 Estructura del Laboratorio

```
Laboratorio 4.5/
├── data/
│   ├── retail_products/
│   │   ├── beverages/
│   │   ├── snacks/
│   │   ├── dairy/
│   │   └── produce/
│   └── shelf_images/
├── src/
│   ├── vit_model.py
│   ├── quantize_model.py
│   ├── edge_inference.py
│   └── camera_interface.py
├── models/
│   ├── vit_base.h5
│   ├── vit_quantized.tflite
│   └── labels.txt
├── edge/
│   ├── deploy_edge.sh
│   ├── inference_loop.py
│   └── monitoring.py
├── tests/
│   ├── test_vit.py
│   └── test_edge.py
├── README.md
└── requirements.txt
```

## 🚀 Pasos del Laboratorio

### **Paso 1: Configuración del Entorno**
```bash
pip install tensorflow==2.15.0 tensorflow-datasets==4.9.0
pip install vit-keras opencv-python==4.9.0 tflite-runtime
```

### **Paso 2: Preparación de Datos Retail**
- Capturar imágenes de productos en estantes reales
- Anotar productos por categoría
- Aplicar augmentations específicas para retail
- Crear dataset balanceado por categoría

### **Paso 3: Implementación de Vision Transformers**
- Cargar ViT pre-entrenado (ViT-B16)
- Configurar transfer learning para productos retail
- Implementar data pipeline eficiente
- Entrenar con técnicas de regularización

### **Paso 4: Quantization para Edge**
- Aplicar post-training quantization
- Convertir a TensorFlow Lite
- Optimizar para hardware específico (ARM)
- Validar precisión post-quantization

### **Paso 5: Despliegue en Edge Device**
- Configurar Raspberry Pi/Jetson Nano
- Instalar TensorFlow Lite Runtime
- Implementar loop de inferencia con cámara
- Configurar monitoreo local

### **Paso 6: Optimización y Validación**
- Medir latencia y throughput
- Optimizar para consumo energético
- Validar en condiciones de iluminación variables
- Implementar fallback para errores

## 📊 Entregables

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| vit_model.py | Implementación con Vision Transformers | Python |
| quantize_model.py | Script de optimización y quantization | Python |
| edge_inference.py | Inferencia optimizada para edge | Python |
| camera_interface.py | Interface con cámara en tiempo real | Python |
| models/ | Modelo quantizado para despliegue | TensorFlow Lite |
| edge/ | Scripts de despliegue en edge | Bash/Python |
| tests/ | Suite de pruebas edge | Python |
| README.md | Documentación completa | Markdown |

## 🎯 Criterios de Éxito

- **Técnicos**: Accuracy >92%, modelo <5MB, inferencia <50ms, consumo <1W
- **Funcionales**: Sistema funcional en edge, cámara integrada, monitoreo activo
- **Profesionales**: Código optimizado, documentación edge completa, testing en hardware

## 📚 Recursos Adicionales

- [Vision Transformers Paper](https://arxiv.org/abs/2010.11929)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [Edge TPU Documentation](https://coral.ai/docs/edgetpu/)
- [Raspberry Pi ML](https://www.raspberrypi.org/learning/machine-learning/)

## 📞 Soporte

- **Foro**: [Enlace al foro del curso]
- **Horario de tutoría**: Martes y Jueves 14:00-16:00 UTC
- **Email**: ia-developer@ejemplo.com

---

**Duración estimada**: 1 semana  
**Dificultad**: Avanzado  
**Prerrequisitos**: U01, U02, U03, Laboratorios 4.1-4.4 completados
