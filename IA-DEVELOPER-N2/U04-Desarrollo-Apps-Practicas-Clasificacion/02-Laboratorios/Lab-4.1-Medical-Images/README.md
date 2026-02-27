# Laboratorio 4.1: Clasificación de Imágenes Médicas con EfficientNetV3 y ViT

## 🎯 Objetivo del Laboratorio
Desarrollar un sistema de clasificación de imágenes médicas para diagnóstico asistido usando EfficientNetV3 y Vision Transformers (ViT).

## 📋 Metodología de Marco Lógico

### **Jerarquía de Objetivos**
- **Fin**: Mejorar la precisión del diagnóstico médico mediante IA
- **Propósito**: Implementar un sistema de clasificación de imágenes médicas con >90% de accuracy
- **Componentes**: Modelo híbrido, API REST, sistema de explicabilidad
- **Actividades**: Preprocesamiento, entrenamiento, optimización, despliegue

### **Indicadores de Verificación**
| Indicador | Meta | Medio de Verificación |
|-----------|------|------------------------|
| Accuracy del modelo | >90% | Reporte de evaluación |
| Latencia de inferencia | <100ms | Logs de API |
| Tamaño del modelo optimizado | <10MB | Archivo .tflite |
| Documentación completa | 100% | README.md |

## 🛠️ Tecnologías y Herramientas

### **Principal**
- TensorFlow 2.15.0
- EfficientNetV3
- Vision Transformers (ViT)
- Albumentations 1.3.0
- OpenCV 4.9.0

### **Despliegue**
- FastAPI
- TensorFlow Lite
- Docker

## 📁 Estructura del Laboratorio

```
Laboratorio 4.1/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_loader.py
│   ├── hybrid_model.py
│   ├── train_eval.py
│   └── api.py
├── models/
│   ├── best_model.h5
│   └── model.tflite
├── README.md
└── requirements.txt
```

## 🚀 Pasos del Laboratorio

### **Paso 1: Configuración del Entorno**
```bash
pip install tensorflow==2.15.0 albumentations==1.3.0 opencv-python==4.9.0 fastapi uvicorn
```

### **Paso 2: Preparación de Datos**
- Descargar dataset de imágenes médicas (ej: Chest X-Ray)
- Organizar en carpetas train/validation/test
- Aplicar aumento de datos con Albumentations

### **Paso 3: Construcción del Modelo Híbrido**
- Implementar EfficientNetV3 como backbone
- Añadir bloque de atención tipo ViT
- Combinar características para clasificación final

### **Paso 4: Entrenamiento y Evaluación**
- Entrenar con early stopping
- Evaluar con métricas médicas (sensitivity, specificity)
- Generar confusion matrix

### **Paso 5: Optimización para Edge**
- Convertir a TensorFlow Lite
- Aplicar quantization
- Validar precisión post-optimización

### **Paso 6: Despliegue con API**
- Crear API FastAPI para predicciones
- Implementar endpoints de salud y predicción
- Dockerizar la aplicación

## 📊 Entregables

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| data_loader.py | Script para carga y aumento de imágenes | Python |
| hybrid_model.py | Implementación de EfficientNetV3 + ViT | Python |
| train_eval.py | Script para entrenar y evaluar el modelo | Python |
| model.tflite | Modelo optimizado para edge | TensorFlow Lite |
| api.py | Servicio FastAPI para clasificación | Python |
| Dockerfile | Contenedor para despliegue | Docker |
| README.md | Documentación completa | Markdown |

## 🎯 Criterios de Éxito

- **Técnicos**: Accuracy >90%, latencia <100ms, modelo <10MB
- **Funcionales**: API funcional, documentación completa
- **Profesionales**: Código limpio, buenas prácticas, testing

## 📚 Recursos Adicionales

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Vision Transformers Paper](https://arxiv.org/abs/2010.11929)
- [Albumentations Documentation](https://albumentations.ai/)
- [Medical ImageNet](https://medmnist.com/)

## 📞 Soporte

- **Foro**: [Enlace al foro del curso]
- **Horario de tutoría**: Martes y Jueves 14:00-16:00 UTC
- **Email**: ia-developer@ejemplo.com

---

**Duración estimada**: 1 semana  
**Dificultad**: Intermedio-Avanzado  
**Prerrequisitos**: U01, U02, U03 completadas
