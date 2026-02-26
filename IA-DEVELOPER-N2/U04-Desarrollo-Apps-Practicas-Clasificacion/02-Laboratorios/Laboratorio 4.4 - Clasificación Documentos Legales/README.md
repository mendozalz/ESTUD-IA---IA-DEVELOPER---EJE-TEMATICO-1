# Laboratorio 4.4: Clasificación de Documentos Legales con Transformers y ONNX

## 🎯 Objetivo del Laboratorio
Desarrollar un sistema de clasificación de documentos legales usando Legal-BERT con optimización ONNX para despliegue en Azure ML.

## 📋 Metodología de Marco Lógico

### **Jerarquía de Objetivos**
- **Fin**: Automatizar la categorización de documentos en bufetes de abogados
- **Propósito**: Implementar clasificador legal con >90% de precisión
- **Componentes**: Modelo Legal-BERT, exportación ONNX, despliegue Azure ML
- **Actividades**: Fine-tuning, optimización, despliegue cloud, monitoreo

### **Indicadores de Verificación**
| Indicador | Meta | Medio de Verificación |
|-----------|------|------------------------|
| Precisión legal | >90% | Reporte de evaluación |
| Optimización ONNX | 50% reducción tamaño | Archivos .onnx |
| Latencia en Azure | <150ms | Azure Monitor |
| Disponibilidad | 99.9% | Azure SLA |

## 🛠️ Tecnologías y Herramientas

### **Principal**
- Transformers 4.30.0
- Legal-BERT
- ONNX Runtime 1.16.0
- PyTorch 2.0.0
- Azure ML SDK

### **Despliegue**
- Azure ML
- Azure Container Instances
- Azure Monitor
- Azure DevOps

## 📁 Estructura del Laboratorio

```
Laboratorio 4.4/
├── data/
│   ├── legal_documents/
│   │   ├── contracts/
│   │   ├── patents/
│   │   ├── court_cases/
│   │   └── regulations/
│   └── annotations/
├── src/
│   ├── legal_bert.py
│   ├── export_onnx.py
│   ├── azure_deploy.py
│   └── monitor.py
├── models/
│   ├── legal_bert.pt
│   └── legal_bert.onnx
├── azure/
│   ├── environment.yml
│   ├── deploy.yml
│   └── scoring_script.py
├── tests/
│   ├── test_model.py
│   └── test_onnx.py
├── README.md
└── requirements.txt
```

## 🚀 Pasos del Laboratorio

### **Paso 1: Configuración del Entorno**
```bash
pip install transformers==4.30.0 onnxruntime==1.16.0 torch==2.0.0
pip install azureml-sdk azureml-core azureml-mlflow
```

### **Paso 2: Preparación de Datos Legales**
- Recolectar corpus de documentos legales
- Anotar documentos por categoría legal
- Preprocesar texto específico del dominio legal
- Dividir dataset con estratificación legal

### **Paso 3: Fine-tuning de Legal-BERT**
- Cargar Legal-BERT pre-entrenado
- Configurar cabeza de clasificación para 5 categorías
- Implementar learning rate scheduling
- Entrenar con técnicas de regularización legal

### **Paso 4: Exportación y Optimización ONNX**
- Convertir modelo PyTorch a ONNX
- Aplicar optimizaciones ONNX Runtime
- Validar precisión post-conversión
- Medir mejoras de rendimiento

### **Paso 5: Despliegue en Azure ML**
- Configurar workspace de Azure ML
- Crear environment y compute target
- Desplegar modelo como endpoint real-time
- Configurar autenticación y escalado

### **Paso 6: Monitoreo y Mantenimiento**
- Implementar logging con Azure Monitor
- Configurar alertas de rendimiento
- Crear dashboard de métricas
- Establecer pipeline de reentrenamiento

## 📊 Entregables

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| legal_bert.py | Implementación con Legal-BERT | Python |
| export_onnx.py | Conversión y optimización ONNX | Python |
| azure_deploy.py | Script de despliegue en Azure ML | Python |
| monitor.py | Sistema de monitoreo | Python |
| azure/ | Configuración completa de Azure | YAML/Python |
| models/ | Modelo optimizado ONNX | ONNX |
| tests/ | Suite de pruebas ONNX/Azure | Python |
| README.md | Documentación completa | Markdown |

## 🎯 Criterios de Éxito

- **Técnicos**: Precisión >90%, optimización 50% tamaño, latencia <150ms
- **Funcionales**: Endpoint Azure funcional, monitoreo activo
- **Profesionales**: Código production-ready, documentación Azure completa

## 📚 Recursos Adicionales

- [Legal-BERT Paper](https://arxiv.org/abs/2010.02502)
- [ONNX Documentation](https://onnx.ai/)
- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [NLP for Legal Text](https://aclanthology.org/2020.lrec-1.207/)

## 📞 Soporte

- **Foro**: [Enlace al foro del curso]
- **Horario de tutoría**: Martes y Jueves 14:00-16:00 UTC
- **Email**: ia-developer@ejemplo.com

---

**Duración estimada**: 1 semana  
**Dificultad**: Avanzado  
**Prerrequisitos**: U01, U02, U03, Laboratorios 4.1, 4.2 y 4.3 completados
