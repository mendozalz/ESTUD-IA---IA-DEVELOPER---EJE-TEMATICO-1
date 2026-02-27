# Laboratorio 4.2: Clasificación de Texto con BERT y Adaptadores

## 🎯 Objetivo del Laboratorio
Crear una plataforma de análisis de sentimientos para redes sociales y reseñas usando BERT con adaptadores para eficiencia y SHAP para explicabilidad.

## 📋 Metodología de Marco Lógico

### **Jerarquía de Objetivos**
- **Fin**: Automatizar el análisis de sentimientos en marketing digital
- **Propósito**: Implementar clasificador de texto con >85% de F1-Score
- **Componentes**: Modelo BERT con adaptadores, API con explicabilidad, dashboard
- **Actividades**: Tokenización, fine-tuning, explicabilidad, despliegue

### **Indicadores de Verificación**
| Indicador | Meta | Medio de Verificación |
|-----------|------|------------------------|
| F1-Score | >0.85 | Reporte de clasificación |
| Tiempo de inferencia | <200ms | Logs de API |
| Explicabilidad con SHAP | 100% | Visualizaciones generadas |
| Documentación API | 100% | Swagger/OpenAPI |

## 🛠️ Tecnologías y Herramientas

### **Principal**
- Transformers 4.30.0
- BERT base-uncased
- PyTorch 2.0.0
- SHAP 0.42.0
- FastAPI

### **Procesamiento**
- NLTK
- spaCy
- Pandas
- NumPy

## 📁 Estructura del Laboratorio

```
Laboratorio 4.2/
├── data/
│   ├── reviews.csv
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── bert_adapters.py
│   ├── explain_model.py
│   └── api.py
├── models/
│   ├── bert_adapters.pt
│   └── tokenizer/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── README.md
└── requirements.txt
```

## 🚀 Pasos del Laboratorio

### **Paso 1: Configuración del Entorno**
```bash
pip install transformers==4.30.0 torch==2.0.0 shap==0.42.0 fastapi uvicorn
pip install adapters==0.2.0
```

### **Paso 2: Preparación de Datos**
- Cargar dataset de reseñas (ej: Amazon Reviews, Twitter)
- Limpieza y preprocesamiento de texto
- Tokenización con BERT tokenizer
- División train/validation/test

### **Paso 3: Implementación con Adaptadores**
- Cargar BERT pre-entrenado
- Añadir adaptadores para clasificación
- Configurar fine-tuning eficiente
- Entrenar solo los adaptadores

### **Paso 4: Explicabilidad con SHAP**
- Configurar SHAP explainer para BERT
- Generar explicaciones para predicciones
- Visualizar importancia de tokens
- Crear reportes de interpretabilidad

### **Paso 5: API REST con FastAPI**
- Implementar endpoints de predicción
- Incluir explicabilidad en respuestas
- Validar entradas con Pydantic
- Documentación automática con Swagger

### **Paso 6: Testing y Validación**
- Pruebas unitarias del modelo
- Tests de integración de API
- Validación de rendimiento
- Documentación de resultados

## 📊 Entregables

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| data_loader.py | Script para tokenización de textos | Python |
| bert_adapters.py | Implementación de BERT + Adaptadores | Python |
| explain_model.py | Visualización de importancia con SHAP | Python |
| api.py | Servicio FastAPI con explicabilidad | Python |
| requirements.txt | Dependencias del proyecto | Text |
| README.md | Documentación completa | Markdown |
| notebooks/ | Análisis exploratorio | Jupyter |

## 🎯 Criterios de Éxito

- **Técnicos**: F1-Score >0.85, inferencia <200ms, explicabilidad funcional
- **Funcionales**: API completa con documentación, visualizaciones SHAP
- **Profesionales**: Código modular, testing adecuado, buenas prácticas

## 📚 Recursos Adicionales

- [Hugging Face Course](https://huggingface.co/course/)
- [AdapterHub Documentation](https://docs.adapterhub.ml/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

## 📞 Soporte

- **Foro**: [Enlace al foro del curso]
- **Horario de tutoría**: Martes y Jueves 14:00-16:00 UTC
- **Email**: ia-developer@ejemplo.com

---

**Duración estimada**: 1 semana  
**Dificultad**: Intermedio  
**Prerrequisitos**: U01, U02, U03 completadas
