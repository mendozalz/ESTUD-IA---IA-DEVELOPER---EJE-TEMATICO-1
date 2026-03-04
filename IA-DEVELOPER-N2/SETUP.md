# 🚀 Setup Guide - IA DEVELOPER N2

Guía rápida de configuración para el curso IA DEVELOPER N2.

## 📋 **Requisitos del Sistema**

### **Hardware Mínimo**
- **CPU**: Intel i5 o AMD Ryzen 5 (2018+)
- **RAM**: 8GB (16GB recomendado)
- **Almacenamiento**: 20GB libres
- **GPU**: No requerida (opcional para aceleración)

### **Software Requerido**
- **Python**: 3.9+ (recomendado 3.10)
- **Git**: 2.30+
- **VS Code** o **PyCharm** (IDE recomendado)

## 🛠️ **Instalación Paso a Paso**

### **1. Verificar Python**
```bash
python --version
# o
python3 --version
```

Si no tienes Python, descárgalo de [python.org](https://python.org)

### **2. Instalar Git**
```bash
# Windows: Descargar desde git-scm.com
# Linux: sudo apt install git
# Mac: brew install git
```

### **3. Clonar el Repositorio**
```bash
git clone https://github.com/tu-usuario/IA-DEVELOPER-N2.git
cd IA-DEVELOPER-N2
```

### **4. Crear Entorno Virtual**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **5. Actualizar pip**
```bash
pip install --upgrade pip
```

### **6. Instalar Dependencias**
```bash
# Instalar todo el curso
pip install -r requirements.txt

# O instalar por unidades
pip install -r U01-Introduccion-IA-Intermedio/requirements.txt
pip install -r U02-TensorFlow-Avanzado/requirements.txt
```

### **7. Verificar Instalación**
```bash
python -c "
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
print('✅ TensorFlow:', tf.__version__)
print('✅ PyTorch:', torch.__version__)
print('✅ NumPy:', np.__version__)
print('✅ Pandas:', pd.__version__)
"
```

## 🔧 **Configuración del IDE**

### **VS Code**
Instalar estas extensiones:
- Python
- Jupyter
- Pylance
- GitLens
- Docker

### **PyCharm**
1. File → New Project
2. Seleccionar "Previously configured interpreter"
3. Apuntar al entorno virtual creado

## 🐳 **Docker (Opcional)**

### **Instalar Docker**
- [Docker Desktop](https://www.docker.com/products/docker-desktop)

### **Construir Imagen**
```bash
docker build -t ia-developer-n2 .
docker run -it ia-developer-n2
```

## 📊 **Verificar Laboratorios**

### **Laboratorio 2.1**
```bash
cd U02-TensorFlow-Avanzado/02-Laboratorios/Lab-2.1-TF-Exercises/01-Carga-Imagenes-tfdata
python ejercicio_completo.py
```

### **Laboratorio 2.2**
```bash
cd U02-TensorFlow-Avanzado/02-Laboratorios/Lab-2.2-Custom-Layers/01-Optimizacion-Rutas-GNN
python simple_route_optimizer.py
```

### **Laboratorio 2.3**
```bash
cd U02-TensorFlow-Avanzado/02-Laboratorios/Lab-2.3-Redes-Avanzadas/proyecto1_logistica
python scripts/train_cnn.py
```

## 🚨 **Solución de Problemas Comunes**

### **Error: "No module named 'tensorflow'"**
```bash
pip install tensorflow
```

### **Error: "CUDA out of memory"**
```bash
# Reducir batch size en los scripts
# O usar CPU:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### **Error: "Permission denied" (Linux/Mac)**
```bash
chmod +x scripts/*.py
```

### **Error: "DLL load failed" (Windows)**
```bash
# Reinstalar TensorFlow
pip uninstall tensorflow
pip install tensorflow
```

## 📱 **Configuración Adicional**

### **Jupyter Notebooks**
```bash
jupyter notebook
# Acceder a http://localhost:8888
```

### **TensorBoard**
```bash
tensorboard --logdir=logs/
# Acceder a http://localhost:6006
```

### **GPU Setup (Opcional)**
```bash
# Verificar GPU
nvidia-smi

# Instalar CUDA y cuDNN (siguiendo guías de NVIDIA)
```

## 🎯 **Primeros Pasos**

1. **Explorar el README** principal
2. **Revisar la estructura** del curso
3. **Ejecutar un laboratorio** simple
4. **Unirse al Discord** del curso
5. **Presentarte** en el foro

## 📞 **Soporte Técnico**

### **Recursos**
- [Documentación de TensorFlow](https://tensorflow.org)
- [Foro del curso](https://forum.ia-developer.com)
- [Discord](https://discord.gg/ia-developer)

### **Contacto**
- **Email**: soporte@ia-developer.com
- **Issues**: [GitHub Issues](https://github.com/tu-usuario/IA-DEVELOPER-N2/issues)

---

## ✅ **Checklist Final**

- [ ] Python 3.9+ instalado
- [ ] Git configurado
- [ ] Repositorio clonado
- [ ] Entorno virtual creado
- [ ] Dependencias instaladas
- [ ] IDE configurado
- [ ] Laboratorio de prueba ejecutado
- [ ] Acceso a recursos del curso

🎉 **¡Listo para empezar!** 

Ahora puedes comenzar tu viaje en IA DEVELOPER N2.
