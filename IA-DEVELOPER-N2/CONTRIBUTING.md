# Contributing to IA DEVELOPER N2

¡Gracias por tu interés en contribuir al curso IA DEVELOPER N2! Esta guía te ayudará a empezar.

## 🤝 **Cómo Contribuir**

### **Reportar Issues**
Si encuentras un error o tienes una sugerencia:
1. Ve a [Issues](https://github.com/tu-usuario/IA-DEVELOPER-N2/issues)
2. Busca si el issue ya existe
3. Si no existe, crea un nuevo issue con:
   - Título descriptivo
   - Descripción detallada
   - Pasos para reproducir (si es un bug)
   - Screenshots si es necesario

### **Proponer Mejoras**
1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/tu-mejora`
3. Realiza tus cambios
4. Commit: `git commit -m 'Añadir: descripción de tu mejora'`
5. Push: `git push origin feature/tu-mejora`
6. Crea un Pull Request

## 📝 **Guía de Estilo**

### **Código Python**
- Seguir [PEP 8](https://pep8.org/)
- Usar type hints cuando sea posible
- Docstrings para todas las funciones y clases
- Nombres descriptivos de variables y funciones

```python
def preprocess_images(image_paths: List[str], target_size: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess images for model input.
    
    Args:
        image_paths: List of image file paths
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed image array
    """
    pass
```

### **Documentación**
- READMEs claros y completos
- Comentarios en código complejo
- Ejemplos de uso
- Referencias a papers o recursos

### **Commits**
- Mensajes descriptivos y cortos
- Usar prefijos: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`
- Ejemplo: `feat: añadir soporte para nuevo dataset`

## 🏗️ **Estructura de Proyectos**

### **Laboratorios**
Cada laboratorio debe incluir:
- `README.md`: Descripción y uso
- `requirements.txt`: Dependencias específicas
- `scripts/`: Código principal
- `notebooks/`: Análisis exploratorio
- `data/`: Datos (sintéticos o públicos)
- `models/`: Modelos entrenados (opcional para git)

### **Proyectos Integradores**
- `app/`: API o aplicación principal
- `docs/`: Documentación técnica
- `tests/`: Tests unitarios
- `docker/`: Configuración Docker

## 🧪 **Testing**

### **Unit Tests**
```python
import pytest
from your_module import your_function

def test_your_function():
    # Arrange
    input_data = "test_input"
    expected = "expected_output"
    
    # Act
    result = your_function(input_data)
    
    # Assert
    assert result == expected
```

### **Integration Tests**
- Testear APIs completas
- Testear pipelines de datos
- Testear modelos con datos reales

## 📋 **Checklist para Pull Requests**

Antes de crear un PR, verifica:

- [ ] El código sigue la guía de estilo
- [ ] Los tests pasan
- [ ] La documentación está actualizada
- [ ] No hay archivos temporales
- [ ] Los commits tienen mensajes claros
- [ ] El PR tiene descripción detallada

## 🚀 **Proceso de Revisión**

1. **Automático**: Tests y linting
2. **Revisión por pares**: Otro estudiante revisa el código
3. **Revisión por instructor**: Validación final
4. **Merge**: Integración al main

## 🏆 **Reconocimiento**

Las contribuciones serán reconocidas en:
- README del proyecto
- Sección de contribuidores
- Certificado del curso (si aplica)

## 📞 **Contacto**

Para dudas sobre contribuciones:
- **Discord**: Canal #contribuciones
- **Email**: contributions@ia-developer.com
- **Issues**: Etiqueta `question`

---

¡Gracias por contribuir al aprendizaje de todos! 🎓
