"""
Caso de Uso 1 - Optimización de Rutas de Entrega con GNNs
Fase 1: Ideación - Árbol de Problemas (Marco Lógico)
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

class ArbolProblemasLogistica:
    """
    Clase para analizar el árbol de problemas del caso de uso de optimización de rutas
    utilizando la metodología de Marco Lógico
    """
    
    def __init__(self):
        self.problema_central = "Ineficiencia en rutas de entrega"
        self.causas = [
            "Falta de datos en tiempo real",
            "Algoritmos de routing obsoletos", 
            "Cambios dinámicos en órdenes de entrega",
            "Infraestructura tecnológica limitada",
            "Falta de optimización para múltiples variables"
        ]
        
        self.efectos = [
            "Aumento en costos operativos",
            "Insatisfacción del cliente",
            "Retrasos en entregas",
            "Desperdicio de recursos (combustible, tiempo)",
            "Pérdida de competitividad"
        ]
        
        self.stakeholders = [
            "Equipos de logística",
            "Conductores", 
            "Clientes finales",
            "Gerencia de operaciones",
            "Proveedores de tecnología"
        ]
    
    def generar_arbol_problemas(self) -> Dict:
        """
        Genera el árbol de problemas completo
        """
        arbol = {
            "problema_central": self.problema_central,
            "causas": self.causas,
            "efectos": self.efectos,
            "stakeholders": self.stakeholders
        }
        return arbol
    
    def convertir_arbol_objetivos(self) -> Dict:
        """
        Convierte el árbol de problemas a árbol de objetivos
        """
        arbol_objetivos = {
            "objetivo_central": "Eficiencia optimizada en rutas de entrega",
            "medios": [
                "Integración de datos en tiempo real",
                "Implementación de algoritmos modernos de routing",
                "Sistema adaptable a cambios dinámicos",
                "Modernización de infraestructura tecnológica",
                "Optimización multivariable con IA"
            ],
            "fines": [
                "Reducción de costos operativos",
                "Mejora en satisfacción del cliente",
                "Cumplimiento en tiempos de entrega",
                "Uso eficiente de recursos",
                "Ventaja competitiva sostenible"
            ]
        }
        return arbol_objetivos
    
    def visualizar_arbol(self, tipo: str = "problemas"):
        """
        Visualiza el árbol de problemas u objetivos
        """
        if tipo == "problemas":
            arbol = self.generar_arbol_problemas()
            titulo = "Árbol de Problemas - Logística 2026"
        else:
            arbol = self.convertir_arbol_objetivos()
            titulo = "Árbol de Objetivos - Logística 2026"
        
        # Crear visualización simple con matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Izquierda: Problemas/Causas
        if tipo == "problemas":
            ax1.barh(range(len(self.causas)), [1]*len(self.causas), 
                     color='red', alpha=0.7)
            ax1.set_yticks(range(len(self.causas)))
            ax1.set_yticklabels(self.causas, fontsize=10)
            ax1.set_title('Causas del Problema Central', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Impacto')
        else:
            medios = arbol['medios']
            ax1.barh(range(len(medios)), [1]*len(medios), 
                     color='green', alpha=0.7)
            ax1.set_yticks(range(len(medios)))
            ax1.set_yticklabels(medios, fontsize=10)
            ax1.set_title('Medios para Alcanzar Objetivo', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Viabilidad')
        
        # Derecha: Efectos/Fines
        if tipo == "problemas":
            ax2.barh(range(len(self.efectos)), [1]*len(self.efectos), 
                     color='orange', alpha=0.7)
            ax2.set_yticks(range(len(self.efectos)))
            ax2.set_yticklabels(self.efectos, fontsize=10)
            ax2.set_title('Efectos del Problema Central', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Severidad')
        else:
            fines = arbol['fines']
            ax2.barh(range(len(fines)), [1]*len(fines), 
                     color='blue', alpha=0.7)
            ax2.set_yticks(range(len(fines)))
            ax2.set_yticklabels(fines, fontsize=10)
            ax2.set_title('Fines del Objetivo Central', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Importancia')
        
        plt.suptitle(titulo, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'arbol_{tipo}_logistica.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generar_documento_ideacion(self) -> str:
        """
        Genera el documento de ideación completo
        """
        arbol_problemas = self.generar_arbol_problemas()
        arbol_objetivos = self.convertir_arbol_objetivos()
        
        documento = f"""
# Documento de Ideación - Optimización de Rutas con GNNs

## 📋 Contexto del Proyecto

### Problema Central
{arbol_problemas['problema_central']}

## 🔍 Análisis de Problemas (Marco Lógico)

### Causas Identificadas
{chr(10).join([f"- {causa}" for causa in arbol_problemas['causas']])}

### Efectos Negativos
{chr(10).join([f"- {efecto}" for efecto in arbol_problemas['efectos']])}

### Stakeholders Afectados
{chr(10).join([f"- {stakeholder}" for stakeholder in arbol_problemas['stakeholders']])}

## 🎯 Análisis de Objetivos (Marco Lógico)

### Objetivo Central
{arbol_objetivos['objetivo_central']}

### Medios para Alcanzar el Objetivo
{chr(10).join([f"- {medio}" for medio in arbol_objetivos['medios']])}

### Fines Esperados
{chr(10).join([f"- {fin}" for fin in arbol_objetivos['fines']])}

## 📊 Métricas de Éxito Propuestas

### KPIs Principales
- Reducción del tiempo de entrega: 15%
- Disminución de costos operativos: 20%
- Mejora en satisfacción del cliente: 25%
- Optimización del uso de recursos: 30%

### Métricas Técnicas
- Precisión del modelo GNN: >90%
- Latencia en predicciones: <50ms
- Escalabilidad: 1000 solicitudes/segundo

## 🛠️ Propuesta de Solución

### Tecnología Principal
- **Graph Neural Networks (GNNs)** con TensorFlow 2.15
- **Capas personalizadas** para manejo de grafos dinámicos
- **Integración con datos en tiempo real** (Waze, GPS, IoT)

### Hardware Requerido
- GPU NVIDIA H100 para entrenamiento
- Edge devices (Jetson Orin) para inferencia
- Infraestructura cloud escalable

## 📅 Cronograma Estimado

1. **Fase 1 - Diseño**: 2 semanas
2. **Fase 2 - Desarrollo**: 4 semanas  
3. **Fase 3 - Optimización**: 2 semanas
4. **Fase 4 - Despliegue**: 1 semana
5. **Fase 5 - Monitoreo**: Continuo

## 🎯 Impacto Esperado

La implementación de esta solución permitirá transformar completamente la operación logística,
posicionando a la empresa como líder en innovación tecnológica del sector.

---
*Documento generado automáticamente con Marco Lógico - 2026*
"""
        return documento

def main():
    """
    Función principal para ejecutar el análisis de ideación
    """
    print("🚚 ANÁLISIS DE IDEACIÓN - OPTIMIZACIÓN DE RUTAS CON GNNS")
    print("=" * 70)
    
    # Crear instancia del analizador
    analizador = ArbolProblemasLogistica()
    
    # Generar árbol de problemas
    print("\n📊 Generando árbol de problemas...")
    arbol_problemas = analizador.generar_arbol_problemas()
    
    print(f"   Problema Central: {arbol_problemas['problema_central']}")
    print(f"   Causas Identificadas: {len(arbol_problemas['causas'])}")
    print(f"   Efectos Negativos: {len(arbol_problemas['efectos'])}")
    print(f"   Stakeholders: {len(arbol_problemas['stakeholders'])}")
    
    # Generar árbol de objetivos
    print("\n🎯 Generando árbol de objetivos...")
    arbol_objetivos = analizador.convertir_arbol_objetivos()
    
    print(f"   Objetivo Central: {arbol_objetivos['objetivo_central']}")
    print(f"   Medios Propuestos: {len(arbol_objetivos['medios'])}")
    print(f"   Fines Esperados: {len(arbol_objetivos['fines'])}")
    
    # Visualizar árboles
    print("\n📊 Generando visualizaciones...")
    analizador.visualizar_arbol("problemas")
    analizador.visualizar_arbol("objetivos")
    
    # Generar documento de ideación
    print("\n📝 Generando documento de ideación...")
    documento = analizador.generar_documento_ideacion()
    
    # Guardar documento
    with open('documento_ideacion.md', 'w', encoding='utf-8') as f:
        f.write(documento)
    
    print("   ✅ Documento guardado como 'documento_ideacion.md'")
    print("   ✅ Visualizaciones guardadas como PNG")
    
    # Guardar estructura JSON para uso posterior
    import json
    estructura_completa = {
        "arbol_problemas": arbol_problemas,
        "arbol_objetivos": arbol_objetivos,
        "documento_ideacion": documento
    }
    
    with open('estructura_ideacion.json', 'w', encoding='utf-8') as f:
        json.dump(estructura_completa, f, indent=2, ensure_ascii=False)
    
    print("   ✅ Estructura completa guardada como 'estructura_ideacion.json'")
    
    print("\n🎯 ANÁLISIS COMPLETADO")
    print("=" * 70)
    print("📋 PRÓXIMOS PASOS:")
    print("   1. Revisar el documento de ideación generado")
    print("   2. Validar con stakeholders")
    print("   3. Proceder a la fase de diseño técnico")
    print("=" * 70)

if __name__ == "__main__":
    main()
