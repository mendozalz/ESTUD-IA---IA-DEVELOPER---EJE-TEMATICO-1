# Laboratorio 6.3: Dashboard de Monitoreo con Streamlit

## 🎯 Objetivos del Laboratorio

### Objetivo General
Crear un dashboard interactivo de monitoreo de modelos de IA usando Streamlit, con visualización en tiempo real de métricas, alertas y análisis de rendimiento.

### Objetivos Específicos
- Desarrollar un dashboard interactivo con Streamlit 2.0
- Integrar múltiples fuentes de datos (Prometheus, logs, bases de datos)
- Implementar visualizaciones en tiempo real con Plotly
- Crear sistema de alertas y notificaciones
- Aplicar metodología Windsor para gestión del proyecto

## 📋 Marco Lógico del Proyecto

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Dashboard Streamlit** | Visualización de métricas en tiempo real | <2s refresh rate | Código app.py con optimización |
| **Integración Prometheus** | Recolectar métricas de modelos | Métricas actualizadas cada 5s | Conector Prometheus funcional |
| **Sistema de Alertas** | Notificar anomalías automáticamente | <30s tiempo de detección | Sistema de alertas configurado |
| **Análisis de Rendimiento** | Identificar cuellos de botella | Reportes generados automáticamente | Módulos de análisis implementados |

## 🛠️ Tecnologías y Herramientas

### Principales Tecnologías
- **Streamlit 2.0**: Framework para aplicaciones de datos
- **Plotly 5.15**: Visualizaciones interactivas
- **Prometheus 3.0**: Sistema de monitoreo
- **Grafana 10.0**: Dashboards complementarios
- **Pandas 2.0**: Manipulación de datos

### Dependencias Adicionales
- **Redis 7.0**: Caching de datos
- **PostgreSQL 15**: Base de datos histórica
- **Docker 24.0**: Contenerización
- **APScheduler**: Tareas programadas

## 📁 Estructura del Proyecto

```
Laboratorio 6.3 - Dashboard de Monitoreo/
├── README.md                           # Guía del laboratorio
├── windsor_plan.md                    # Planificación Windsor
├── requirements.txt                    # Dependencias Python
├── docker-compose.yml                 # Configuración local
├── Dockerfile                          # Imagen Docker
├── app.py                             # Aplicación Streamlit principal
├── config/                            # Configuraciones
│   ├── __init__.py
│   ├── settings.py                    # Configuración general
│   └── prometheus_config.py           # Configuración Prometheus
├── data/                              # Datos y caché
│   ├── cache/                         # Caché Redis
│   └── logs/                          # Logs de la aplicación
├── src/                               # Código fuente
│   ├── __init__.py
│   ├── components/                     # Componentes Streamlit
│   │   ├── __init__.py
│   │   ├── metrics_dashboard.py       # Dashboard de métricas
│   │   ├── alerts_panel.py            # Panel de alertas
│   │   ├── performance_analysis.py    # Análisis de rendimiento
│   │   └── model_monitoring.py       # Monitoreo de modelos
│   ├── services/                       # Servicios de datos
│   │   ├── __init__.py
│   │   ├── prometheus_service.py      # Conector Prometheus
│   │   ├── database_service.py        # Conector base de datos
│   │   └── alert_service.py           # Servicio de alertas
│   ├── utils/                          # Utilidades
│   │   ├── __init__.py
│   │   ├── data_processor.py          # Procesamiento de datos
│   │   ├── visualization.py            # Utilidades de visualización
│   │   └── notifications.py            # Sistema de notificaciones
│   └── models/                         # Modelos de datos
│       ├── __init__.py
│       ├── metrics.py                  # Modelos de métricas
│       └── alerts.py                   # Modelos de alertas
├── tests/                             # Tests
│   ├── test_app.py
│   ├── test_services.py
│   └── test_components.py
├── monitoring/                        # Configuración monitoreo
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alert_rules.yml
│   └── grafana/
│       └── dashboards/
└── docs/                              # Documentación
    ├── user_guide.md
    ├── api_reference.md
    └── deployment_guide.md
```

## 🔧 Implementación Detallada

### Fase 1: Aplicación Streamlit Principal

#### app.py:
```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Optional

# Configuración de la página
st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar componentes
from src.components.metrics_dashboard import MetricsDashboard
from src.components.alerts_panel import AlertsPanel
from src.components.performance_analysis import PerformanceAnalysis
from src.components.model_monitoring import ModelMonitoring
from src.services.prometheus_service import PrometheusService
from src.services.alert_service import AlertService
from src.utils.data_processor import DataProcessor
from config.settings import Settings

# Inicializar servicios
@st.cache_resource
def init_services():
    """Inicializar servicios de la aplicación"""
    return {
        'prometheus': PrometheusService(),
        'alerts': AlertService(),
        'data_processor': DataProcessor()
    }

# Configuración principal
def main():
    """Función principal de la aplicación"""
    
    # Título y descripción
    st.title("🤖 ML Model Monitoring Dashboard")
    st.markdown("---")
    
    # Inicializar servicios
    services = init_services()
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selección de modelos
        selected_models = st.multiselect(
            "Seleccionar Modelos",
            ["product_classifier", "sentiment_analyzer", "recommendation_engine"],
            default=["product_classifier"]
        )
        
        # Rango de tiempo
        time_range = st.selectbox(
            "Rango de Tiempo",
            ["Última hora", "Últimas 6 horas", "Últimas 24 horas", "Últimos 7 días"],
            index=1
        )
        
        # Frecuencia de actualización
        refresh_interval = st.selectbox(
            "Frecuencia de Actualización",
            [5, 10, 30, 60],
            index=0
        )
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        
        st.markdown("---")
        
        # Estado del sistema
        st.header("📊 Estado del Sistema")
        
        # Métricas del sistema
        system_metrics = services['prometheus'].get_system_metrics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "CPU Usage",
                f"{system_metrics.get('cpu_usage', 0):.1f}%",
                delta=f"{system_metrics.get('cpu_delta', 0):.1f}%"
            )
        
        with col2:
            st.metric(
                "Memory Usage",
                f"{system_metrics.get('memory_usage', 0):.1f}%",
                delta=f"{system_metrics.get('memory_delta', 0):.1f}%"
            )
    
    # Contenido principal
    if not selected_models:
        st.warning("Por favor selecciona al menos un modelo para monitorear")
        return
    
    # Tabs para diferentes vistas
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Métricas en Tiempo Real",
        "🚨 Alertas",
        "📈 Análisis de Rendimiento",
        "🤖 Monitoreo de Modelos"
    ])
    
    with tab1:
        metrics_dashboard = MetricsDashboard(services)
        metrics_dashboard.render(selected_models, time_range)
    
    with tab2:
        alerts_panel = AlertsPanel(services)
        alerts_panel.render()
    
    with tab3:
        performance_analysis = PerformanceAnalysis(services)
        performance_analysis.render(selected_models, time_range)
    
    with tab4:
        model_monitoring = ModelMonitoring(services)
        model_monitoring.render(selected_models)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

# Página de métricas detalladas
def metrics_detail_page():
    """Página detallada de métricas"""
    st.title("📊 Métricas Detalladas")
    
    # Parámetros
    model_name = st.query_params.get("model", "product_classifier")
    metric_type = st.selectbox(
        "Tipo de Métrica",
        ["Latencia", "Throughput", "Error Rate", "CPU", "Memory"]
    )
    
    # Obtener datos
    services = init_services()
    prometheus_service = services['prometheus']
    
    # Gráfico de métricas
    fig = go.Figure()
    
    if metric_type == "Latencia":
        data = prometheus_service.get_latency_metrics(model_name, hours=24)
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['value'],
            mode='lines+markers',
            name='Latencia (ms)',
            line=dict(color='blue')
        ))
    
    elif metric_type == "Throughput":
        data = prometheus_service.get_throughput_metrics(model_name, hours=24)
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['value'],
            mode='lines+markers',
            name='Requests/sec',
            line=dict(color='green')
        ))
    
    fig.update_layout(
        title=f"{metric_type} - {model_name}",
        xaxis_title="Tiempo",
        yaxis_title=metric_type,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas
    st.subheader("📈 Estadísticas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Promedio", f"{data['value'].mean():.2f}")
    
    with col2:
        st.metric("Mínimo", f"{data['value'].min():.2f}")
    
    with col3:
        st.metric("Máximo", f"{data['value'].max():.2f}")
    
    with col4:
        st.metric("Percentil 95", f"{data['value'].quantile(0.95):.2f}")

# Página de configuración de alertas
def alerts_config_page():
    """Página de configuración de alertas"""
    st.title("🚨 Configuración de Alertas")
    
    services = init_services()
    alert_service = services['alerts']
    
    # Formulario de nueva alerta
    with st.form("new_alert"):
        st.subheader("Nueva Alerta")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox(
                "Modelo",
                ["product_classifier", "sentiment_analyzer", "recommendation_engine"]
            )
            
            metric_type = st.selectbox(
                "Métrica",
                ["latency", "error_rate", "cpu_usage", "memory_usage"]
            )
        
        with col2:
            threshold = st.number_input(
                "Umbral",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.1
            )
            
            comparison = st.selectbox(
                "Comparación",
                [">", "<", ">=", "<=", "=="]
            )
        
        notification_channels = st.multiselect(
            "Canales de Notificación",
            ["email", "slack", "webhook"],
            default=["email"]
        )
        
        submitted = st.form_submit_button("Crear Alerta")
        
        if submitted:
            alert_config = {
                'model_name': model_name,
                'metric_type': metric_type,
                'threshold': threshold,
                'comparison': comparison,
                'notification_channels': notification_channels
            }
            
            try:
                alert_service.create_alert(alert_config)
                st.success("Alerta creada exitosamente")
            except Exception as e:
                st.error(f"Error al crear alerta: {e}")
    
    # Lista de alertas existentes
    st.subheader("Alertas Configuradas")
    
    alerts = alert_service.get_all_alerts()
    
    if alerts:
        for alert in alerts:
            with st.expander(f"Alerta: {alert['model_name']} - {alert['metric_type']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.json(alert)
                
                with col2:
                    if st.button("Eliminar", key=f"delete_{alert['id']}"):
                        alert_service.delete_alert(alert['id'])
                        st.rerun()
    else:
        st.info("No hay alertas configuradas")

# Navegación
def navigation():
    """Sistema de navegación"""
    page = st.sidebar.selectbox(
        "Navegación",
        ["Dashboard Principal", "Métricas Detalladas", "Configuración de Alertas"]
    )
    
    if page == "Dashboard Principal":
        main()
    elif page == "Métricas Detalladas":
        metrics_detail_page()
    elif page == "Configuración de Alertas":
        alerts_config_page()

if __name__ == "__main__":
    navigation()
```

### Fase 2: Componentes del Dashboard

#### src/components/metrics_dashboard.py:
```python
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

class MetricsDashboard:
    def __init__(self, services: Dict):
        self.prometheus_service = services['prometheus']
        self.data_processor = services['data_processor']
    
    def render(self, selected_models: List[str], time_range: str):
        """Renderizar el dashboard de métricas"""
        
        # Métricas en tiempo real
        st.header("📊 Métricas en Tiempo Real")
        
        # Grid de métricas principales
        cols = st.columns(len(selected_models))
        
        for i, model in enumerate(selected_models):
            with cols[i]:
                self._render_model_metrics(model)
        
        # Gráficos de tendencias
        st.subheader("📈 Tendencias de Rendimiento")
        
        # Convertir time_range a horas
        time_mapping = {
            "Última hora": 1,
            "Últimas 6 horas": 6,
            "Últimas 24 horas": 24,
            "Últimos 7 días": 168
        }
        
        hours = time_mapping.get(time_range, 6)
        
        # Gráficos por modelo
        for model in selected_models:
            with st.expander(f"Tendencias - {model}", expanded=True):
                self._render_trend_charts(model, hours)
        
        # Comparación entre modelos
        if len(selected_models) > 1:
            st.subheader("🔄 Comparación de Modelos")
            self._render_model_comparison(selected_models, hours)
    
    def _render_model_metrics(self, model_name: str):
        """Renderizar métricas de un modelo específico"""
        
        # Obtener métricas actuales
        metrics = self.prometheus_service.get_current_metrics(model_name)
        
        # Tarjeta del modelo
        st.markdown(f"### {model_name}")
        
        # Métricas principales
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Latencia",
                f"{metrics.get('latency', 0):.1f}ms",
                delta=f"{metrics.get('latency_delta', 0):.1f}ms"
            )
            
            st.metric(
                "Throughput",
                f"{metrics.get('throughput', 0):.1f} req/s",
                delta=f"{metrics.get('throughput_delta', 0):.1f} req/s"
            )
        
        with col2:
            st.metric(
                "Error Rate",
                f"{metrics.get('error_rate', 0):.2f}%",
                delta=f"{metrics.get('error_rate_delta', 0):.2f}%"
            )
            
            st.metric(
                "CPU",
                f"{metrics.get('cpu_usage', 0):.1f}%",
                delta=f"{metrics.get('cpu_delta', 0):.1f}%"
            )
        
        # Estado del modelo
        status = metrics.get('status', 'unknown')
        status_color = {
            'healthy': '🟢',
            'warning': '🟡',
            'critical': '🔴',
            'unknown': '⚪'
        }.get(status, '⚪')
        
        st.markdown(f"**Estado:** {status_color} {status.title()}")
    
    def _render_trend_charts(self, model_name: str, hours: int):
        """Renderizar gráficos de tendencias"""
        
        # Obtener datos históricos
        latency_data = self.prometheus_service.get_latency_metrics(model_name, hours)
        throughput_data = self.prometheus_service.get_throughput_metrics(model_name, hours)
        error_data = self.prometheus_service.get_error_rate_metrics(model_name, hours)
        
        # Crear subplots
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de latencia
            fig_latency = go.Figure()
            fig_latency.add_trace(go.Scatter(
                x=latency_data['timestamp'],
                y=latency_data['value'],
                mode='lines+markers',
                name='Latencia (ms)',
                line=dict(color='blue', width=2)
            ))
            
            fig_latency.update_layout(
                title="Latencia (ms)",
                xaxis_title="Tiempo",
                yaxis_title="Latencia (ms)",
                height=300
            )
            
            st.plotly_chart(fig_latency, use_container_width=True)
        
        with col2:
            # Gráfico de throughput
            fig_throughput = go.Figure()
            fig_throughput.add_trace(go.Scatter(
                x=throughput_data['timestamp'],
                y=throughput_data['value'],
                mode='lines+markers',
                name='Requests/sec',
                line=dict(color='green', width=2)
            ))
            
            fig_throughput.update_layout(
                title="Throughput (req/s)",
                xaxis_title="Tiempo",
                yaxis_title="Requests/sec",
                height=300
            )
            
            st.plotly_chart(fig_throughput, use_container_width=True)
        
        # Gráfico de error rate
        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(
            x=error_data['timestamp'],
            y=error_data['value'],
            mode='lines+markers',
            name='Error Rate (%)',
            line=dict(color='red', width=2)
        ))
        
        fig_error.update_layout(
            title="Error Rate (%)",
            xaxis_title="Tiempo",
            yaxis_title="Error Rate (%)",
            height=300
        )
        
        st.plotly_chart(fig_error, use_container_width=True)
    
    def _render_model_comparison(self, models: List[str], hours: int):
        """Renderizar comparación entre modelos"""
        
        # Recolectar datos de todos los modelos
        comparison_data = []
        
        for model in models:
            metrics = self.prometheus_service.get_current_metrics(model)
            comparison_data.append({
                'Model': model,
                'Latency (ms)': metrics.get('latency', 0),
                'Throughput (req/s)': metrics.get('throughput', 0),
                'Error Rate (%)': metrics.get('error_rate', 0),
                'CPU Usage (%)': metrics.get('cpu_usage', 0)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Gráficos de comparación
        col1, col2 = st.columns(2)
        
        with col1:
            # Comparación de latencia
            fig_latency = px.bar(
                df_comparison,
                x='Model',
                y='Latency (ms)',
                title="Comparación de Latencia",
                color='Model'
            )
            st.plotly_chart(fig_latency, use_container_width=True)
        
        with col2:
            # Comparación de throughput
            fig_throughput = px.bar(
                df_comparison,
                x='Model',
                y='Throughput (req/s)',
                title="Comparación de Throughput",
                color='Model'
            )
            st.plotly_chart(fig_throughput, use_container_width=True)
        
        # Tabla comparativa
        st.subheader("📊 Tabla Comparativa")
        st.dataframe(df_comparison, use_container_width=True)
        
        # Radar chart para comparación multidimensional
        fig_radar = go.Figure()
        
        for i, model in enumerate(models):
            model_data = df_comparison[df_comparison['Model'] == model].iloc[0]
            
            # Normalizar valores para radar chart
            normalized_values = [
                model_data['Latency (ms)'] / df_comparison['Latency (ms)'].max(),
                model_data['Throughput (req/s)'] / df_comparison['Throughput (req/s)'].max(),
                1 - (model_data['Error Rate (%)'] / df_comparison['Error Rate (%)'].max()),
                1 - (model_data['CPU Usage (%)'] / df_comparison['CPU Usage (%)'].max())
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=['Latency', 'Throughput', 'Stability', 'Efficiency'],
                fill='toself',
                name=model
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparación Multidimensional"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
```

## 📊 Entregables del Laboratorio

### 1. Código Fuente Completo
- **app.py**: Aplicación Streamlit principal
- **src/components/**: Componentes del dashboard
- **src/services/**: Servicios de integración
- **config/**: Configuraciones del sistema

### 2. Configuración de Despliegue
- **Dockerfile**: Imagen Docker optimizada
- **docker-compose.yml**: Configuración local completa
- **monitoring/**: Configuración Prometheus/Grafana

### 3. Tests y Documentación
- **tests/**: Suite de tests completa
- **docs/**: Documentación de usuario y API

---

**Duración Estimada**: 6-8 horas  
**Dificultad**: Intermedia  
**Prerrequisitos**: Conocimientos de Streamlit, Prometheus y visualización de datos
