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
