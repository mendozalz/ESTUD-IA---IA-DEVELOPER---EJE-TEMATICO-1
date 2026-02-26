import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

class SegmentadorClientes:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.datos_originales = None
        self.datos_escalados = None
        self.n_clusters_optimo = None
        
    def generar_datos_clientes(self, n_muestras=300):
        """Genera datos sintéticos de clientes"""
        print("👥 Generando datos sintéticos de clientes...")
        
        np.random.seed(42)
        
        # Crear grupos de clientes con diferentes patrones
        centros = [
            [25, 30000, 3],    # Jóvenes, ingresos bajos, compras frecuentes
            [45, 80000, 2],    # Mediana edad, ingresos medios, compras moderadas
            [65, 120000, 1],   # Mayores, ingresos altos, compras ocasionales
            [35, 50000, 4]     # Adultos jóvenes, ingresos medios-bajos, compras muy frecuentes
        ]
        
        # Generar datos con diferentes dispersiones
        X, y = make_blobs(
            n_samples=n_muestras,
            centers=centros,
            cluster_std=0.7,
            random_state=42
        )
        
        # Crear DataFrame
        self.datos_originales = pd.DataFrame(X, columns=['edad', 'ingresos', 'frecuencia_compra'])
        
        # Añadir ruido realista
        self.datos_originales['edad'] = np.abs(self.datos_originales['edad'])
        self.datos_originales['ingresos'] = np.abs(self.datos_originales['ingresos']) * 1000
        self.datos_originales['frecuencia_compra'] = np.abs(self.datos_originales['frecuencia_compra'])
        
        print(f"✅ Datos generados: {len(self.datos_originales)} clientes")
        print(f"📊 Estadísticas descriptivas:")
        print(self.datos_originales.describe())
        
        return self.datos_originales
    
    def visualizar_datos_originales(self):
        """Visualiza los datos originales"""
        if self.datos_originales is None:
            print("❌ Datos no generados")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Scatter plot 3D proyectado en 2D
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(
            self.datos_originales['edad'], 
            self.datos_originales['ingresos'],
            c=self.datos_originales['frecuencia_compra'],
            cmap='viridis', alpha=0.6
        )
        plt.colorbar(scatter, label='Frecuencia de compra')
        plt.xlabel('Edad')
        plt.ylabel('Ingresos ($)')
        plt.title('Clientes: Edad vs Ingresos')
        
        # Distribución de edad
        plt.subplot(1, 3, 2)
        self.datos_originales['edad'].hist(bins=20, alpha=0.7)
        plt.xlabel('Edad')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Edad')
        
        # Distribución de ingresos
        plt.subplot(1, 3, 3)
        self.datos_originales['ingresos'].hist(bins=20, alpha=0.7, color='orange')
        plt.xlabel('Ingresos ($)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Ingresos')
        
        plt.tight_layout()
        plt.show()
        
    def preprocesar_datos(self):
        """Preprocesa los datos para clustering"""
        print("🔧 Preprocesando datos...")
        
        if self.datos_originales is None:
            print("❌ Datos no generados")
            return
        
        # Escalar datos
        self.datos_escalados = self.scaler.fit_transform(self.datos_originales)
        
        print("✅ Datos escalados con StandardScaler")
        print(f"📊 Forma de datos escalados: {self.datos_escalados.shape}")
        
    def encontrar_numero_optimo_clusters(self, max_k=10):
        """Encuentra el número óptimo de clusters usando múltiples métodos"""
        print("🔍 Buscando número óptimo de clusters...")
        
        if self.datos_escalados is None:
            print("❌ Datos no preprocesados")
            return
        
        # Método del codo
        inercias = []
        silhouettes = []
        davies_bouldins = []
        
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            etiquetas = kmeans.fit_predict(self.datos_escalados)
            
            inercias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.datos_escalados, etiquetas))
            davies_bouldins.append(davies_bouldin_score(self.datos_escalados, etiquetas))
        
        # Visualizar métricas
        plt.figure(figsize=(15, 5))
        
        # Método del codo
        plt.subplot(1, 3, 1)
        plt.plot(k_range, inercias, 'bo-')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Inercia')
        plt.title('Método del Codo')
        plt.grid(True)
        
        # Silhouette score
        plt.subplot(1, 3, 2)
        plt.plot(k_range, silhouettes, 'go-')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        plt.grid(True)
        
        # Davies-Bouldin score
        plt.subplot(1, 3, 3)
        plt.plot(k_range, davies_bouldins, 'ro-')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Davies-Bouldin Score')
        plt.title('Davies-Bouldin Score')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Encontrar k óptimo
        # Para silhouette: mayor es mejor
        # Para Davies-Bouldin: menor es mejor
        k_optimo_silhouette = k_range[np.argmax(silhouettes)]
        k_optimo_db = k_range[np.argmin(davies_bouldins)]
        
        print(f"📊 Resultados:")
        print(f"   k óptimo (Silhouette): {k_optimo_silhouette} (score: {max(silhouettes):.3f})")
        print(f"   k óptimo (Davies-Bouldin): {k_optimo_db} (score: {min(davies_bouldins):.3f})")
        
        # Elegir k basado en consenso o silhouette
        self.n_clusters_optimo = k_optimo_silhouette
        print(f"✅ k seleccionado: {self.n_clusters_optimo}")
        
        return k_optimo_silhouette, k_optimo_db
        
    def construir_y_ajustar_modelo(self, n_clusters=None):
        """Construye y ajusta el modelo K-means"""
        if n_clusters is None:
            n_clusters = self.n_clusters_optimo
            
        print(f"🏗️  Construyendo modelo K-means con {n_clusters} clusters...")
        
        if self.datos_escalados is None:
            print("❌ Datos no preprocesados")
            return
        
        # Construir y ajustar modelo
        self.modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        etiquetas = self.modelo.fit_predict(self.datos_escalados)
        
        print("✅ Modelo ajustado")
        
        # Calcular métricas finales
        sil_score = silhouette_score(self.datos_escalados, etiquetas)
        db_score = davies_bouldin_score(self.datos_escalados, etiquetas)
        
        print(f"📊 Métricas finales:")
        print(f"   Silhouette Score: {sil_score:.3f}")
        print(f"   Davies-Bouldin Score: {db_score:.3f}")
        
        # Añadir etiquetas a los datos originales
        self.datos_originales['cluster'] = etiquetas
        
        return etiquetas
    
    def visualizar_clusters(self):
        """Visualiza los clusters encontrados"""
        if self.modelo is None or self.datos_originales is None:
            print("❌ Modelo o datos no disponibles")
            return
        
        # Reducir dimensionalidad para visualización
        pca = PCA(n_components=2)
        datos_pca = pca.fit_transform(self.datos_escalados)
        
        # Crear DataFrame para visualización
        df_viz = pd.DataFrame(datos_pca, columns=['PCA1', 'PCA2'])
        df_viz['cluster'] = self.datos_originales['cluster']
        
        # Visualizar
        plt.figure(figsize=(15, 5))
        
        # Scatter plot con PCA
        plt.subplot(1, 3, 1)
        sns.scatterplot(data=df_viz, x='PCA1', y='PCA2', hue='cluster', palette='viridis', s=60)
        plt.title(f'Clusters Visualizados con PCA (Varianza explicada: {pca.explained_variance_ratio_.sum():.2%})')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        
        # Edad vs Ingresos por cluster
        plt.subplot(1, 3, 2)
        sns.scatterplot(
            data=self.datos_originales,
            x='edad', y='ingresos',
            hue='cluster', palette='viridis', s=60
        )
        plt.title('Clusters: Edad vs Ingresos')
        plt.xlabel('Edad')
        plt.ylabel('Ingresos ($)')
        
        # Frecuencia de compra por cluster
        plt.subplot(1, 3, 3)
        sns.boxplot(data=self.datos_originales, x='cluster', y='frecuencia_compra')
        plt.title('Frecuencia de Compra por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Frecuencia de Compra')
        
        plt.tight_layout()
        plt.show()
        
    def analizar_perfiles_clusters(self):
        """Analiza las características de cada cluster"""
        if self.datos_originales is None or 'cluster' not in self.datos_originales.columns:
            print("❌ Datos o clusters no disponibles")
            return
        
        print("📊 Análisis de perfiles de clusters:")
        
        # Estadísticas por cluster
        perfil_cluster = self.datos_originales.groupby('cluster').agg({
            'edad': ['mean', 'std'],
            'ingresos': ['mean', 'std'],
            'frecuencia_compra': ['mean', 'std'],
            'cluster': 'count'
        }).round(2)
        
        print(perfil_cluster)
        
        # Interpretación de clusters
        print("\n🎯 Interpretación de clusters:")
        for cluster_id in sorted(self.datos_originales['cluster'].unique()):
            datos_cluster = self.datos_originales[self.datos_originales['cluster'] == cluster_id]
            
            edad_media = datos_cluster['edad'].mean()
            ingresos_medios = datos_cluster['ingresos'].mean()
            frecuencia_media = datos_cluster['frecuencia_compra'].mean()
            
            print(f"\n   Cluster {cluster_id}:")
            print(f"     Edad media: {edad_media:.1f} años")
            print(f"     Ingresos medios: ${ingresos_medios:,.0f}")
            print(f"     Frecuencia media: {frecuencia_media:.1f} compras/mes")
            
            # Asignar perfil
            if edad_media < 30 and ingresos_medios < 50000:
                perfil = "Jóvenes con ingresos bajos"
            elif edad_media > 50 and ingresos_medios > 80000:
                perfil = "Adultos mayores con ingresos altos"
            elif 30 <= edad_media <= 50 and 40000 <= ingresos_medios <= 80000:
                perfil = "Adultos con ingresos medios"
            else:
                perfil = "Segmento mixto"
                
            if frecuencia_media > 3:
                perfil += " - Compradores frecuentes"
            elif frecuencia_media < 2:
                perfil += " - Compradores ocasionales"
            else:
                perfil += " - Compradores moderados"
                
            print(f"     Perfil: {perfil}")
            
    def predecir_cluster(self, nuevos_clientes):
        """Predice el cluster para nuevos clientes"""
        if self.modelo is None:
            print("❌ Modelo no entrenado")
            return None
        
        # Escalar nuevos datos
        nuevos_escalados = self.scaler.transform(nuevos_clientes)
        
        # Predecir clusters
        clusters_predichos = self.modelo.predict(nuevos_escalados)
        
        print("🔮 Predicciones para nuevos clientes:")
        for i, (cliente, cluster) in enumerate(zip(nuevos_clientes, clusters_predichos)):
            print(f"   Cliente {i+1}: Cluster {cluster}")
            print(f"     Características: Edad={cliente[0]:.1f}, Ingresos=${cliente[1]:,.0f}, Frecuencia={cliente[2]:.1f}")
        
        return clusters_predichos

def demo_segmentador_clientes():
    """Demostración completa del segmentador de clientes"""
    print("=" * 60)
    print("👥 SEGMENTADOR DE CLIENTES (APRENDIZAJE NO SUPERVISADO)")
    print("=" * 60)
    
    # Crear instancia
    segmentador = SegmentadorClientes()
    
    # Generar datos
    segmentador.generar_datos_clientes(n_muestras=300)
    segmentador.visualizar_datos_originales()
    
    # Preprocesar
    segmentador.preprocesar_datos()
    
    # Encontrar número óptimo de clusters
    segmentador.encontrar_numero_optimo_clusters(max_k=8)
    
    # Construir y ajustar modelo
    segmentador.construir_y_ajustar_modelo()
    
    # Visualizar clusters
    segmentador.visualizar_clusters()
    
    # Analizar perfiles
    segmentador.analizar_perfiles_clusters()
    
    # Predecir con nuevos clientes
    nuevos_clientes = np.array([
        [28, 35000, 4],    # Joven, ingresos bajos, compras frecuentes
        [55, 95000, 1],    # Adulto mayor, ingresos altos, compras ocasionales
        [38, 60000, 2]     # Adulto, ingresos medios, compras moderadas
    ])
    
    segmentador.predecir_cluster(nuevos_clientes)
    
    print("\n" + "=" * 60)
    print("🎉 DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)

if __name__ == "__main__":
    demo_segmentador_clientes()
