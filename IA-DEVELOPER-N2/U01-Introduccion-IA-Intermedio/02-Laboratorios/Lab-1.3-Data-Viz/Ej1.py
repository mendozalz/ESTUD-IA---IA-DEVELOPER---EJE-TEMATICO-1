import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ClasificadorIris:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.datos = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_names = None
        
    def cargar_datos(self):
        """Carga y explora el dataset Iris"""
        print("🌸 Cargando dataset Iris...")
        
        # Cargar dataset
        iris = load_iris()
        self.datos = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.datos['target'] = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        print(f"✅ Dataset cargado: {self.datos.shape}")
        print(f"📊 Características: {self.feature_names}")
        print(f"🎯 Clases: {self.target_names}")
        print(f"\n📈 Estadísticas descriptivas:")
        print(self.datos.describe())
        
        return self.datos
    
    def visualizar_datos(self):
        """Visualiza el dataset Iris"""
        if self.datos is None:
            print("❌ Datos no cargados")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Pair plot
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=self.datos, x='sepal length (cm)', y='sepal width (cm)', 
                        hue='target', palette='viridis', s=60)
        plt.title('Sepal Length vs Sepal Width')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Sepal Width (cm)')
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=self.datos, x='petal length (cm)', y='petal width (cm)', 
                        hue='target', palette='viridis', s=60)
        plt.title('Petal Length vs Petal Width')
        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Petal Width (cm)')
        
        # Distribución de características
        plt.subplot(2, 2, 3)
        self.datos.drop('target', axis=1).boxplot()
        plt.title('Distribución de características')
        plt.xticks(rotation=45)
        
        # Balance de clases
        plt.subplot(2, 2, 4)
        self.datos['target'].value_counts().plot(kind='bar')
        plt.title('Balance de clases')
        plt.xlabel('Clase')
        plt.ylabel('Cantidad')
        plt.xticks(range(3), self.target_names, rotation=0)
        
        plt.tight_layout()
        plt.show()
        
    def preparar_datos(self, test_size=0.2, random_state=42):
        """Prepara los datos para entrenamiento"""
        print("📂 Preparando datos para entrenamiento...")
        
        if self.datos is None:
            print("❌ Datos no cargados")
            return
        
        # Separar características y target
        X = self.datos.drop('target', axis=1)
        y = self.datos['target']
        
        # Dividir en train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Escalar características
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"📊 División de datos:")
        print(f"   Entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"   Prueba: {self.X_test.shape[0]} muestras")
        print(f"   Características: {self.X_train.shape[1]}")
        
    def construir_modelo(self, n_estimators=100, random_state=42):
        """Construye el modelo Random Forest"""
        print("🌳 Construyendo modelo Random Forest...")
        
        self.modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        print("✅ Modelo construido")
        print(f"📊 Parámetros: n_estimators={n_estimators}, max_depth=3")
        
    def entrenar(self):
        """Entrena el modelo"""
        print("🎯 Entrenando modelo...")
        
        if self.modelo is None or self.X_train_scaled is None:
            print("❌ Modelo o datos no preparados")
            return
        
        # Entrenar
        self.modelo.fit(self.X_train_scaled, self.y_train)
        
        print("✅ Entrenamiento completado")
        
        # Importancia de características
        importancia = self.modelo.feature_importances_
        print("\n📊 Importancia de características:")
        for nombre, imp in zip(self.feature_names, importancia):
            print(f"   {nombre}: {imp:.4f}")
            
    def evaluar(self):
        """Evalúa el modelo"""
        print("📊 Evaluando modelo...")
        
        if self.modelo is None or self.X_test_scaled is None:
            print("❌ Modelo o datos no disponibles")
            return
        
        # Predicciones
        y_pred = self.modelo.predict(self.X_test_scaled)
        
        # Métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Reporte de clasificación
        print("\n📋 Reporte de clasificación:")
        print(classification_report(self.y_test, y_pred, target_names=self.target_names))
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.show()
        
        return accuracy
    
    def predecir(self, nuevas_muestras):
        """Predice clases para nuevas muestras"""
        if self.modelo is None:
            print("❌ Modelo no entrenado")
            return None
        
        # Escalar nuevas muestras
        muestras_escaladas = self.scaler.transform(nuevas_muestras)
        
        # Predecir
        predicciones = self.modelo.predict(muestras_escaladas)
        probabilidades = self.modelo.predict_proba(muestras_escaladas)
        
        # Convertir a nombres de clases
        clases_predichas = [self.target_names[p] for p in predicciones]
        
        print("🔮 Predicciones:")
        for i, (muestra, clase, probs) in enumerate(zip(nuevas_muestras, clases_predichas, probabilidades)):
            print(f"   Muestra {i+1}: {clase} (confianza: {max(probs):.3f})")
            print(f"   Características: {muestra}")
        
        return clases_predichas, probabilidades

def demo_clasificador_iris():
    """Demostración completa del clasificador Iris"""
    print("=" * 60)
    print("🌸 CLASIFICADOR DE FLORES IRIS (APRENDIZAJE SUPERVISADO)")
    print("=" * 60)
    
    # Crear instancia
    clasificador = ClasificadorIris()
    
    # Cargar y explorar datos
    clasificador.cargar_datos()
    clasificador.visualizar_datos()
    
    # Preparar datos
    clasificador.preparar_datos(test_size=0.2, random_state=42)
    
    # Construir y entrenar modelo
    clasificador.construir_modelo(n_estimators=100, random_state=42)
    clasificador.entrenar()
    
    # Evaluar modelo
    accuracy = clasificador.evaluar()
    
    # Predecir con nuevas muestras
    nuevas_muestras = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Probablemente setosa
        [6.7, 3.0, 5.2, 2.3],  # Probablemente virginica
        [5.9, 3.0, 4.2, 1.5]   # Probablemente versicolor
    ])
    
    clasificador.predecir(nuevas_muestras)
    
    print("\n" + "=" * 60)
    print("🎉 DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)

if __name__ == "__main__":
    demo_clasificador_iris()
