# -*- coding: utf-8 -*-
"""
Proyecto 2: Automatización en Salud
Entrenamiento de Transformers para análisis de informes médicos
"""

import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MedicalReportAnalyzer:
    """Clase para analizar informes médicos usando Transformers"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.history = None
        
    def load_data(self, csv_path="data/processed/reports.csv"):
        """Cargar datos de informes médicos"""
        
        # Datos simulados si no existe el archivo
        if not os.path.exists(csv_path):
            print("📝 Generando datos simulados de informes médicos...")
            data = {
                'text': [
                    "El paciente presenta opacidades en lóbulo inferior derecho con infiltrados.",
                    "Radiografía de tórax sin hallazgos patológicos significativos.",
                    "Infiltrados bilaterales sugestivos de neumonía en base pulmonar.",
                    "Neumotórax derecho sin signos evidentes de neumonía.",
                    "Cardiomegalia moderada con congestión pulmonar leve.",
                    "Campos pulmonares claros, sin evidencia de proceso infeccioso.",
                    "Consolidación en lóbulo superior izquierdo compatible con neumonía.",
                    "Patrón intersticial bilateral, posible fibrosis pulmonar.",
                    "Radiografía normal para la edad del paciente.",
                    "Atelectasia segmentaria en base derecha, resto normal."
                ] * 50,  # Repetir para tener más datos
                'label': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0] * 50  # 1: neumonía, 0: normal
            }
            df = pd.DataFrame(data)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
        else:
            df = pd.read_csv(csv_path)
            
        print(f"📊 Dataset cargado: {len(df)} informes")
        print(f"   • Con neumonía: {df['label'].sum()}")
        print(f"   • Normales: {len(df) - df['label'].sum()}")
        
        return df
    
    def preprocess_data(self, df, max_length=128):
        """Preprocesar textos para el modelo Transformer"""
        
        # Tokenizar textos
        inputs = self.tokenizer(
            df['text'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="tf"
        )
        
        # Convertir etiquetas a tensor
        labels = tf.convert_to_tensor(df['label'].values)
        
        return inputs, labels
    
    def split_data(self, inputs, labels, test_size=0.2):
        """Dividir datos en entrenamiento y prueba"""
        
        # Dividir índices
        indices = np.arange(len(labels))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Crear datasets de entrenamiento y prueba
        train_inputs = {
            'input_ids': tf.gather(inputs['input_ids'], train_indices),
            'attention_mask': tf.gather(inputs['attention_mask'], train_indices)
        }
        train_labels = tf.gather(labels, train_indices)
        
        test_inputs = {
            'input_ids': tf.gather(inputs['input_ids'], test_indices),
            'attention_mask': tf.gather(inputs['attention_mask'], test_indices)
        }
        test_labels = tf.gather(labels, test_indices)
        
        return (train_inputs, train_labels), (test_inputs, test_labels)
    
    def build_model(self, num_labels=2):
        """Construir y compilar el modelo Transformer"""
        
        model = TFDistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, train_inputs, train_labels, test_inputs, test_labels, epochs=3):
        """Entrenar el modelo Transformer"""
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1),
            tf.keras.callbacks.ModelCheckpoint(
                'models/transformer_best_model.h5', save_best_only=True
            )
        ]
        
        # Entrenamiento
        history = self.model.fit(
            train_inputs,
            train_labels,
            validation_data=(test_inputs, test_labels),
            epochs=epochs,
            batch_size=8,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        return history
    
    def evaluate_model(self, test_inputs, test_labels):
        """Evaluar el modelo y generar métricas"""
        
        # Predicciones
        predictions = self.model.predict(test_inputs)
        predicted_labels = np.argmax(predictions.logits, axis=1)
        
        # Métricas
        report = classification_report(test_labels.numpy(), predicted_labels, output_dict=True)
        cm = confusion_matrix(test_labels.numpy(), predicted_labels)
        
        print("\n📊 Resultados de Evaluación:")
        print(f"   • Accuracy: {report['accuracy']:.4f}")
        print(f"   • Precision (Normal): {report['0']['precision']:.4f}")
        print(f"   • Recall (Normal): {report['0']['recall']:.4f}")
        print(f"   • Precision (Neumonía): {report['1']['precision']:.4f}")
        print(f"   • Recall (Neumonía): {report['1']['recall']:.4f}")
        
        return report, cm
    
    def plot_results(self, cm):
        """Visualizar matriz de confusión y curvas de aprendizaje"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matriz de Confusión')
        axes[0].set_xlabel('Predicho')
        axes[0].set_ylabel('Real')
        axes[0].set_xticklabels(['Normal', 'Neumonía'])
        axes[0].set_yticklabels(['Normal', 'Neumonía'])
        
        # Curvas de aprendizaje
        if self.history:
            axes[1].plot(self.history.history['accuracy'], label='Train')
            axes[1].plot(self.history.history['val_accuracy'], label='Validation')
            axes[1].set_title('Curvas de Aprendizaje')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('notebooks/transformer_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def predict_report(self, text):
        """Predecir diagnóstico para un nuevo informe"""
        
        if not self.model:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train_model primero.")
        
        # Tokenizar texto
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="tf"
        )
        
        # Predecir
        outputs = self.model(inputs)
        prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]
        confidence = tf.nn.softmax(outputs.logits)[0][prediction].numpy()
        
        diagnosis = "neumonía" if prediction == 1 else "normal"
        
        return {
            'diagnosis': diagnosis,
            'confidence': float(confidence),
            'prediction': int(prediction)
        }
    
    def save_model(self, path='models/transformer_model'):
        """Guardar modelo y tokenizer"""
        
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"💾 Modelo guardado en '{path}'")

def main():
    """Función principal de entrenamiento"""
    
    print("🏥 Iniciando entrenamiento de Transformer para análisis de informes médicos")
    print("=" * 70)
    
    # Crear directorios necesarios
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    # 1. Inicializar analizador
    print("\n🔧 Inicializando analizador médico...")
    analyzer = MedicalReportAnalyzer()
    
    # 2. Cargar datos
    print("\n📁 Cargando datos de informes médicos...")
    df = analyzer.load_data()
    
    # 3. Preprocesar datos
    print("\n🔄 Preprocesando textos...")
    inputs, labels = analyzer.preprocess_data(df)
    (train_inputs, train_labels), (test_inputs, test_labels) = analyzer.split_data(inputs, labels)
    
    print(f"   • Entrenamiento: {len(train_labels)} muestras")
    print(f"   • Prueba: {len(test_labels)} muestras")
    
    # 4. Construir modelo
    print("\n🏗️ Construyendo modelo Transformer...")
    model = analyzer.build_model()
    print(f"   • Modelo: {analyzer.model_name}")
    print(f"   • Parámetros: {model.num_parameters():,}")
    
    # 5. Entrenar modelo
    print("\n🎯 Entrenando modelo...")
    history = analyzer.train_model(train_inputs, train_labels, test_inputs, test_labels, epochs=5)
    
    # 6. Evaluar modelo
    print("\n📈 Evaluando modelo...")
    report, cm = analyzer.evaluate_model(test_inputs, test_labels)
    
    # 7. Visualizar resultados
    print("\n📊 Generando visualizaciones...")
    analyzer.plot_results(cm)
    
    # 8. Guardar modelo
    print("\n💾 Guardando modelo...")
    analyzer.save_model()
    
    # 9. Probar predicciones
    print("\n🧪 Probando predicciones...")
    test_reports = [
        "El paciente presenta infiltrados bilaterales compatibles con neumonía.",
        "Radiografía de tórax normal sin hallazgos patológicos.",
        "Opacidades en lóbulo inferior con signos de consolidación."
    ]
    
    for i, report in enumerate(test_reports, 1):
        result = analyzer.predict_report(report)
        print(f"   Informe {i}: {result['diagnosis']} (confianza: {result['confidence']:.4f})")
    
    # 10. Resumen final
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"📊 Métricas Finales:")
    print(f"   • Accuracy: {report['accuracy']:.4f}")
    print(f"   • Precision (Neumonía): {report['1']['precision']:.4f}")
    print(f"   • Recall (Neumonía): {report['1']['recall']:.4f}")
    print(f"📁 Archivos generados:")
    print(f"   • models/transformer_model/ - Modelo y tokenizer guardados")
    print(f"   • models/transformer_best_model.h5 - Mejor modelo durante entrenamiento")
    print(f"   • notebooks/transformer_results.png - Matriz de confusión y curvas")
    print("=" * 70)

if __name__ == "__main__":
    main()
