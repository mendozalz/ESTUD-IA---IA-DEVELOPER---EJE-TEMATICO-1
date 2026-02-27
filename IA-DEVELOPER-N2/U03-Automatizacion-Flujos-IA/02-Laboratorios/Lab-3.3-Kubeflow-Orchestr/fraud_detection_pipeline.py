"""
Fraud Detection Pipeline using Kubeflow
Sector: Finance
Objective: Real-time fraud detection with Kubeflow and TimescaleDB
"""

import kfp
from kfp import dsl
from kfp.components import create_component_from_func, InputPath, OutputPath
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import psycopg2
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@create_component_from_func
def preprocess_financial_data(
    input_path: str,
    output_path: str,
    schema_path: OutputPath(str)
):
    """
    Preprocess financial transaction data for fraud detection
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import json
    
    logger.info(f"Loading data from {input_path}")
    
    # Load transaction data
    df = pd.read_csv(input_path)
    
    # Data preprocessing
    logger.info("Preprocessing financial transaction data...")
    
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Feature engineering
    df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['transaction_day'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['amount_log'] = np.log1p(df['amount'])
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['amount', 'amount_log', 'transaction_hour', 'transaction_day']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Create schema for validation
    schema = {
        "features": numerical_features,
        "target": "is_fraud",
        "preprocessing_steps": [
            "missing_value_imputation",
            "feature_engineering",
            "standardization"
        ]
    }
    
    # Save processed data and schema
    df.to_csv(output_path, index=False)
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    logger.info(f"Preprocessed data saved to {output_path}")
    logger.info(f"Schema saved to {schema_path}")

@create_component_from_func
def train_fraud_model(
    data_path: str,
    model_path: OutputPath(str),
    metrics_path: OutputPath(str)
):
    """
    Train fraud detection model using TensorFlow
    """
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    import json
    
    logger.info("Training fraud detection model...")
    
    # Load preprocessed data
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop(['is_fraud', 'timestamp'], axis=1, errors='ignore')
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build fraud detection model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred_class, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred)
    
    metrics = {
        "classification_report": report,
        "auc_score": float(auc_score),
        "training_history": {
            "loss": history.history['loss'][-1],
            "val_loss": history.history['val_loss'][-1],
            "accuracy": history.history['accuracy'][-1],
            "val_accuracy": history.history['val_accuracy'][-1]
        }
    }
    
    # Save model and metrics
    model.save(model_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Model AUC: {auc_score:.4f}")

@create_component_from_func
def validate_model(
    model_path: InputPath(str),
    test_data_path: str,
    validation_report_path: OutputPath(str)
):
    """
    Validate fraud detection model performance
    """
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix, classification_report
    import json
    
    logger.info("Validating fraud detection model...")
    
    # Load model and test data
    model = tf.keras.models.load_model(model_path)
    df = pd.read_csv(test_data_path)
    
    # Prepare test data
    X_test = df.drop(['is_fraud', 'timestamp'], axis=1, errors='ignore')
    y_test = df['is_fraud']
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    
    # Calculate validation metrics
    cm = confusion_matrix(y_test, y_pred_class)
    report = classification_report(y_test, y_pred_class, output_dict=True)
    
    # Calculate fraud-specific metrics
    tn, fp, fn, tp = cm.ravel()
    fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    validation_report = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "fraud_detection_rate": float(fraud_detection_rate),
        "false_positive_rate": float(false_positive_rate),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }
    
    # Save validation report
    with open(validation_report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"Validation completed. Fraud detection rate: {fraud_detection_rate:.4f}")

@create_component_from_func
def deploy_model_to_kubernetes(
    model_path: InputPath(str),
    deployment_config_path: OutputPath(str)
):
    """
    Deploy fraud detection model to Kubernetes
    """
    import json
    import yaml
    
    logger.info("Creating Kubernetes deployment configuration...")
    
    # Create Kubernetes deployment YAML
    deployment_config = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "fraud-detection-model",
            "labels": {
                "app": "fraud-detection",
                "component": "model"
            }
        },
        "spec": {
            "replicas": 3,
            "selector": {
                "matchLabels": {
                    "app": "fraud-detection"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "fraud-detection"
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "fraud-model",
                        "image": "fraud-detection:latest",
                        "ports": [{"containerPort": 8080}],
                        "resources": {
                            "requests": {
                                "cpu": "500m",
                                "memory": "1Gi"
                            },
                            "limits": {
                                "cpu": "1000m",
                                "memory": "2Gi"
                            }
                        },
                        "env": [
                            {"name": "MODEL_PATH", "value": "/app/model"},
                            {"name": "TIMESCALEDB_HOST", "value": "timescaledb-service"}
                        ]
                    }]
                }
            }
        }
    }
    
    # Create service configuration
    service_config = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "fraud-detection-service"
        },
        "spec": {
            "selector": {
                "app": "fraud-detection"
            },
            "ports": [{
                "port": 80,
                "targetPort": 8080,
                "protocol": "TCP"
            }],
            "type": "ClusterIP"
        }
    }
    
    # Save deployment configuration
    config = {
        "deployment": deployment_config,
        "service": service_config
    }
    
    with open(deployment_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Kubernetes deployment configuration created")

@create_component_from_func
def setup_timescaledb_integration(
    schema_path: InputPath(str),
    db_config_path: OutputPath(str)
):
    """
    Setup TimescaleDB integration for fraud detection
    """
    import json
    import yaml
    
    logger.info("Setting up TimescaleDB integration...")
    
    # Database configuration
    db_config = {
        "database": "fraud_detection_db",
        "host": "timescaledb-service",
        "port": 5432,
        "user": "fraud_user",
        "password": "secure_password"
    }
    
    # Table creation SQL
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS transactions (
        time TIMESTAMPTZ NOT NULL,
        transaction_id VARCHAR(255) PRIMARY KEY,
        amount DECIMAL(15,2) NOT NULL,
        location VARCHAR(100),
        merchant_id VARCHAR(100),
        user_id VARCHAR(100),
        is_fraud BOOLEAN DEFAULT FALSE,
        fraud_probability DECIMAL(5,4),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    -- Create hypertable for time-series data
    SELECT create_hypertable('transactions', 'time', chunk_time_interval => INTERVAL '1 day');
    
    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions (time DESC);
    CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions (user_id);
    CREATE INDEX IF NOT EXISTS idx_transactions_fraud ON transactions (is_fraud) WHERE is_fraud = TRUE;
    """
    
    # Integration configuration
    integration_config = {
        "database": db_config,
        "table_schema": create_table_sql,
        "connection_pool": {
            "min_connections": 5,
            "max_connections": 20,
            "connection_timeout": 30
        },
        "retention_policy": {
            "transactions": "2 years",
            "fraud_alerts": "5 years"
        }
    }
    
    # Save configuration
    with open(db_config_path, 'w') as f:
        yaml.dump(integration_config, f, default_flow_style=False)
    
    logger.info("TimescaleDB integration configuration created")

@dsl.pipeline(
    name="fraud-detection-pipeline",
    description="End-to-end fraud detection pipeline using Kubeflow and TimescaleDB"
)
def fraud_detection_pipeline(
    input_data_path: str = "gs://fraud-detection-data/transactions.csv",
    test_data_path: str = "gs://fraud-detection-data/test_transactions.csv"
):
    """
    Kubeflow pipeline for fraud detection
    """
    
    # Preprocess financial data
    preprocess_task = preprocess_financial_data(
        input_path=input_data_path
    )
    
    # Train fraud detection model
    train_task = train_fraud_model(
        data_path=preprocess_task.outputs['output_path']
    )
    
    # Validate model
    validate_task = validate_model(
        model_path=train_task.outputs['model_path'],
        test_data_path=test_data_path
    )
    
    # Setup TimescaleDB integration
    setup_db_task = setup_timescaledb_integration(
        schema_path=preprocess_task.outputs['schema_path']
    )
    
    # Deploy model to Kubernetes
    deploy_task = deploy_model_to_kubernetes(
        model_path=train_task.outputs['model_path']
    )
    
    # Define dependencies
    train_task.after(preprocess_task)
    validate_task.after(train_task)
    deploy_task.after(validate_task)
    setup_db_task.after(preprocess_task)

# Compile the pipeline
if __name__ == "__main__":
    from kfp import compiler
    
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_detection_pipeline.yaml"
    )
    
    logger.info("Fraud detection pipeline compiled successfully!")
