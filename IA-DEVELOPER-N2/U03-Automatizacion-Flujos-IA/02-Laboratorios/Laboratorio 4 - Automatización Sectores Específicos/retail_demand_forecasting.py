"""
Retail Demand Forecasting Pipeline using Kubeflow
Sector: Retail
Objective: Demand forecasting and inventory optimization with Kubeflow and InfluxDB
"""

import kfp
from kfp import dsl
from kfp.components import create_component_from_func, InputPath, OutputPath
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@create_component_from_func
def preprocess_sales_data(
    input_path: str,
    output_path: str,
    config_path: OutputPath(str)
):
    """
    Preprocess retail sales data for demand forecasting
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import json
    
    logger.info(f"Loading sales data from {input_path}")
    
    # Load sales data
    df = pd.read_csv(input_path)
    
    # Data preprocessing
    logger.info("Preprocessing retail sales data...")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Feature engineering
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['date'].dt.date.astype('datetime64').isin([
        '2024-01-01', '2024-12-25', '2024-07-04'  # Sample holidays
    ]).astype(int)
    
    # Create lag features
    for lag in [1, 7, 14, 30]:  # 1 day, 1 week, 2 weeks, 1 month
        df[f'sales_lag_{lag}'] = df.groupby('product_id')['sales'].shift(lag)
    
    # Create rolling features
    for window in [7, 14, 30]:
        df[f'sales_rolling_mean_{window}'] = df.groupby('product_id')['sales'].rolling(window).mean().reset_index(0, drop=True)
        df[f'sales_rolling_std_{window}'] = df.groupby('product_id')['sales'].rolling(window).std().reset_index(0, drop=True)
    
    # Handle missing values
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = ['sales', 'price', 'promotion'] + [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Create time series sequences for LSTM
    def create_sequences(data, sequence_length=30):
        sequences = []
        targets = []
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id].sort_values('date')
            if len(product_data) > sequence_length:
                for i in range(len(product_data) - sequence_length):
                    sequences.append(product_data.iloc[i:i+sequence_length][numerical_features].values)
                    targets.append(product_data.iloc[i+sequence_length]['sales'])
        return np.array(sequences), np.array(targets)
    
    X, y = create_sequences(df)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx]
    
    # Save processed data
    processed_data = {
        'X_train': X_train.tolist(),
        'X_test': X_test.tolist(),
        'y_train': y_train.tolist(),
        'y_test': y_test.tolist(),
        'feature_names': numerical_features,
        'scaler': scaler
    }
    
    # Configuration for model
    config = {
        "sequence_length": 30,
        "n_features": len(numerical_features),
        "n_products": len(df['product_id'].unique()),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "preprocessing_steps": [
            "date_conversion",
            "feature_engineering",
            "lag_features",
            "rolling_features",
            "normalization",
            "sequence_creation"
        ]
    }
    
    # Save data and config
    with open(output_path, 'w') as f:
        json.dump(processed_data, f)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Preprocessed data saved to {output_path}")
    logger.info(f"Configuration saved to {config_path}")

@create_component_from_func
def train_demand_model(
    data_path: InputPath(str),
    config_path: InputPath(str),
    model_path: OutputPath(str),
    metrics_path: OutputPath(str)
):
    """
    Train demand forecasting model using LSTM
    """
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import json
    
    logger.info("Training demand forecasting model...")
    
    # Load data and config
    with open(data_path, 'r') as f:
        data = json.load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    X_train = np.array(data['X_train'])
    X_test = np.array(data['X_test'])
    y_train = np.array(data['y_train'])
    y_test = np.array(data['y_test'])
    
    # Build LSTM model for demand forecasting
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(config['sequence_length'], config['n_features'])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    logger.info("Starting LSTM training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='/tmp/best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ],
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate business metrics
    def calculate_business_metrics(actual, predicted):
        # Forecast accuracy within 10% tolerance
        tolerance = 0.1
        accurate_forecasts = np.abs((predicted - actual) / actual) < tolerance
        accuracy = np.mean(accurate_forecasts)
        
        # Stock optimization potential
        overstock_cost = np.sum(np.maximum(predicted - actual, 0)) * 0.5  # 50% of overstock value
        understock_cost = np.sum(np.maximum(actual - predicted, 0)) * 2.0  # 200% of lost sales
        total_cost = overstock_cost + understock_cost
        
        return {
            'forecast_accuracy': float(accuracy),
            'overstock_cost': float(overstock_cost),
            'understock_cost': float(understock_cost),
            'total_inventory_cost': float(total_cost)
        }
    
    business_metrics = calculate_business_metrics(y_test, y_pred)
    
    # Training metrics
    metrics = {
        "model_performance": {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse)
        },
        "business_metrics": business_metrics,
        "training_history": {
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "final_mae": float(history.history['mae'][-1]),
            "final_val_mae": float(history.history['val_mae'][-1])
        },
        "model_info": {
            "total_parameters": model.count_params(),
            "sequence_length": config['sequence_length'],
            "n_features": config['n_features']
        }
    }
    
    # Load best model and save
    best_model = tf.keras.models.load_model('/tmp/best_model.h5')
    best_model.save(model_path)
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Model MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    logger.info(f"Forecast Accuracy: {business_metrics['forecast_accuracy']:.4f}")

@create_component_from_func
def setup_influxdb_integration(
    config_path: InputPath(str),
    influx_config_path: OutputPath(str)
):
    """
    Setup InfluxDB integration for retail demand forecasting
    """
    import json
    import yaml
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    from datetime import datetime
    
    logger.info("Setting up InfluxDB integration for retail...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # InfluxDB configuration
    influx_config = {
        "url": "http://influxdb-service:8086",
        "token": "retail-token",
        "org": "retail-org",
        "bucket": "demand_forecasting"
    }
    
    # Create bucket configuration
    bucket_config = {
        "org_id": "retail-org",
        "name": "demand_forecasting",
        "retention_rules": [{
            "type": "expire",
            "every_seconds": 31536000  # 1 year
        }]
    }
    
    # Sample data points for setup
    sample_points = [
        Point("sales_data")\
            .tag("product_id", "prod_001")\
            .tag("store_id", "store_01")\
            .field("sales", 150)\
            .field("price", 29.99)\
            .field("promotion", 1)\
            .time(datetime.utcnow()),
        
        Point("demand_forecast")\
            .tag("product_id", "prod_001")\
            .tag("model_version", "v1.0")\
            .field("predicted_demand", 165)\
            .field("confidence_interval_lower", 140)\
            .field("confidence_interval_upper", 190)\
            .field("forecast_horizon_days", 7)\
            .time(datetime.utcnow()),
        
        Point("inventory_metrics")\
            .tag("product_id", "prod_001")\
            .tag("store_id", "store_01")\
            .field("current_stock", 200)\
            .field("reorder_point", 100)\
            .field("safety_stock", 50)\
            .field("days_of_supply", 45)\
            .time(datetime.utcnow())
    ]
    
    # Integration configuration
    integration_config = {
        "influxdb": influx_config,
        "bucket": bucket_config,
        "data_schema": {
            "sales_data": {
                "tags": ["product_id", "store_id", "region"],
                "fields": ["sales", "price", "promotion", "customer_count"],
                "timestamp": "date"
            },
            "demand_forecast": {
                "tags": ["product_id", "model_version", "forecast_type"],
                "fields": ["predicted_demand", "confidence_interval_lower", "confidence_interval_upper", "forecast_horizon_days"],
                "timestamp": "forecast_date"
            },
            "inventory_metrics": {
                "tags": ["product_id", "store_id", "warehouse"],
                "fields": ["current_stock", "reorder_point", "safety_stock", "days_of_supply", "turnover_rate"],
                "timestamp": "metric_date"
            }
        },
        "sample_data": [str(point) for point in sample_points],
        "retention_policies": {
            "sales_data": "2 years",
            "demand_forecast": "1 year",
            "inventory_metrics": "6 months"
        }
    }
    
    # Save configuration
    with open(influx_config_path, 'w') as f:
        yaml.dump(integration_config, f, default_flow_style=False)
    
    logger.info("InfluxDB integration configuration created")

@create_component_from_func
def create_alerting_system(
    model_path: InputPath(str),
    alert_config_path: OutputPath(str)
):
    """
    Create alerting system for inventory management
    """
    import json
    import yaml
    import tensorflow as tf
    from datetime import datetime, timedelta
    
    logger.info("Creating alerting system for inventory management...")
    
    # Load model to get prediction capabilities
    model = tf.keras.models.load_model(model_path)
    
    # Alerting configuration
    alert_config = {
        "alert_rules": [
            {
                "name": "low_stock_alert",
                "description": "Alert when stock falls below reorder point",
                "condition": "current_stock < reorder_point",
                "severity": "high",
                "notification_channels": ["email", "slack", "webhook"],
                "thresholds": {
                    "critical": 0.5,  # 50% below reorder point
                    "warning": 0.8   # 80% of reorder point
                }
            },
            {
                "name": "overstock_alert",
                "description": "Alert when stock exceeds maximum capacity",
                "condition": "current_stock > max_capacity",
                "severity": "medium",
                "notification_channels": ["email"],
                "thresholds": {
                    "warning": 0.9,  # 90% of max capacity
                    "critical": 1.0  # At max capacity
                }
            },
            {
                "name": "demand_spike_alert",
                "description": "Alert when demand increases significantly",
                "condition": "predicted_demand > (historical_average * 1.5)",
                "severity": "high",
                "notification_channels": ["slack", "webhook"],
                "thresholds": {
                    "warning": 1.3,   # 30% increase
                    "critical": 1.5   # 50% increase
                }
            },
            {
                "name": "forecast_accuracy_alert",
                "description": "Alert when forecast accuracy drops",
                "condition": "forecast_accuracy < 0.8",
                "severity": "medium",
                "notification_channels": ["email"],
                "thresholds": {
                    "warning": 0.85,  # 85% accuracy
                    "critical": 0.8   # 80% accuracy
                }
            }
        ],
        "notification_config": {
            "email": {
                "smtp_server": "smtp.company.com",
                "smtp_port": 587,
                "recipients": ["inventory-manager@company.com", "procurement@company.com"]
            },
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/...",
                "channel": "#inventory-alerts"
            },
            "webhook": {
                "url": "https://api.company.com/inventory-alerts",
                "auth_token": "secure-webhook-token"
            }
        },
        "escalation_policy": {
            "level_1": {
                "delay_minutes": 0,
                "channels": ["slack", "webhook"]
            },
            "level_2": {
                "delay_minutes": 15,
                "channels": ["email", "slack", "webhook"]
            },
            "level_3": {
                "delay_minutes": 60,
                "channels": ["email", "slack", "webhook", "sms"]
            }
        },
        "model_info": {
            "model_path": model_path,
            "prediction_horizon_days": 7,
            "confidence_level": 0.95,
            "retraining_frequency": "weekly"
        }
    }
    
    # Save alerting configuration
    with open(alert_config_path, 'w') as f:
        yaml.dump(alert_config, f, default_flow_style=False)
    
    logger.info("Alerting system configuration created")

@create_component_from_func
def deploy_retail_api(
    model_path: InputPath(str),
    influx_config_path: InputPath(str),
    alert_config_path: InputPath(str),
    deployment_config_path: OutputPath(str)
):
    """
    Deploy retail demand forecasting API
    """
    import json
    import yaml
    
    logger.info("Creating deployment configuration for retail API...")
    
    # Load configurations
    with open(influx_config_path, 'r') as f:
        influx_config = yaml.safe_load(f)
    with open(alert_config_path, 'r') as f:
        alert_config = yaml.safe_load(f)
    
    # Create FastAPI application code
    api_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta
import json
import logging

app = FastAPI(title="Retail Demand Forecasting API", version="1.0.0")

# Load model
model = tf.keras.models.load_model("/app/model")
logger = logging.getLogger(__name__)

class DemandRequest(BaseModel):
    product_id: str
    store_id: str
    historical_days: int = 30

class DemandResponse(BaseModel):
    product_id: str
    store_id: str
    predicted_demand: float
    confidence_interval: dict
    forecast_horizon_days: int
    model_version: str

class InventoryStatus(BaseModel):
    product_id: str
    store_id: str
    current_stock: int
    reorder_point: int
    days_of_supply: int
    alert_status: str

@app.post("/predict", response_model=DemandResponse)
async def predict_demand(request: DemandRequest):
    """Predict demand for a product at a specific store"""
    try:
        # Get historical data from InfluxDB
        # ... (implementation would query InfluxDB)
        
        # Make prediction
        # ... (implementation would prepare data and predict)
        
        predicted_demand = 150.0  # Sample prediction
        confidence_interval = {"lower": 130.0, "upper": 170.0}
        
        return DemandResponse(
            product_id=request.product_id,
            store_id=request.store_id,
            predicted_demand=predicted_demand,
            confidence_interval=confidence_interval,
            forecast_horizon_days=7,
            model_version="v1.0"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/inventory/{product_id}/{store_id}", response_model=InventoryStatus)
async def get_inventory_status(product_id: str, store_id: str):
    """Get current inventory status and alerts"""
    try:
        # Get current inventory from InfluxDB
        # ... (implementation would query InfluxDB)
        
        current_stock = 200
        reorder_point = 100
        days_of_supply = 45
        
        # Determine alert status
        if current_stock < reorder_point * 0.5:
            alert_status = "critical"
        elif current_stock < reorder_point * 0.8:
            alert_status = "warning"
        else:
            alert_status = "normal"
        
        return InventoryStatus(
            product_id=product_id,
            store_id=store_id,
            current_stock=current_stock,
            reorder_point=reorder_point,
            days_of_supply=days_of_supply,
            alert_status=alert_status
        )
    except Exception as e:
        logger.error(f"Inventory status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get inventory status")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
'''
    
    # Kubernetes deployment configuration
    deployment_config = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "retail-demand-forecasting",
            "labels": {
                "app": "retail-demand",
                "component": "api"
            }
        },
        "spec": {
            "replicas": 3,
            "selector": {
                "matchLabels": {
                    "app": "retail-demand"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "retail-demand"
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "demand-api",
                        "image": "retail-demand-forecasting:latest",
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
                            {"name": "INFLUXDB_URL", "value": influx_config["influxdb"]["url"]},
                            {"name": "INFLUXDB_TOKEN", "value": influx_config["influxdb"]["token"]},
                            {"name": "INFLUXDB_ORG", "value": influx_config["influxdb"]["org"]},
                            {"name": "INFLUXDB_BUCKET", "value": influx_config["influxdb"]["bucket"]}
                        ],
                        "volumeMounts": [{
                            "name": "model-volume",
                            "mountPath": "/app/model"
                        }]
                    }],
                    "volumes": [{
                        "name": "model-volume",
                        "persistentVolumeClaim": {
                            "claimName": "model-pvc"
                        }
                    }]
                }
            }
        }
    }
    
    # Service configuration
    service_config = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "retail-demand-service"
        },
        "spec": {
            "selector": {
                "app": "retail-demand"
            },
            "ports": [{
                "port": 80,
                "targetPort": 8080,
                "protocol": "TCP"
            }],
            "type": "ClusterIP"
        }
    }
    
    # HPA configuration
    hpa_config = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": "retail-demand-hpa"
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": "retail-demand-forecasting"
            },
            "minReplicas": 3,
            "maxReplicas": 10,
            "metrics": [{
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": 70
                    }
                }
            }]
        }
    }
    
    # Save deployment configuration
    config = {
        "api_code": api_code,
        "deployment": deployment_config,
        "service": service_config,
        "hpa": hpa_config
    }
    
    with open(deployment_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Retail API deployment configuration created")

@dsl.pipeline(
    name="retail-demand-forecasting-pipeline",
    description="End-to-end retail demand forecasting pipeline using Kubeflow and InfluxDB"
)
def retail_demand_forecasting_pipeline(
    input_data_path: str = "gs://retail-data/sales_data.csv"
):
    """
    Kubeflow pipeline for retail demand forecasting
    """
    
    # Preprocess sales data
    preprocess_task = preprocess_sales_data(
        input_path=input_data_path
    )
    
    # Train demand forecasting model
    train_task = train_demand_model(
        data_path=preprocess_task.outputs['output_path'],
        config_path=preprocess_task.outputs['config_path']
    )
    
    # Setup InfluxDB integration
    setup_db_task = setup_influxdb_integration(
        config_path=preprocess_task.outputs['config_path']
    )
    
    # Create alerting system
    alert_task = create_alerting_system(
        model_path=train_task.outputs['model_path']
    )
    
    # Deploy retail API
    deploy_task = deploy_retail_api(
        model_path=train_task.outputs['model_path'],
        influx_config_path=setup_db_task.outputs['influx_config_path'],
        alert_config_path=alert_task.outputs['alert_config_path']
    )
    
    # Define dependencies
    train_task.after(preprocess_task)
    setup_db_task.after(preprocess_task)
    alert_task.after(train_task)
    deploy_task.after([train_task, setup_db_task, alert_task])

# Compile the pipeline
if __name__ == "__main__":
    from kfp import compiler
    
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=retail_demand_forecasting_pipeline,
        package_path="retail_demand_forecasting_pipeline.yaml"
    )
    
    logger.info("Retail demand forecasting pipeline compiled successfully!")
