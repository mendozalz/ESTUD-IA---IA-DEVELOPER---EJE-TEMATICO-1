"""
Laboratorio 2 - Sistema CI-CD Modelos ML
Pipeline completo de CI/CD para modelos de machine learning
"""

import os
import sys
import json
import yaml
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import docker
import kubernetes
import mlflow
import mlflow.tensorflow
import requests
from github import Github
from pydantic import BaseModel, Field

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CICDConfig(BaseModel):
    """
    Configuración del pipeline CI/CD
    """
    
    project_name: str = Field(..., description="Nombre del proyecto")
    github_repo: str = Field(..., description="Repositorio GitHub")
    docker_registry: str = Field(..., description="Registry de Docker")
    kubernetes_namespace: str = Field(..., description="Namespace de Kubernetes")
    mlflow_tracking_uri: str = Field(..., description="URI de MLflow")
    model_name: str = Field(..., description="Nombre del modelo")
    
    class Config:
        extra = "forbid"

class ModelRegistry:
    """
    Gestor de registro de modelos con MLflow
    """
    
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(self, model_path: str, model_name: str, version: str):
        """
        Registra un nuevo modelo en MLflow
        """
        logger.info(f"Registrando modelo {model_name} versión {version}")
        
        with mlflow.start_run(run_name=f"register_{model_name}_{version}"):
            mlflow.log_artifact(model_path, artifact_path="model")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            # Agregar tags y metadata
            self.client.set_model_version_tag(
                name=model_name,
                version=registered_model.version,
                key="version",
                value=version
            )
            
            self.client.set_model_version_tag(
                name=model_name,
                version=registered_model.version,
                key="registered_at",
                value=datetime.now().isoformat()
            )
            
            logger.info(f"Modelo registrado exitosamente: {model_name}:{registered_model.version}")
            return registered_model
    
    def promote_model(self, model_name: str, version: str, stage: str = "Production"):
        """
        Promueve un modelo a un stage específico
        """
        logger.info(f"Promoviendo modelo {model_name}:{version} a stage {stage}")
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        
        logger.info(f"Modelo promovido exitosamente a {stage}")
    
    def get_model_info(self, model_name: str, stage: str = "Production"):
        """
        Obtiene información del modelo en un stage específico
        """
        model_versions = self.client.get_latest_versions(model_name, stages=[stage])
        
        if not model_versions:
            return None
        
        latest_version = model_versions[0]
        return {
            'name': latest_version.name,
            'version': latest_version.version,
            'stage': latest_version.current_stage,
            'creation_timestamp': latest_version.creation_timestamp,
            'run_id': latest_version.run_id
        }

class DockerBuilder:
    """
    Constructor de imágenes Docker para modelos ML
    """
    
    def __init__(self, registry: str):
        self.registry = registry
        self.client = docker.from_env()
    
    def build_image(self, dockerfile_path: str, context_path: str, image_tag: str):
        """
        Construye imagen Docker
        """
        logger.info(f"Construyendo imagen Docker: {image_tag}")
        
        try:
            image, build_logs = self.client.images.build(
                path=context_path,
                dockerfile=dockerfile_path,
                tag=image_tag,
                rm=True,
                forcerm=True
            )
            
            logger.info(f"Imagen construida exitosamente: {image.id}")
            return image
            
        except docker.errors.BuildError as e:
            logger.error(f"Error construyendo imagen: {str(e)}")
            raise
    
    def push_image(self, image_tag: str):
        """
        Push imagen al registry
        """
        logger.info(f"Haciendo push de imagen: {image_tag}")
        
        try:
            push_logs = self.client.images.push(image_tag, stream=True, decode=True)
            
            for log_line in push_logs:
                if 'status' in log_line:
                    logger.info(f"Push status: {log_line['status']}")
                if 'error' in log_line:
                    logger.error(f"Push error: {log_line['error']}")
            
            logger.info(f"Imagen push exitosamente: {image_tag}")
            
        except docker.errors.APIError as e:
            logger.error(f"Error haciendo push: {str(e)}")
            raise

class KubernetesDeployer:
    """
    Despliegue en Kubernetes
    """
    
    def __init__(self, namespace: str):
        kubernetes.config.load_kube_config()
        self.v1 = kubernetes.client.CoreV1Api()
        self.apps_v1 = kubernetes.client.AppsV1Api()
        self.namespace = namespace
    
    def deploy_model(self, deployment_config: Dict):
        """
        Despliega modelo en Kubernetes
        """
        deployment_name = deployment_config['metadata']['name']
        logger.info(f"Desplegando en Kubernetes: {deployment_name}")
        
        try:
            # Crear o actualizar deployment
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment_config
            )
            
            # Esperar a que el deployment esté listo
            self.wait_for_deployment_ready(deployment_name)
            
            logger.info(f"Deployment exitoso: {deployment_name}")
            
        except kubernetes.client.ApiException as e:
            logger.error(f"Error en deployment: {str(e)}")
            raise
    
    def wait_for_deployment_ready(self, deployment_name: str, timeout: int = 300):
        """
        Espera a que el deployment esté listo
        """
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            if (deployment.status.ready_replicas == 
                deployment.spec.replicas and 
                deployment.status.ready_replicas > 0):
                logger.info(f"Deployment {deployment_name} está listo")
                return True
            
            logger.info(f"Esperando deployment {deployment_name}...")
            time.sleep(10)
        
        raise TimeoutError(f"Deployment {deployment_name} no está listo después de {timeout} segundos")

class GitHubIntegration:
    """
    Integración con GitHub para CI/CD
    """
    
    def __init__(self, token: str):
        self.github = Github(token)
    
    def create_status_update(self, repo_name: str, commit_sha: str, 
                          status: str, description: str, context: str):
        """
        Crea actualización de status en GitHub
        """
        logger.info(f"Actualizando status en GitHub: {status}")
        
        repo = self.github.get_repo(repo_name)
        commit = repo.get_commit(commit_sha)
        
        commit.create_status(
            state=status,
            description=description,
            context=context
        )
        
        logger.info(f"Status actualizado: {status} - {description}")
    
    def trigger_workflow(self, repo_name: str, workflow_name: str, 
                      inputs: Optional[Dict] = None):
        """
        Dispara workflow de GitHub Actions
        """
        logger.info(f"Disparando workflow: {workflow_name}")
        
        repo = self.github.get_repo(repo_name)
        workflow = repo.get_workflow(workflow_name)
        
        workflow.create_dispatch(inputs=inputs or {})
        
        logger.info(f"Workflow {workflow_name} disparado exitosamente")

class CICDPipeline:
    """
    Pipeline principal de CI/CD para modelos ML
    """
    
    def __init__(self, config: CICDConfig):
        self.config = config
        self.model_registry = ModelRegistry(config.mlflow_tracking_uri)
        self.docker_builder = DockerBuilder(config.docker_registry)
        self.k8s_deployer = KubernetesDeployer(config.kubernetes_namespace)
        self.github_integration = GitHubIntegration(os.getenv('GITHUB_TOKEN'))
    
    def run_ci_pipeline(self, commit_sha: str):
        """
        Ejecuta pipeline de CI
        """
        logger.info("Iniciando pipeline de CI")
        
        try:
            # 1. Code Quality Checks
            self.github_integration.create_status_update(
                self.config.github_repo, commit_sha, 
                'pending', 'Running code quality checks', 'ci/code-quality'
            )
            
            if self.run_code_quality_checks():
                self.github_integration.create_status_update(
                    self.config.github_repo, commit_sha, 
                    'success', 'Code quality checks passed', 'ci/code-quality'
                )
            else:
                self.github_integration.create_status_update(
                    self.config.github_repo, commit_sha, 
                    'failure', 'Code quality checks failed', 'ci/code-quality'
                )
                return False
            
            # 2. Unit Tests
            self.github_integration.create_status_update(
                self.config.github_repo, commit_sha, 
                'pending', 'Running unit tests', 'ci/tests'
            )
            
            if self.run_unit_tests():
                self.github_integration.create_status_update(
                    self.config.github_repo, commit_sha, 
                    'success', 'Unit tests passed', 'ci/tests'
                )
            else:
                self.github_integration.create_status_update(
                    self.config.github_repo, commit_sha, 
                    'failure', 'Unit tests failed', 'ci/tests'
                )
                return False
            
            # 3. Model Training
            self.github_integration.create_status_update(
                self.config.github_repo, commit_sha, 
                'pending', 'Training model', 'ci/training'
            )
            
            model_info = self.train_model()
            
            if model_info:
                self.github_integration.create_status_update(
                    self.config.github_repo, commit_sha, 
                    'success', f'Model trained: {model_info["version"]}', 'ci/training'
                )
            else:
                self.github_integration.create_status_update(
                    self.config.github_repo, commit_sha, 
                    'failure', 'Model training failed', 'ci/training'
                )
                return False
            
            logger.info("Pipeline de CI completado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error en pipeline CI: {str(e)}")
            self.github_integration.create_status_update(
                self.config.github_repo, commit_sha, 
                'error', f'CI pipeline error: {str(e)}', 'ci/pipeline'
            )
            return False
    
    def run_cd_pipeline(self, model_version: str):
        """
        Ejecuta pipeline de CD
        """
        logger.info("Iniciando pipeline de CD")
        
        try:
            # 1. Build Docker Image
            image_tag = f"{self.config.docker_registry}/{self.config.project_name}:{model_version}"
            
            self.docker_builder.build_image(
                dockerfile_path="Dockerfile",
                context_path=".",
                image_tag=image_tag
            )
            
            # 2. Push Image
            self.docker_builder.push_image(image_tag)
            
            # 3. Deploy to Kubernetes
            deployment_config = self.create_deployment_config(image_tag, model_version)
            self.k8s_deployer.deploy_model(deployment_config)
            
            # 4. Update Model Registry
            self.model_registry.promote_model(
                self.config.model_name, 
                model_version, 
                "Production"
            )
            
            # 5. Health Check
            if self.health_check():
                logger.info("Pipeline de CD completado exitosamente")
                return True
            else:
                logger.error("Health check failed después del deployment")
                return False
                
        except Exception as e:
            logger.error(f"Error en pipeline CD: {str(e)}")
            return False
    
    def run_code_quality_checks(self) -> bool:
        """
        Ejecuta checks de calidad de código
        """
        logger.info("Ejecutando checks de calidad de código")
        
        try:
            # Black formatting check
            result = subprocess.run(['black', '--check', '.'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Black formatting issues: {result.stdout}")
                return False
            
            # Flake8 linting
            result = subprocess.run(['flake8', '.'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Flake8 issues: {result.stdout}")
                return False
            
            # MyPy type checking
            result = subprocess.run(['mypy', '.'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"MyPy issues: {result.stdout}")
                return False
            
            logger.info("Checks de calidad de código pasados")
            return True
            
        except Exception as e:
            logger.error(f"Error en checks de calidad: {str(e)}")
            return False
    
    def run_unit_tests(self) -> bool:
        """
        Ejecuta tests unitarios
        """
        logger.info("Ejecutando tests unitarios")
        
        try:
            result = subprocess.run([
                'pytest', 
                '--cov=src',
                '--cov-report=xml',
                '--cov-report=html',
                'tests/'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Tests unitarios pasados")
                return True
            else:
                logger.error(f"Tests unitarios fallidos: {result.stdout}")
                return False
                
        except Exception as e:
            logger.error(f"Error ejecutando tests: {str(e)}")
            return False
    
    def train_model(self) -> Optional[Dict]:
        """
        Entrena el modelo
        """
        logger.info("Entrenando modelo")
        
        try:
            # Simulación de entrenamiento
            # En producción, esto llamaría al script real de entrenamiento
            import uuid
            
            model_version = str(uuid.uuid4())[:8]
            model_path = f"models/{self.config.model_name}_{model_version}.h5"
            
            # Crear archivo de modelo simulado
            os.makedirs("models", exist_ok=True)
            with open(model_path, 'w') as f:
                f.write("# Simulated model file")
            
            # Registrar en MLflow
            registered_model = self.model_registry.register_model(
                model_path, self.config.model_name, model_version
            )
            
            return {
                'version': model_version,
                'path': model_path,
                'registered_version': registered_model.version
            }
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {str(e)}")
            return None
    
    def create_deployment_config(self, image_tag: str, model_version: str) -> Dict:
        """
        Crea configuración de deployment para Kubernetes
        """
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'{self.config.project_name}-deployment',
                'namespace': self.config.kubernetes_namespace,
                'labels': {
                    'app': self.config.project_name,
                    'version': model_version
                }
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'app': self.config.project_name
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config.project_name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.config.project_name,
                            'image': image_tag,
                            'ports': [{
                                'containerPort': 8000
                            }],
                            'env': [
                                {
                                    'name': 'MODEL_VERSION',
                                    'value': model_version
                                },
                                {
                                    'name': 'MLFLOW_TRACKING_URI',
                                    'value': self.config.mlflow_tracking_uri
                                }
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '512Mi',
                                    'cpu': '250m'
                                },
                                'limits': {
                                    'memory': '1Gi',
                                    'cpu': '500m'
                                }
                            }
                        }]
                    }
                }
            }
        }
    
    def health_check(self) -> bool:
        """
        Verifica salud del deployment
        """
        logger.info("Ejecutando health check")
        
        try:
            # En producción, esto haría una petición real al endpoint
            # Aquí simulamos el health check
            import time
            time.sleep(2)  # Simular tiempo de respuesta
            
            logger.info("Health check exitoso")
            return True
            
        except Exception as e:
            logger.error(f"Error en health check: {str(e)}")
            return False

def main():
    """
    Función principal
    """
    logger.info("Iniciando Pipeline CI/CD para Modelos ML")
    
    # Cargar configuración
    config = CICDConfig(
        project_name="ml-model-pipeline",
        github_repo="company/ml-models",
        docker_registry="registry.company.com",
        kubernetes_namespace="ml-production",
        mlflow_tracking_uri="http://mlflow:5000",
        model_name="sales-prediction"
    )
    
    # Crear pipeline
    pipeline = CICDPipeline(config)
    
    # Ejecutar pipeline basado en argumentos
    if len(sys.argv) > 1:
        action = sys.argv[1]
        
        if action == "ci":
            commit_sha = sys.argv[2] if len(sys.argv) > 2 else "latest"
            success = pipeline.run_ci_pipeline(commit_sha)
            sys.exit(0 if success else 1)
            
        elif action == "cd":
            model_version = sys.argv[2] if len(sys.argv) > 2 else "latest"
            success = pipeline.run_cd_pipeline(model_version)
            sys.exit(0 if success else 1)
            
        else:
            logger.error(f"Acción desconocida: {action}")
            sys.exit(1)
    else:
        logger.error("Se requiere especificar acción: ci o cd")
        sys.exit(1)

if __name__ == "__main__":
    main()
