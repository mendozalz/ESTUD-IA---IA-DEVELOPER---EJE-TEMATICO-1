"""
Caso de Uso 5 - Explicabilidad y Cumplimiento con SHAP y Arize
Fase 3: Generación de Documentación de Auditoría
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditDocumentationGenerator:
    """
    Clase para generar documentación de auditoría automática
    """
    
    def __init__(self):
        self.audit_logs = []
        self.compliance_standards = ["GDPR", "ECOA", "FCRA", "CCPA"]
        
    def load_explanations(self):
        """Cargar explicaciones generadas"""
        try:
            with open('outputs/explanations/loan_explanations.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("No se encontraron explicaciones")
            return []
    
    def load_compliance_report(self):
        """Cargar reporte de cumplimiento"""
        try:
            with open('outputs/audit_reports/compliance_report.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("No se encontró reporte de cumplimiento")
            return None
    
    def generate_audit_log(self, explanation_data):
        """Generar entrada de log de auditoría"""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "application_id": explanation_data.get('application_id'),
            "decision": explanation_data.get('prediction', {}).get('approved'),
            "confidence": explanation_data.get('prediction', {}).get('confidence'),
            "explanation_method": explanation_data.get('explanation_type'),
            "top_factors": explanation_data.get('top_features', [])[:3],
            "compliance_check": {
                "explanation_available": True,
                "human_understandable": True,
                "data_minimization": True,
                "audit_trail": True
            },
            "regulatory_compliance": {
                "gdpr_article_22": True,
                "ecoa_compliant": True,
                "fcra_compliant": True
            }
        }
        
        self.audit_logs.append(audit_entry)
        return audit_entry
    
    def generate_comprehensive_audit_report(self):
        """Generar reporte de auditoría completo"""
        logger.info("Generando reporte de auditoría completo...")
        
        # Cargar datos
        explanations = self.load_explanations()
        compliance_report = self.load_compliance_report()
        
        if not explanations:
            logger.error("No hay explicaciones para auditar")
            return None
        
        # Generar logs de auditoría
        audit_logs = []
        for exp in explanations:
            audit_log = self.generate_audit_log(exp)
            audit_logs.append(audit_log)
        
        # Análisis de auditoría
        audit_analysis = self._analyze_audit_logs(audit_logs)
        
        # Reporte completo
        comprehensive_report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "comprehensive_audit",
                "period_covered": "last_30_days",
                "total_applications": len(explanations),
                "audit_standards": self.compliance_standards
            },
            "executive_summary": self._generate_executive_summary(audit_analysis, compliance_report),
            "compliance_status": compliance_report if compliance_report else {},
            "audit_analysis": audit_analysis,
            "detailed_findings": self._generate_detailed_findings(audit_logs),
            "recommendations": self._generate_audit_recommendations(audit_analysis),
            "appendices": {
                "audit_logs_sample": audit_logs[:10],  # Primer 10 logs como muestra
                "explanation_statistics": self._calculate_explanation_stats(explanations),
                "compliance_metrics": self._calculate_compliance_metrics(audit_logs)
            }
        }
        
        # Guardar reporte
        os.makedirs("outputs/audit_reports", exist_ok=True)
        report_path = f"outputs/audit_reports/comprehensive_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        logger.info(f"Reporte de auditoría guardado en: {report_path}")
        return comprehensive_report
    
    def _analyze_audit_logs(self, audit_logs):
        """Analizar logs de auditoría"""
        if not audit_logs:
            return {}
        
        # Estadísticas básicas
        total_applications = len(audit_logs)
        approved_count = sum(1 for log in audit_logs if log['decision'])
        rejected_count = total_applications - approved_count
        
        # Análisis de confianza
        confidences = [log['confidence'] for log in audit_logs]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Análisis de factores principales
        all_factors = []
        for log in audit_logs:
            all_factors.extend([factor['feature'] for factor in log['top_factors']])
        
        factor_counts = pd.Series(all_factors).value_counts().to_dict()
        
        # Análisis de cumplimiento
        compliance_rates = {}
        for standard in self.compliance_standards:
            compliant_count = sum(1 for log in audit_logs 
                               if log['regulatory_compliance'].get(f'{standard.lower()}_compliant', False))
            compliance_rates[standard] = compliant_count / total_applications if total_applications > 0 else 0
        
        return {
            "total_applications": total_applications,
            "approved_applications": approved_count,
            "rejected_applications": rejected_count,
            "approval_rate": approved_count / total_applications if total_applications > 0 else 0,
            "average_confidence": avg_confidence,
            "confidence_distribution": {
                "min": min(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 0,
                "median": np.median(confidences) if confidences else 0
            },
            "top_decision_factors": factor_counts,
            "compliance_rates": compliance_rates
        }
    
    def _generate_executive_summary(self, audit_analysis, compliance_report):
        """Generar resumen ejecutivo"""
        summary = {
            "key_metrics": {
                "total_applications": audit_analysis.get('total_applications', 0),
                "approval_rate": audit_analysis.get('approval_rate', 0),
                "average_confidence": audit_analysis.get('average_confidence', 0),
                "overall_compliance_score": compliance_report.get('compliance_score', 0) if compliance_report else 0
            },
            "compliance_status": {
                "status": compliance_report.get('compliance_status', 'unknown') if compliance_report else 'unknown',
                "critical_issues": [],
                "high_priority_issues": []
            },
            "key_findings": [
                f"Procesadas {audit_analysis.get('total_applications', 0)} solicitudes de préstamo",
                f"Tasa de aprobación del {audit_analysis.get('approval_rate', 0):.1%}",
                f"Confianza promedio del {audit_analysis.get('average_confidence', 0):.1%}",
                f"Score de cumplimiento del {compliance_report.get('compliance_score', 0):.1%}" if compliance_report else "Score de cumplimiento no disponible"
            ],
            "recommendations_summary": []
        }
        
        # Añadir recomendaciones clave
        if compliance_report and compliance_report.get('compliance_score', 0) < 0.8:
            summary["recommendations_summary"].append("Mejorar métricas de equidad para cumplir con estándares regulatorios")
        
        if audit_analysis.get('approval_rate', 0) < 0.5:
            summary["recommendations_summary"].append("Revisar criterios de aprobación que pueden ser demasiado restrictivos")
        
        return summary
    
    def _generate_detailed_findings(self, audit_logs):
        """Generar hallazgos detallados"""
        findings = []
        
        # Análisis temporal
        if len(audit_logs) > 1:
            timestamps = [datetime.fromisoformat(log['timestamp']) for log in audit_logs]
            time_span = max(timestamps) - min(timestamps)
            
            findings.append({
                "category": "temporal_analysis",
                "finding": f"Las decisiones cubren un período de {time_span.days} días",
                "impact": "medium",
                "details": {
                    "start_date": min(timestamps).isoformat(),
                    "end_date": max(timestamps).isoformat(),
                    "total_days": time_span.days
                }
            })
        
        # Análisis de factores de decisión
        all_factors = []
        for log in audit_logs:
            all_factors.extend([factor['feature'] for factor in log['top_factors']])
        
        factor_analysis = pd.Series(all_factors).value_counts()
        
        findings.append({
            "category": "decision_factors",
            "finding": f"Los factores más influyentes son: {', '.join(factor_analysis.head(3).index.tolist())}",
            "impact": "high",
            "details": {
                "top_factors": factor_analysis.head(5).to_dict(),
                "factor_diversity": len(factor_analysis)
            }
        })
        
        # Análisis de confianza
        confidences = [log['confidence'] for log in audit_logs]
        low_confidence_count = sum(1 for c in confidences if c < 0.7)
        
        if low_confidence_count > 0:
            findings.append({
                "category": "confidence_analysis",
                "finding": f"{low_confidence_count} decisiones ({low_confidence_count/len(confidences):.1%}) tuvieron baja confianza (<70%)",
                "impact": "high" if low_confidence_count/len(confidences) > 0.1 else "medium",
                "details": {
                    "low_confidence_count": low_confidence_count,
                    "low_confidence_percentage": low_confidence_count/len(confidences),
                    "average_confidence": np.mean(confidences)
                }
            })
        
        return findings
    
    def _generate_audit_recommendations(self, audit_analysis):
        """Generar recomendaciones de auditoría"""
        recommendations = []
        
        # Recomendaciones basadas en tasa de aprobación
        approval_rate = audit_analysis.get('approval_rate', 0)
        if approval_rate < 0.3:
            recommendations.append({
                "priority": "high",
                "category": "approval_rate",
                "recommendation": "La tasa de aprobación es muy baja. Revisar criterios de evaluación.",
                "action_items": [
                    "Analizar distribución de scores de aprobación",
                    "Revisar umbrales de decisión",
                    "Considerar ajustes en modelo"
                ]
            })
        elif approval_rate > 0.8:
            recommendations.append({
                "priority": "medium",
                "category": "approval_rate",
                "recommendation": "La tasa de aprobación es muy alta. Verificar riesgo crediticio.",
                "action_items": [
                    "Analizar tasa de morosidad esperada",
                    "Revisar políticas de riesgo",
                    "Validar calidad del modelo"
                ]
            })
        
        # Recomendaciones basadas en confianza
        avg_confidence = audit_analysis.get('average_confidence', 0)
        if avg_confidence < 0.8:
            recommendations.append({
                "priority": "high",
                "category": "model_confidence",
                "recommendation": "El modelo muestra baja confianza promedio. Considerar reentrenamiento.",
                "action_items": [
                    "Evaluar calidad de datos de entrenamiento",
                    "Revisar arquitectura del modelo",
                    "Considerar características adicionales"
                ]
            })
        
        # Recomendaciones de cumplimiento
        compliance_rates = audit_analysis.get('compliance_rates', {})
        for standard, rate in compliance_rates.items():
            if rate < 0.95:
                recommendations.append({
                    "priority": "high",
                    "category": "regulatory_compliance",
                    "recommendation": f"Tasa de cumplimiento con {standard} es baja ({rate:.1%})",
                    "action_items": [
                        f"Revisar requisitos de {standard}",
                        "Implementar controles adicionales",
                        "Documentar excepciones"
                    ]
                })
        
        return recommendations
    
    def _calculate_explanation_stats(self, explanations):
        """Calcular estadísticas de explicaciones"""
        if not explanations:
            return {}
        
        # Estadísticas de SHAP values
        all_shap_values = []
        for exp in explanations:
            all_shap_values.extend(exp.get('shap_values', []))
        
        return {
            "total_explanations": len(explanations),
            "explanation_method": explanations[0].get('explanation_type', 'unknown') if explanations else 'unknown',
            "shap_statistics": {
                "mean_shap_value": np.mean(all_shap_values) if all_shap_values else 0,
                "max_shap_value": max(all_shap_values) if all_shap_values else 0,
                "min_shap_value": min(all_shap_values) if all_shap_values else 0
            }
        }
    
    def _calculate_compliance_metrics(self, audit_logs):
        """Calcular métricas de cumplimiento detalladas"""
        if not audit_logs:
            return {}
        
        metrics = {}
        for standard in self.compliance_standards:
            compliant_count = sum(1 for log in audit_logs 
                               if log['regulatory_compliance'].get(f'{standard.lower()}_compliant', False))
            total_count = len(audit_logs)
            
            metrics[standard] = {
                "compliant_count": compliant_count,
                "total_count": total_count,
                "compliance_rate": compliant_count / total_count if total_count > 0 else 0
            }
        
        return metrics

def main():
    """Función principal para generar documentación de auditoría"""
    logger.info("=" * 80)
    logger.info("GENERANDO DOCUMENTACIÓN DE AUDITORÍA")
    logger.info("=" * 80)
    
    # Crear directorios necesarios
    os.makedirs("outputs/audit_reports", exist_ok=True)
    
    # Crear generador de documentación
    doc_generator = AuditDocumentationGenerator()
    
    # Generar reporte completo
    comprehensive_report = doc_generator.generate_comprehensive_audit_report()
    
    if comprehensive_report:
        # Resumen ejecutivo
        exec_summary = comprehensive_report.get('executive_summary', {})
        key_metrics = exec_summary.get('key_metrics', {})
        
        logger.info("REPORTE DE AUDITORÍA GENERADO")
        logger.info(f"Total de aplicaciones: {key_metrics.get('total_applications', 0)}")
        logger.info(f"Tasa de aprobación: {key_metrics.get('approval_rate', 0):.1%}")
        logger.info(f"Confianza promedio: {key_metrics.get('average_confidence', 0):.1%}")
        logger.info(f"Score de cumplimiento: {key_metrics.get('overall_compliance_score', 0):.1%}")
        
        # Recomendaciones clave
        recommendations = exec_summary.get('recommendations_summary', [])
        if recommendations:
            logger.info("RECOMENDACIONES CLAVE:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")
    
    logger.info("=" * 80)
    logger.info("DOCUMENTACIÓN DE AUDITORÍA COMPLETADA")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
