# Module 07 - Monitoring, Détection & Réponse aux incidents ML

> **Durée** : ~6h | **Niveau** : Avancé

---

## 7.1 Monitoring de sécurité ML : vue d'ensemble

Le monitoring MLSecOps va bien au-delà du monitoring classique (latence, throughput). Il doit détecter des comportements adversariaux subtils qui ne déclenchent pas d'erreurs techniques.

```
MONITORING STACK MLSECOPS
──────────────────────────────────────────────────────────────────
                     ┌─────────────────────────────────────────┐
                     │          ALERTMANAGER / PAGERDUTY       │
                     └────────────────┬────────────────────────┘
                                      │ alertes
                     ┌────────────────▼────────────────────────┐
                     │              GRAFANA                    │
                     │   Dashboards sécurité + ML drift        │
                     └────────────────┬────────────────────────┘
                                      │ queries
          ┌───────────────────────────▼────────────────────────┐
          │                    PROMETHEUS                       │
          │   Métriques custom ML + security events            │
          └──────┬────────────────────────────┬────────────────┘
                 │ scrape                      │ push
    ┌────────────▼───────────┐    ┌────────────▼───────────────┐
    │   ML Serving (FastAPI)  │    │   ML Security Exporter     │
    │   /metrics endpoint     │    │   (custom collector)       │
    └─────────────────────────┘    └────────────────────────────┘
                 │                              │
    ┌────────────▼───────────┐    ┌────────────▼───────────────┐
    │   Evidently AI          │    │   Alibi Detect             │
    │   Data/Model Drift      │    │   Adversarial Detection    │
    └─────────────────────────┘    └────────────────────────────┘
```

---

## 7.2 Data Drift & Model Drift Detection

### 7.2.1 Détection avec Evidently

```python
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    ColumnDistributionMetric,
)
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import TestColumnDrift
import prometheus_client as prom

class MLDriftMonitor:
    """
    Moniteur de drift combinant Evidently et Prometheus.
    Détecte : data drift, model drift, distributional shift.
    """

    # Métriques Prometheus
    DRIFT_SCORE = prom.Gauge(
        'ml_drift_score', 
        'Dataset drift score (0=stable, 1=full drift)',
        ['feature_name']
    )
    DRIFT_DETECTED = prom.Gauge(
        'ml_drift_detected',
        'Binary flag: 1 if drift detected',
        ['dataset_name']
    )
    PSI_SCORE = prom.Gauge(
        'ml_psi_score',
        'Population Stability Index per feature',
        ['feature_name']
    )
    PREDICTION_DRIFT = prom.Gauge(
        'ml_prediction_distribution_drift',
        'Drift in prediction distribution'
    )

    def __init__(self, reference_df: pd.DataFrame, 
                 feature_columns: list[str],
                 label_column: str = None,
                 drift_threshold: float = 0.1):
        self.reference_df = reference_df
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.drift_threshold = drift_threshold

    def analyze_drift(self, current_df: pd.DataFrame, 
                       window_label: str = "current") -> dict:
        """
        Analyse complète du drift entre référence et données courantes.
        """
        # Rapport Evidently
        report = Report(metrics=[
            DatasetDriftMetric(stattest_threshold=self.drift_threshold),
            DatasetMissingValuesMetric(),
            *[ColumnDriftMetric(column_name=col) for col in self.feature_columns]
        ])

        report.run(
            reference_data=self.reference_df,
            current_data=current_df
        )

        report_dict = report.as_dict()
        drift_result = self._parse_report(report_dict, window_label)

        # Export vers Prometheus
        self._export_to_prometheus(drift_result)

        return drift_result

    def _parse_report(self, report_dict: dict, window_label: str) -> dict:
        """Parse les résultats Evidently en structure exploitable."""
        result = {
            "window": window_label,
            "dataset_drift_detected": False,
            "drift_share": 0.0,
            "feature_drifts": {}
        }

        for metric in report_dict.get("metrics", []):
            if metric["metric"] == "DatasetDriftMetric":
                result["dataset_drift_detected"] = metric["result"]["dataset_drift"]
                result["drift_share"] = metric["result"]["share_of_drifted_columns"]

            elif metric["metric"] == "ColumnDriftMetric":
                col = metric["result"]["column_name"]
                result["feature_drifts"][col] = {
                    "drift_detected": metric["result"]["drift_detected"],
                    "stattest": metric["result"]["stattest"],
                    "drift_score": metric["result"]["drift_score"],
                    "p_value": metric["result"].get("p_value")
                }

        return result

    def _export_to_prometheus(self, drift_result: dict):
        """Exporte les métriques de drift vers Prometheus."""
        dataset_drift = 1 if drift_result["dataset_drift_detected"] else 0
        self.DRIFT_DETECTED.labels(dataset_name="production").set(dataset_drift)

        for feature, drift_info in drift_result["feature_drifts"].items():
            self.DRIFT_SCORE.labels(feature_name=feature).set(
                drift_info["drift_score"]
            )

    def compute_psi(self, reference_series: pd.Series, 
                     current_series: pd.Series, 
                     n_bins: int = 10) -> float:
        """
        Population Stability Index (PSI).
        PSI < 0.1  → stable
        PSI < 0.2  → légère variation (surveillance)
        PSI >= 0.2 → drift significatif (alerte)
        PSI >= 0.25 → drift critique (rollback à envisager)
        """
        ref_counts, bin_edges = np.histogram(reference_series.dropna(), bins=n_bins)
        cur_counts, _ = np.histogram(current_series.dropna(), bins=bin_edges)

        ref_pct = (ref_counts + 1e-6) / len(reference_series)
        cur_pct = (cur_counts + 1e-6) / len(current_series)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def monitor_prediction_drift(self, 
                                   reference_preds: np.ndarray,
                                   current_preds: np.ndarray) -> dict:
        """
        Détecte le drift dans la distribution des prédictions.
        Un drift dans les outputs peut indiquer un poisoning ou un concept drift.
        """
        from scipy.stats import ks_2samp, wasserstein_distance

        ks_stat, ks_pvalue = ks_2samp(reference_preds, current_preds)
        w_dist = wasserstein_distance(reference_preds, current_preds)

        drift_detected = ks_pvalue < 0.05 or w_dist > 0.1

        self.PREDICTION_DRIFT.set(float(ks_stat))

        return {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "wasserstein_distance": float(w_dist),
            "drift_detected": drift_detected,
            "severity": (
                "CRITICAL" if w_dist > 0.3 else
                "HIGH" if w_dist > 0.2 else
                "MEDIUM" if drift_detected else
                "OK"
            )
        }
```

### 7.2.2 Test suite CI/CD pour la qualité des données

```python
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfMissingValues,
    TestShareOfMissingValues,
    TestColumnDrift,
    TestDatasetDrift,
    TestShareOfOutRangeValues,
    TestHighlyCorrelatedColumns,
    TestTargetFeaturesCorrelations,
)

def run_data_quality_gate(reference_df: pd.DataFrame, 
                           current_df: pd.DataFrame,
                           feature_columns: list[str]) -> dict:
    """
    Gate de qualité données à intégrer dans la CI.
    Bloque le déploiement si les tests échouent.
    """
    suite = TestSuite(tests=[
        TestNumberOfMissingValues(lte=0),
        TestShareOfMissingValues(lte=0.05),
        TestDatasetDrift(stattest_threshold=0.1),
        *[TestColumnDrift(column_name=col, stattest_threshold=0.1)
          for col in feature_columns],
        *[TestShareOfOutRangeValues(column_name=col, lte=0.01)
          for col in feature_columns],
    ])

    suite.run(reference_data=reference_df, current_data=current_df)
    results = suite.as_dict()

    passed = all(t["status"] == "SUCCESS" for t in results["tests"])
    failures = [t for t in results["tests"] if t["status"] != "SUCCESS"]

    return {
        "passed": passed,
        "total_tests": len(results["tests"]),
        "failed_tests": len(failures),
        "failures": failures
    }
```

---

## 7.3 Détection d'attaques adversariales en production

### 7.3.1 Détecteur d'inputs adversariaux avec Alibi Detect

```python
from alibi_detect.cd import MMDDrift, KSDrift, ChiSquareDrift
from alibi_detect.od import IForest, Mahalanobis, SpectralResidual
from alibi_detect.utils.saving import save_detector, load_detector
import numpy as np

class AdversarialInputDetector:
    """
    Détection d'inputs adversariaux et d'anomalies en production.
    Combine plusieurs approches complémentaires.
    """

    def __init__(self, X_train: np.ndarray, 
                 contamination: float = 0.01):
        """
        Args:
            X_train: données d'entraînement (référence)
            contamination: proportion attendue d'anomalies (1%)
        """
        self.X_train = X_train
        self.contamination = contamination
        self._detectors = {}
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialise les détecteurs sur les données de référence."""

        # 1. Isolation Forest - Détection d'outliers
        self._detectors["iforest"] = IForest(
            threshold=None,
            n_estimators=100,
            max_samples="auto",
            contamination=self.contamination
        )
        self._detectors["iforest"].fit(self.X_train)

        # 2. Mahalanobis Distance - Détection basée sur la distribution
        self._detectors["mahalanobis"] = Mahalanobis(threshold=None)
        self._detectors["mahalanobis"].fit(self.X_train)

        # 3. MMD Drift - Maximum Mean Discrepancy pour le drift de batch
        self._detectors["mmd_drift"] = MMDDrift(
            self.X_train,
            backend='numpy',
            p_val=0.05,
            n_permutations=100
        )

        # 4. KS Drift - Par feature
        self._detectors["ks_drift"] = KSDrift(
            self.X_train,
            p_val=0.05
        )

    def detect_single_input(self, x: np.ndarray) -> dict:
        """
        Détection pour un input unique.
        Utilisé en temps réel lors de l'inférence.
        """
        x_2d = x.reshape(1, -1) if x.ndim == 1 else x

        # IForest
        iforest_result = self._detectors["iforest"].predict(x_2d)
        is_outlier_iforest = bool(iforest_result["data"]["is_outlier"][0])
        iforest_score = float(iforest_result["data"]["instance_score"][0])

        # Mahalanobis
        mahal_result = self._detectors["mahalanobis"].predict(x_2d)
        mahal_score = float(mahal_result["data"]["instance_score"][0])

        # Score de risque composite
        risk_score = self._compute_composite_risk(iforest_score, mahal_score)

        return {
            "adversarial_detected": is_outlier_iforest or risk_score > 0.8,
            "risk_score": risk_score,
            "iforest_score": iforest_score,
            "mahalanobis_score": mahal_score,
            "severity": (
                "CRITICAL" if risk_score > 0.9 else
                "HIGH" if risk_score > 0.7 else
                "MEDIUM" if risk_score > 0.5 else
                "LOW"
            )
        }

    def detect_batch_drift(self, X_batch: np.ndarray) -> dict:
        """
        Détection de drift sur un batch de requêtes.
        À exécuter périodiquement (toutes les N requêtes).
        """
        mmd_result = self._detectors["mmd_drift"].predict(X_batch)
        ks_result = self._detectors["ks_drift"].predict(X_batch)

        return {
            "mmd_drift_detected": bool(mmd_result["data"]["is_drift"]),
            "mmd_pvalue": float(mmd_result["data"]["p_val"]),
            "ks_drift_detected": bool(ks_result["data"]["is_drift"]),
            "ks_feature_pvalues": ks_result["data"]["p_val"].tolist(),
            "drift_severity": (
                "CRITICAL" if mmd_result["data"]["p_val"] < 0.001 else
                "HIGH" if mmd_result["data"]["p_val"] < 0.01 else
                "MEDIUM" if mmd_result["data"]["is_drift"] else
                "OK"
            )
        }

    def _compute_composite_risk(self, iforest_score: float, 
                                  mahal_score: float) -> float:
        """Score de risque composite normalisé [0, 1]."""
        # Normalisation empirique (à calibrer sur vos données)
        norm_iforest = min(max(iforest_score / 0.5, 0), 1)
        norm_mahal = min(max(mahal_score / 10.0, 0), 1)
        return 0.6 * norm_iforest + 0.4 * norm_mahal
```

### 7.3.2 Détection de model extraction en temps réel

```python
from collections import deque
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ModelExtractionDetector:
    """
    Détecte les tentatives d'extraction de modèle via analyse des patterns de requêtes.

    Signaux :
    - Requêtes near-decision-boundary concentrées
    - Haute similarité cosinus entre inputs successifs
    - Volume anormalement élevé depuis une même source
    - Distribution des inputs non représentative des données réelles
    """

    def __init__(self, 
                 window_size: int = 500,
                 boundary_confidence_threshold: float = 0.55,
                 similarity_threshold: float = 0.97,
                 boundary_rate_threshold: float = 0.25):
        self.window = deque(maxlen=window_size)
        self.boundary_threshold = boundary_confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.boundary_rate_threshold = boundary_rate_threshold
        self.per_ip_counts: dict = {}
        self.alerts: list = []

    def record_request(self, client_ip: str, 
                        input_features: np.ndarray,
                        prediction_proba: np.ndarray) -> dict:
        """
        Enregistre une requête et évalue le risque d'extraction.
        """
        now = time.time()
        max_confidence = float(prediction_proba.max())
        is_near_boundary = max_confidence < self.boundary_threshold

        record = {
            "timestamp": now,
            "ip": client_ip,
            "features": input_features,
            "max_confidence": max_confidence,
            "near_boundary": is_near_boundary
        }
        self.window.append(record)

        # Compteur par IP
        self.per_ip_counts[client_ip] = \
            self.per_ip_counts.get(client_ip, 0) + 1

        return self._analyze_window(client_ip, input_features)

    def _analyze_window(self, current_ip: str, 
                         current_features: np.ndarray) -> dict:
        """Analyse la fenêtre courante pour détecter des patterns suspects."""
        if len(self.window) < 50:
            return {"risk": "INSUFFICIENT_DATA", "score": 0.0}

        window_list = list(self.window)
        risks = []

        # 1. Taux de requêtes near-boundary
        boundary_rate = sum(1 for r in window_list if r["near_boundary"]) / len(window_list)
        if boundary_rate > self.boundary_rate_threshold:
            risks.append({
                "type": "HIGH_BOUNDARY_RATE",
                "value": boundary_rate,
                "threshold": self.boundary_rate_threshold,
                "severity": "HIGH"
            })

        # 2. Similarité avec les dernières requêtes du même IP
        same_ip_records = [r for r in window_list[-50:] if r["ip"] == current_ip]
        if len(same_ip_records) > 5:
            recent_features = np.vstack([r["features"] for r in same_ip_records[-5:]])
            similarities = cosine_similarity([current_features], recent_features)[0]
            avg_similarity = float(similarities.mean())

            if avg_similarity > self.similarity_threshold:
                risks.append({
                    "type": "HIGH_INPUT_SIMILARITY",
                    "value": avg_similarity,
                    "threshold": self.similarity_threshold,
                    "severity": "HIGH"
                })

        # 3. Volume anormal par IP
        ip_count = self.per_ip_counts.get(current_ip, 0)
        total_count = len(self.window)
        ip_share = ip_count / max(total_count, 1)

        if ip_share > 0.5 and ip_count > 100:
            risks.append({
                "type": "HIGH_VOLUME_SINGLE_IP",
                "value": ip_share,
                "ip_count": ip_count,
                "severity": "CRITICAL"
            })

        # Score composite
        severity_scores = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.2}
        composite_score = max(
            (severity_scores.get(r["severity"], 0) for r in risks),
            default=0.0
        )

        result = {
            "risks": risks,
            "composite_score": composite_score,
            "risk_level": (
                "CRITICAL" if composite_score >= 1.0 else
                "HIGH" if composite_score >= 0.7 else
                "MEDIUM" if composite_score >= 0.4 else
                "LOW"
            ),
            "boundary_rate": boundary_rate,
            "ip_request_count": ip_count
        }

        if composite_score >= 0.7:
            self._trigger_alert(current_ip, result)

        return result

    def _trigger_alert(self, ip: str, analysis: dict):
        alert = {
            "timestamp": time.time(),
            "type": "POTENTIAL_MODEL_EXTRACTION",
            "source_ip": ip,
            "analysis": analysis
        }
        self.alerts.append(alert)
        # En production : envoyer vers SIEM / Alertmanager
        print(f"[SECURITY ALERT] Model extraction attempt from {ip}: "
              f"score={analysis['composite_score']:.2f}")
```

---

## 7.4 Dashboards Grafana pour MLSecOps

```json
{
  "title": "MLSecOps Security Dashboard",
  "panels": [
    {
      "title": "Prompt Injection Attempts",
      "type": "stat",
      "targets": [{
        "expr": "sum(rate(llm_prompt_injection_attempts_total[5m])) * 60",
        "legendFormat": "Injections/min"
      }],
      "thresholds": {"steps": [
        {"color": "green", "value": 0},
        {"color": "yellow", "value": 5},
        {"color": "red", "value": 10}
      ]}
    },
    {
      "title": "Data Drift Score - Features",
      "type": "timeseries",
      "targets": [{
        "expr": "ml_drift_score",
        "legendFormat": "{{feature_name}}"
      }],
      "thresholds": {"steps": [
        {"color": "green", "value": 0},
        {"color": "yellow", "value": 0.1},
        {"color": "red", "value": 0.25}
      ]}
    },
    {
      "title": "Adversarial Input Detection Rate",
      "type": "timeseries",
      "targets": [{
        "expr": "rate(ml_adversarial_inputs_detected_total[5m])",
        "legendFormat": "Adversarial inputs/s"
      }]
    },
    {
      "title": "Model Extraction Risk Score",
      "type": "gauge",
      "targets": [{
        "expr": "ml_extraction_risk_score",
        "legendFormat": "Extraction Risk"
      }],
      "thresholds": {"steps": [
        {"color": "green", "value": 0},
        {"color": "orange", "value": 0.5},
        {"color": "red", "value": 0.8}
      ]}
    },
    {
      "title": "PII in LLM Outputs",
      "type": "stat",
      "targets": [{
        "expr": "sum(increase(llm_pii_in_output_total[1h]))",
        "legendFormat": "PII détectées (1h)"
      }],
      "thresholds": {"steps": [
        {"color": "green", "value": 0},
        {"color": "red", "value": 1}
      ]}
    },
    {
      "title": "Prediction Distribution Drift (KS stat)",
      "type": "timeseries",
      "targets": [{
        "expr": "ml_prediction_distribution_drift",
        "legendFormat": "KS stat"
      }]
    }
  ]
}
```

### Règles d'alerte Prometheus

```yaml
# prometheus/rules/mlsecops.yml

groups:
  - name: mlsecops_security
    interval: 30s
    rules:

      # Drift critique
      - alert: MLDatasetDriftCritical
        expr: ml_drift_detected{dataset_name="production"} == 1
        for: 5m
        labels:
          severity: critical
          team: mlops
        annotations:
          summary: "Dataset drift détecté en production"
          description: >
            Le dataset de production a dérivé significativement du dataset de référence.
            Vérifier les nouvelles données et envisager un ré-entraînement.
          runbook: "https://wiki.company.com/runbooks/ml-drift"

      # Tentatives d'extraction
      - alert: ModelExtractionAttempt
        expr: ml_extraction_risk_score > 0.8
        for: 2m
        labels:
          severity: critical
          team: security
        annotations:
          summary: "Tentative d'extraction de modèle détectée"
          description: >
            Score de risque d'extraction : {{ $value }}.
            Analyser les requêtes récentes et envisager un blocage IP.

      # Injections prompt LLM
      - alert: LLMPromptInjectionSpike
        expr: sum(rate(llm_prompt_injection_attempts_total[5m])) > 0.5
        for: 3m
        labels:
          severity: high
          team: security
        annotations:
          summary: "Pic d'injections prompt LLM"
          description: "Taux d'injections : {{ $value }} /s sur 5 min"

      # PII dans les outputs
      - alert: PIILeakageInOutput
        expr: increase(llm_pii_in_output_total[5m]) > 0
        labels:
          severity: critical
          team: security
        annotations:
          summary: "PII détectée dans les outputs LLM"
          description: >
            Des données personnelles ont été détectées dans les réponses du LLM.
            Action immédiate requise - vérifier les guardrails.

      # Latence anormale (sponge attack)
      - alert: MLServingAbnormalLatency
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: high
          team: mlops
        annotations:
          summary: "Latence d'inférence anormalement haute (possible sponge attack)"

      # PSI critique par feature
      - alert: FeaturePSICritical
        expr: ml_psi_score > 0.25
        for: 10m
        labels:
          severity: high
          team: mlops
        annotations:
          summary: "PSI critique sur la feature {{ $labels.feature_name }}"
          description: "PSI = {{ $value }} (seuil critique = 0.25)"
```

---

## 7.5 Réponse aux incidents ML

### 7.5.1 Playbooks d'incident response

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable
import time

class IncidentType(Enum):
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    ADVERSARIAL_ATTACK = "adversarial_attack"
    LLM_JAILBREAK = "llm_jailbreak"
    DATA_DRIFT = "data_drift"
    PII_LEAKAGE = "pii_leakage"

@dataclass
class MLIncident:
    incident_type: IncidentType
    severity: str           # CRITICAL, HIGH, MEDIUM, LOW
    detected_at: float = field(default_factory=time.time)
    source_ip: str = ""
    details: dict = field(default_factory=dict)
    status: str = "OPEN"    # OPEN, INVESTIGATING, CONTAINED, RESOLVED
    timeline: list = field(default_factory=list)

    def add_event(self, event: str):
        self.timeline.append({
            "timestamp": time.time(),
            "event": event
        })

class MLIncidentResponsePlaybook:
    """
    Playbooks de réponse aux incidents spécifiques au ML.
    """

    PLAYBOOKS: dict[IncidentType, list[dict]] = {

        IncidentType.DATA_POISONING: [
            {"step": 1, "action": "CONTAIN",
             "description": "Suspendre le pipeline d'entraînement automatique",
             "automated": True,
             "command": "kubectl scale cronjob ml-training --replicas=0"},

            {"step": 2, "action": "INVESTIGATE",
             "description": "Identifier la fenêtre temporelle de l'injection",
             "automated": False,
             "guidance": "Vérifier les hashs SHA256 des datasets via le data manifest"},

            {"step": 3, "action": "INVESTIGATE",
             "description": "Rollback du dataset à la dernière version validée",
             "automated": True,
             "command": "dvc checkout data/processed/@last-validated"},

            {"step": 4, "action": "RECOVER",
             "description": "Ré-entraîner sur les données validées",
             "automated": False,
             "guidance": "Lancer le pipeline de training sur les données assainies"},

            {"step": 5, "action": "RECOVER",
             "description": "Valider les métriques du nouveau modèle",
             "automated": True,
             "command": "python src/adversarial_eval.py --model models/retrained.pkl"},

            {"step": 6, "action": "MONITOR",
             "description": "Surveillance renforcée pendant 48h post-incident",
             "automated": True,
             "command": "python src/set_alert_sensitivity.py --level HIGH --duration 48h"},
        ],

        IncidentType.MODEL_EXTRACTION: [
            {"step": 1, "action": "CONTAIN",
             "description": "Bloquer l'IP source au niveau WAF/API Gateway",
             "automated": True,
             "command": "python src/block_ip.py --ip {source_ip} --duration 24h"},

            {"step": 2, "action": "CONTAIN",
             "description": "Réduire le rate limit global temporairement",
             "automated": True,
             "command": "kubectl set env deployment/ml-serving RATE_LIMIT=10"},

            {"step": 3, "action": "INVESTIGATE",
             "description": "Estimer le volume de données exfiltrées (queries log)",
             "automated": False,
             "guidance": "Analyser les logs des dernières 24h : volume, patterns, IPs"},

            {"step": 4, "action": "RECOVER",
             "description": "Évaluer si un retrain ou une rotation de modèle est nécessaire",
             "automated": False,
             "guidance": "Si > 10k queries near-boundary : considérer le watermarking du modèle"},

            {"step": 5, "action": "HARDEN",
             "description": "Activer le Prediction API obfuscation",
             "automated": True,
             "command": "kubectl set env deployment/ml-serving OBFUSCATE_OUTPUTS=true"},
        ],

        IncidentType.LLM_JAILBREAK: [
            {"step": 1, "action": "CONTAIN",
             "description": "Logger et bloquer l'utilisateur/session",
             "automated": True},

            {"step": 2, "action": "INVESTIGATE",
             "description": "Analyser le payload pour comprendre la technique",
             "automated": False,
             "guidance": "Classifier la technique : DAN, prefix injection, many-shot, etc."},

            {"step": 3, "action": "HARDEN",
             "description": "Ajouter le pattern au filtre d'injection",
             "automated": False,
             "guidance": "Mettre à jour .banlist et llm_guard patterns"},

            {"step": 4, "action": "RECOVER",
             "description": "Vérifier si des données sensibles ont été exposées",
             "automated": False,
             "guidance": "Analyser les outputs de la session compromise"},
        ],

        IncidentType.PII_LEAKAGE: [
            {"step": 1, "action": "CONTAIN",
             "description": "Suspendre le service LLM immédiatement",
             "automated": True,
             "command": "kubectl scale deployment llm-serving --replicas=0"},

            {"step": 2, "action": "INVESTIGATE",
             "description": "Identifier les données exposées et les utilisateurs affectés",
             "automated": False,
             "guidance": "RGPD Art.33 : notification CNIL sous 72h si violation confirmée"},

            {"step": 3, "action": "HARDEN",
             "description": "Renforcer les filtres output avant remise en production",
             "automated": False},

            {"step": 4, "action": "NOTIFY",
             "description": "Notifier DPO et équipe juridique",
             "automated": False,
             "guidance": "Template notification CNIL disponible dans confluence/legal"},
        ]
    }

    def execute_playbook(self, incident: MLIncident,
                          auto_execute: bool = False) -> list[dict]:
        """
        Exécute le playbook pour un type d'incident donné.
        """
        playbook = self.PLAYBOOKS.get(incident.incident_type, [])
        results = []

        incident.add_event(f"Playbook démarré pour {incident.incident_type.value}")

        for step in playbook:
            print(f"\n[STEP {step['step']}] [{step['action']}] {step['description']}")

            step_result = {
                "step": step["step"],
                "action": step["action"],
                "description": step["description"],
                "status": "PENDING"
            }

            if step.get("automated") and auto_execute and "command" in step:
                cmd = step["command"].format(
                    source_ip=incident.source_ip
                )
                print(f"  → Commande automatique : {cmd}")
                # En production : subprocess.run(cmd.split())
                step_result["status"] = "EXECUTED"
                step_result["command"] = cmd

            elif step.get("guidance"):
                print(f"  → Guidance : {step['guidance']}")
                step_result["status"] = "MANUAL_REQUIRED"
                step_result["guidance"] = step["guidance"]

            incident.add_event(f"Step {step['step']} : {step_result['status']}")
            results.append(step_result)

        return results
```

### 7.5.2 Post-Mortem template ML

```markdown
# Post-Mortem - Incident ML

## Métadonnées
- **ID Incident** : INC-ML-YYYY-XXX
- **Type** : [Data Poisoning / Model Extraction / Adversarial / LLM Jailbreak / PII Leakage]
- **Sévérité** : [CRITICAL / HIGH / MEDIUM / LOW]
- **Détecté le** : YYYY-MM-DD HH:MM UTC
- **Résolu le** : YYYY-MM-DD HH:MM UTC
- **Durée totale** : Xh Ymin
- **Modèles impactés** : model_name vX.X.X

## Résumé (5 lignes max)

## Impact
- Utilisateurs affectés : N
- Prédictions compromises : N (X% du volume)
- Données exposées : [OUI/NON] - si OUI, nature des données
- Coût estimé : €

## Timeline

| Heure (UTC) | Événement |
|-------------|-----------|
| HH:MM | Première anomalie détectée (monitoring) |
| HH:MM | Alerte déclenchée |
| HH:MM | Équipe notifiée |
| HH:MM | Incident confirmé |
| HH:MM | Containment appliqué |
| HH:MM | Modèle rollback |
| HH:MM | Service restauré |
| HH:MM | Incident clôturé |

## Cause racine (Root Cause Analysis)

### Cause immédiate
_Qu'est-ce qui a directement causé l'incident ?_

### Causes profondes (5 Pourquoi)
1. **Pourquoi ?** → ...
2. **Pourquoi ?** → ...
3. **Pourquoi ?** → ...

## Ce qui a bien fonctionné
- Le monitoring X a détecté l'anomalie en Y minutes
- Le playbook a été suivi correctement
- ...

## Ce qui n'a pas fonctionné
- L'alerte A n'a pas déclenché car le seuil était trop élevé
- ...

## Actions correctives

| Action | Owner | Délai | Priorité |
|--------|-------|-------|----------|
| Baisser le seuil d'alerte PSI à 0.15 | @mlops | J+7 | HIGH |
| Ajouter le pattern d'injection au filtre | @security | J+2 | CRITICAL |
| ... | | | |

## Leçons apprises
```

---

## Résumé du module

- Evidently fournit la détection de data/model drift avec export Prometheus natif
- Le PSI (Population Stability Index) est la métrique standard pour la stabilité des features
- Alibi Detect (IForest + Mahalanobis + MMD) couvre la détection d'anomalies et d'adversarial inputs
- La détection de model extraction repose sur des heuristiques comportementales (boundary queries, similarité)
- Les playbooks d'incident response doivent couvrir : containment, investigation, recovery, hardening
- Le post-mortem ML inclut les métriques spécifiques : prédictions compromises, rollback modèle

---

*Module suivant → [08 - Conformité, Gouvernance & EU AI Act](08_COMPLIANCE.md)*  
*Module précédent → [06 - Secure MLOps Pipeline & CI/CD](06_SECURE_MLOPS.md)*
