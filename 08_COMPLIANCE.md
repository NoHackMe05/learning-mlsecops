# Module 08 - Conformité, Gouvernance & EU AI Act

> **Durée** : ~6h | **Niveau** : Avancé

---

## 8.1 EU AI Act - Vue d'ensemble et classification des risques

Le règlement européen sur l'intelligence artificielle (EU AI Act, en vigueur depuis août 2024, application progressive jusqu'en 2026) introduit un cadre réglementaire basé sur la **classification par niveau de risque**.

### 8.1.1 Les 4 niveaux de risque

```
EU AI ACT - PYRAMIDE DES RISQUES
──────────────────────────────────────────────────────
                    ┌─────────────────┐
                    │  INACCEPTABLE   │  ← Interdit
                    │  (Art. 5)       │
                    └────────┬────────┘
                    ┌────────▼────────┐
                    │  HAUT RISQUE   │  ← Obligations strictes
                    │  (Art. 6-9,    │     (CE marking, audit, logs)
                    │   Annex III)   │
                    └────────┬────────┘
                    ┌────────▼────────┐
                    │  RISQUE LIMITÉ │  ← Transparence
                    │  (Art. 50)     │     (disclosure obligatoire)
                    └────────┬────────┘
                    ┌────────▼────────┐
                    │  RISQUE MINIMAL│  ← Code de conduite
                    │  (majorité)    │     (volontaire)
                    └─────────────────┘
```

### 8.1.2 Systèmes interdits (Art. 5)

- Manipulation comportementale subliminale
- Exploitation des vulnérabilités de groupes spécifiques
- **Social scoring** par autorités publiques
- Identification biométrique en temps réel dans l'espace public (sauf exceptions)
- Reconnaissance d'émotions en milieu professionnel/scolaire
- Catégorisation biométrique inférant race, opinions politiques, orientation sexuelle

### 8.1.3 Systèmes à haut risque (Annex III)

```python
HIGH_RISK_DOMAINS = {
    "biometric_identification": [
        "Identification et catégorisation biométrique",
        "Reconnaissance faciale",
        "Emotion recognition"
    ],
    "critical_infrastructure": [
        "Gestion du trafic routier",
        "Distribution eau/énergie/gaz",
        "Infrastructure numérique critique"
    ],
    "education": [
        "Scoring et évaluation des élèves",
        "Accès aux établissements",
        "Détection de triche"
    ],
    "employment": [
        "Recrutement et sélection",
        "Promotion, résiliation",
        "Monitoring de productivité"
    ],
    "essential_services": [
        "Scoring de crédit",
        "Souscription d'assurance",
        "Évaluation des risques vie/santé"
    ],
    "law_enforcement": [
        "Profilage individuel",
        "Évaluation fiabilité des preuves",
        "Prédiction des risques criminels"
    ],
    "migration": [
        "Évaluation des risques migrants",
        "Traitement des demandes d'asile",
        "Vérification authenticité documents"
    ],
    "justice": [
        "Application de la loi par IA",
        "Résolution de litiges",
        "Interprétation de la loi"
    ]
}
```

---

## 8.2 Obligations pour les systèmes haut risque

### 8.2.1 Checklist de conformité EU AI Act

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class ComplianceStatus(Enum):
    COMPLIANT = "COMPLIANT"
    PARTIAL = "PARTIAL"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_APPLICABLE = "NOT_APPLICABLE"

@dataclass
class ComplianceRequirement:
    article: str
    requirement: str
    description: str
    evidence_required: list[str]
    status: ComplianceStatus = ComplianceStatus.NON_COMPLIANT
    evidence_provided: list[str] = field(default_factory=list)
    notes: str = ""

EU_AI_ACT_REQUIREMENTS = [

    ComplianceRequirement(
        article="Art. 9",
        requirement="Risk Management System",
        description="Système documenté d'identification, analyse et mitigation des risques",
        evidence_required=[
            "Risk assessment document",
            "Residual risks documentation",
            "Risk mitigation measures",
            "Periodic review records"
        ]
    ),

    ComplianceRequirement(
        article="Art. 10",
        requirement="Data and Data Governance",
        description="Données d'entraînement et validation conformes, représentatives et sans biais",
        evidence_required=[
            "Data governance policy",
            "Data quality metrics",
            "Bias assessment report",
            "Data lineage documentation",
            "Privacy impact assessment"
        ]
    ),

    ComplianceRequirement(
        article="Art. 11",
        requirement="Technical Documentation",
        description="Documentation technique complète du système AI",
        evidence_required=[
            "System architecture documentation",
            "Model card",
            "Training data description",
            "Performance metrics",
            "Limitations and known issues"
        ]
    ),

    ComplianceRequirement(
        article="Art. 12",
        requirement="Record-Keeping",
        description="Logs automatiques permettant la traçabilité post-déploiement",
        evidence_required=[
            "Logging system documentation",
            "Audit trail examples",
            "Log retention policy",
            "Access control to logs"
        ]
    ),

    ComplianceRequirement(
        article="Art. 13",
        requirement="Transparency and Information",
        description="Les utilisateurs doivent être informés qu'ils interagissent avec un système AI",
        evidence_required=[
            "User disclosure policy",
            "UI/UX disclosure screenshots",
            "Information provided to deployers"
        ]
    ),

    ComplianceRequirement(
        article="Art. 14",
        requirement="Human Oversight",
        description="Possibilité de supervision humaine effective, override et arrêt",
        evidence_required=[
            "Human oversight procedures",
            "Override capability documentation",
            "Stop function documentation",
            "Operator training records"
        ]
    ),

    ComplianceRequirement(
        article="Art. 15",
        requirement="Accuracy, Robustness and Cybersecurity",
        description="Niveau approprié d'accuracy, robustesse aux erreurs et aux attaques",
        evidence_required=[
            "Accuracy metrics on diverse test sets",
            "Adversarial robustness test results",
            "Cybersecurity assessment",
            "Fallback mechanism documentation"
        ]
    ),

    ComplianceRequirement(
        article="Art. 17",
        requirement="Quality Management System",
        description="Système de management qualité couvrant tout le cycle de vie",
        evidence_required=[
            "QMS documentation",
            "Version control policy",
            "Change management process",
            "Post-market monitoring plan"
        ]
    ),
]


class EUAIActComplianceAssessor:
    """
    Outil d'évaluation de conformité EU AI Act.
    """

    def __init__(self, system_name: str, risk_level: str):
        self.system_name = system_name
        self.risk_level = risk_level  # "high", "limited", "minimal"
        self.requirements = EU_AI_ACT_REQUIREMENTS.copy()

    def assess(self, evidence_map: dict[str, list[str]]) -> dict:
        """
        Évalue la conformité en fonction des preuves fournies.

        Args:
            evidence_map: {article: [liste des preuves disponibles]}
        """
        results = []
        overall_score = 0

        for req in self.requirements:
            provided = evidence_map.get(req.article, [])
            required = req.evidence_required

            coverage = len([e for e in required if any(e.lower() in p.lower() for p in provided)])
            coverage_pct = coverage / len(required) if required else 1.0

            if coverage_pct >= 1.0:
                status = ComplianceStatus.COMPLIANT
                score = 1.0
            elif coverage_pct >= 0.5:
                status = ComplianceStatus.PARTIAL
                score = 0.5
            else:
                status = ComplianceStatus.NON_COMPLIANT
                score = 0.0

            overall_score += score
            results.append({
                "article": req.article,
                "requirement": req.requirement,
                "status": status.value,
                "coverage_pct": round(coverage_pct * 100, 1),
                "missing_evidence": [e for e in required if not any(e.lower() in p.lower() for p in provided)]
            })

        overall_pct = (overall_score / len(self.requirements)) * 100

        return {
            "system_name": self.system_name,
            "risk_level": self.risk_level,
            "overall_compliance_pct": round(overall_pct, 1),
            "ready_for_deployment": overall_pct >= 90,
            "requirements": results,
            "critical_gaps": [r for r in results if r["status"] == "NON_COMPLIANT"]
        }
```

### 8.2.2 Technical Documentation (Art. 11) - Template

```markdown
# EU AI Act Technical Documentation
## [Nom du Système AI] - Version X.X.X

*Conforme EU AI Act Annex IV*

---

### 1. Description générale du système

**Nom** : ...
**Fournisseur** : ...
**Version** : ...
**Date de création** : ...
**Classification risque** : Haut risque - [domaine Annex III]

**Usage prévu** :
[Description de l'usage prévu, du contexte de déploiement, des utilisateurs finaux]

**Usage abusif raisonnablement prévisible** :
[Usage qui n'est pas prévu mais peut raisonnablement être anticipé]

---

### 2. Architecture du système

**Type de modèle** : [XGBoost / Neural Network / LLM / ...]
**Inputs** : [Liste des données d'entrée]
**Outputs** : [Liste des sorties et leur format]
**Composants** : [Pipeline complet de l'ingestion à l'inférence]

---

### 3. Données d'entraînement

| Attribut | Valeur |
|----------|--------|
| Sources | ... |
| Volume | N échantillons |
| Période couverte | AAAA–AAAA |
| Démographie couverte | ... |
| Méthode de labelling | ... |
| Biais identifiés | ... |
| Mesures de mitigation des biais | ... |

**Conformité RGPD** :
- Base légale : [Intérêt légitime / Consentement / Contrat]
- DPO notifié : OUI / NON
- AIPD réalisée : OUI / NON - Référence : ...

---

### 4. Métriques de performance

| Métrique | Valeur globale | Sur sous-groupe A | Sur sous-groupe B |
|----------|---------------|-------------------|-------------------|
| Accuracy | X% | X% | X% |
| F1-Score | X | X | X |
| Precision | X% | X% | X% |
| Recall | X% | X% | X% |
| AUC-ROC | X | X | X |

**Analyse des biais** :
[Résultats des tests d'équité par groupe démographique]

---

### 5. Robustesse et cybersécurité

| Test | Résultat | Seuil requis | Pass/Fail |
|------|----------|--------------|-----------|
| FGSM (ε=0.1) | X% acc | ≥ 70% | ✅/❌ |
| PGD (ε=0.1) | X% acc | ≥ 60% | ✅/❌ |
| MIA accuracy | X% | ≤ 60% | ✅/❌ |
| Dependency CVEs | N HIGH | 0 CRITICAL | ✅/❌ |

---

### 6. Limitations connues

1. ...
2. ...

---

### 7. Supervision humaine

**Mécanismes** :
- Dashboard de monitoring avec alertes
- Possibilité de rollback en < 15 min
- Bouton d'arrêt d'urgence (kill switch)
- Review humaine obligatoire pour [cas limite X]

---

### 8. Logs et traçabilité

- Rétention : 7 ans (exigence EU AI Act Art. 12)
- Format : JSON immuable (Write-Once storage)
- Contenu : input hash, output, timestamp, model version, user ID hashé
- Accès : restreint à [liste des rôles]
```

---

## 8.3 NIST AI Risk Management Framework (AI RMF)

### 8.3.1 Les 4 fonctions du NIST AI RMF

```
NIST AI RMF 1.0 - 4 FONCTIONS PRINCIPALES
──────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────┐
│  GOVERN                                                         │
│  ────────                                                       │
│  Culture organisationnelle, politiques, rôles, responsabilités  │
│  Policies · Accountability · Culture · Teams                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │           MAP              │
         │  Identifier les risques AI  │
         │  Context · Categorization  │
         │  Risk identification        │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │          MEASURE           │
         │  Évaluer et mesurer        │
         │  Analyse · Prioritization  │
         │  Testing · Evaluation      │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │          MANAGE            │
         │  Traiter et gérer les      │
         │  risques identifiés        │
         │  Response · Recovery       │
         └────────────────────────────┘
```

### 8.3.2 Mapping NIST AI RMF → MLSecOps

```python
NIST_AIRF_MAPPING = {

    "GOVERN": {
        "GV-1.1": {
            "description": "Politiques, processus, procédures et pratiques AI documentés",
            "mlsecops_controls": [
                "ML Security Policy documentée",
                "Threat modeling process",
                "Incident response playbooks",
                "Model risk governance committee"
            ]
        },
        "GV-1.2": {
            "description": "Responsabilités de sécurité AI assignées",
            "mlsecops_controls": [
                "ML Security Champion désigné",
                "RACI matrix pour la sécurité ML",
                "Formation MLSecOps de l'équipe"
            ]
        },
        "GV-6.1": {
            "description": "Politiques de gestion de la supply chain AI",
            "mlsecops_controls": [
                "Vendor assessment pour datasets et modèles",
                "ML-SBOM systématique",
                "Politique d'approbation des sources de modèles"
            ]
        }
    },

    "MAP": {
        "MP-2.3": {
            "description": "Catégorisation des risques AI par probabilité et impact",
            "mlsecops_controls": [
                "DREAD scoring adapté ML",
                "Attack tree modeling",
                "Risk register ML"
            ]
        },
        "MP-5.1": {
            "description": "Identification des risques de la supply chain ML",
            "mlsecops_controls": [
                "Audit des sources de datasets",
                "Scan des modèles pré-entraînés",
                "pip-audit automatisé en CI"
            ]
        }
    },

    "MEASURE": {
        "MS-2.5": {
            "description": "Évaluation de la robustesse et de la résilience",
            "mlsecops_controls": [
                "Tests adversariaux FGSM/PGD en CI",
                "CLEVER score measurement",
                "Membership inference evaluation"
            ]
        },
        "MS-2.6": {
            "description": "Évaluation des risques privacy",
            "mlsecops_controls": [
                "MIA accuracy monitoring",
                "Differential privacy budget tracking",
                "PII scan sur les datasets"
            ]
        },
        "MS-4.1": {
            "description": "Évaluation continue en production",
            "mlsecops_controls": [
                "Data drift monitoring (Evidently)",
                "Adversarial input detection (Alibi Detect)",
                "Behavioral monitoring dashboards"
            ]
        }
    },

    "MANAGE": {
        "MG-2.2": {
            "description": "Stratégies de réponse aux incidents AI documentées",
            "mlsecops_controls": [
                "Playbooks par type d'incident ML",
                "Rollback automatisé",
                "Post-mortem process"
            ]
        },
        "MG-3.1": {
            "description": "Mécanismes de monitoring post-déploiement",
            "mlsecops_controls": [
                "Prometheus + Grafana MLSecOps",
                "Alertmanager avec escalade",
                "Periodic red-teaming"
            ]
        }
    }
}
```

---

## 8.4 ISO/IEC 42001 - Management System for AI

ISO/IEC 42001 (2023) est la première norme de système de management pour l'IA, analogue à l'ISO 27001 pour la sécurité de l'information.

### 8.4.1 Structure de la norme

```
ISO/IEC 42001:2023 - STRUCTURE (Format Annex SL)
──────────────────────────────────────────────────
Clause 4  : Contexte de l'organisation
  4.1 Compréhension de l'organisation
  4.2 Parties prenantes
  4.3 Périmètre du AIMS
  4.4 Système de management AI (AIMS)

Clause 5  : Leadership
  5.1 Leadership et engagement de la direction
  5.2 Politique AI
  5.3 Rôles, responsabilités, autorités

Clause 6  : Planification
  6.1 Évaluation des risques et opportunités AI
  6.2 Objectifs et plans AI

Clause 7  : Support
  7.1 Ressources
  7.2 Compétences
  7.3 Sensibilisation
  7.4 Communication
  7.5 Information documentée

Clause 8  : Exploitation
  8.1 Planification et contrôle opérationnel
  8.2 Évaluation de l'impact AI
  8.3 Système AI - développement et acquisition
  8.4 Cycle de vie du système AI

Clause 9  : Évaluation des performances
  9.1 Monitoring, mesure, analyse et évaluation
  9.2 Audit interne
  9.3 Revue de direction

Clause 10 : Amélioration
  10.1 Non-conformités et actions correctives
  10.2 Amélioration continue
```

### 8.4.2 Mapping ISO 42001 vs ISO 27001 pour le MLSecOps Engineer

| Concept | ISO 27001 | ISO 42001 |
|---------|-----------|-----------|
| Actif principal | Information | Système AI + données |
| Évaluation des risques | Risques SI | Risques AI (biais, robustesse, privacy) |
| Contrôles | Annexe A (93 contrôles) | Annexe A (38 objectifs AI) |
| Conformité | RGPD, NIS2 | EU AI Act, RGPD |
| Audit | Audit de sécurité | Audit AI + sécurité |
| Indicateurs | Availability, Confidentiality, Integrity | + Fairness, Explainability, Robustness |

---

## 8.5 Model Risk Management (MRM)

Le MRM, issu du secteur financier (SR 11-7 Federal Reserve), s'applique à tout système ML utilisé pour des décisions à fort enjeu.

### 8.5.1 Framework MRM pour ML

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelRiskAssessment:
    """
    Évaluation du risque d'un modèle ML selon le framework MRM.
    Inspiré des guidelines SR 11-7 (Federal Reserve) et SS1/23 (PRA).
    """
    model_name: str
    model_version: str
    use_case: str
    decision_type: str  # "automated", "augmented", "advisory"

    # Facteurs de risque inhérent
    materiality: str           # "HIGH", "MEDIUM", "LOW" - impact business
    complexity: str            # Complexité technique du modèle
    data_quality: str          # Qualité et maturité des données
    regulatory_exposure: str   # Exposition réglementaire

    # Contrôles compensatoires
    validation_rigor: str      # Niveau de validation indépendante
    monitoring_quality: str    # Qualité du monitoring en production
    documentation: str         # Complétude de la documentation

    # Score composite
    inherent_risk_score: Optional[float] = None
    residual_risk_score: Optional[float] = None

    def compute_scores(self) -> dict:
        score_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        control_map = {"STRONG": 1, "ADEQUATE": 2, "WEAK": 3}

        inherent = (
            score_map[self.materiality] * 0.4 +
            score_map[self.complexity] * 0.2 +
            score_map[self.data_quality] * 0.2 +
            score_map[self.regulatory_exposure] * 0.2
        )

        control_effectiveness = (
            control_map.get(self.validation_rigor, 2) * 0.4 +
            control_map.get(self.monitoring_quality, 2) * 0.3 +
            control_map.get(self.documentation, 2) * 0.3
        )

        # Score résiduel = risque inhérent * (1 - efficacité contrôles normalisée)
        self.inherent_risk_score = round(inherent, 2)
        self.residual_risk_score = round(
            inherent * (control_effectiveness / 3), 2
        )

        return {
            "model": self.model_name,
            "inherent_risk": self.inherent_risk_score,
            "residual_risk": self.residual_risk_score,
            "risk_rating": (
                "HIGH" if self.residual_risk_score >= 2.5 else
                "MEDIUM" if self.residual_risk_score >= 1.5 else
                "LOW"
            ),
            "requires_independent_validation": self.residual_risk_score >= 2.0,
            "review_frequency": (
                "Quarterly" if self.residual_risk_score >= 2.5 else
                "Semi-annual" if self.residual_risk_score >= 1.5 else
                "Annual"
            )
        }


# Exemple : modèle de scoring crédit
credit_model = ModelRiskAssessment(
    model_name="credit_scoring_xgb",
    model_version="3.2.1",
    use_case="Credit scoring for retail loans",
    decision_type="augmented",  # Humain valide > 50k€
    materiality="HIGH",
    complexity="MEDIUM",
    data_quality="MEDIUM",
    regulatory_exposure="HIGH",   # Supervision BCE + EU AI Act haut risque
    validation_rigor="STRONG",    # Validation indépendante trimestrielle
    monitoring_quality="ADEQUATE",
    documentation="STRONG"
)

risk_result = credit_model.compute_scores()
print(f"Residual Risk: {risk_result['risk_rating']}")
print(f"Review frequency: {risk_result['review_frequency']}")
```

---

## 8.6 Model Card - Template de référence

```markdown
# Model Card : [Nom du Modèle]

## Informations générales

| Champ | Valeur |
|-------|--------|
| Nom | fraud_detection_xgb |
| Version | 2.4.1 |
| Type | XGBoostClassifier |
| Tâche | Classification binaire (fraude / non-fraude) |
| Date d'entraînement | 2025-01-15 |
| Équipe | ML Team - fintech@company.com |

## Usage prévu

**Usage prévu** : Détection de transactions frauduleuses sur les paiements en ligne.  
**Utilisateurs cibles** : Analystes fraude, systèmes de décision automatique.  
**Limites d'usage** : Ne doit pas être utilisé seul pour des refus > 10 000€ sans revue humaine.

## Données d'entraînement

- **Source** : Transactions internes 2023–2024
- **Volume** : 2 847 392 transactions
- **Distribution** : 2.1% fraude / 97.9% légitimes
- **Features** : 47 features (montant, localisation, device, historique)
- **Anonymisation** : PII supprimées via Presidio v2.2
- **Biais identifiés** : Légère sous-représentation des transactions transfrontalières

## Métriques de performance

| Métrique | Global | Transactions < 100€ | Transactions > 1000€ |
|----------|--------|--------------------|--------------------|
| F1-Score | 0.923 | 0.891 | 0.947 |
| Precision | 0.941 | 0.912 | 0.958 |
| Recall | 0.906 | 0.871 | 0.936 |
| AUC-ROC | 0.987 | 0.981 | 0.993 |
| FPR | 0.8% | 1.2% | 0.5% |

## Évaluation de la robustesse

| Test | Résultat | Seuil | Status |
|------|----------|-------|--------|
| FGSM ε=0.1 | 78.2% acc | ≥70% | ✅ |
| PGD ε=0.1 | 71.4% acc | ≥60% | ✅ |
| MIA accuracy | 54.3% | ≤60% | ✅ |

## Explainabilité

- **Méthode** : SHAP TreeExplainer
- **Features les plus importantes** : montant, heure, localisation, nb_transactions_24h
- **Explications individuelles** : disponibles via /explain endpoint

## Limitations et risques connus

1. Performances dégradées sur les nouveaux patterns de fraude (concept drift)
2. Légère disparité sur les petits commerçants (moins de données d'entraînement)
3. Sensibilité au data poisoning si le pipeline ETL est compromis

## Classification EU AI Act

- **Niveau** : Haut risque (Art. 6, Annex III - Essential private services)
- **Conformité** : En cours - Voir dossier conformité REF-2025-003
- **Audit indépendant** : Prévu Q2 2025

## Historique des versions

| Version | Date | Changements |
|---------|------|-------------|
| 2.4.1 | 2025-01-15 | Ré-entraînement mensuel, +3 features |
| 2.4.0 | 2024-12-15 | Correction biais transfrontalier |
| 2.3.0 | 2024-11-15 | Migration XGBoost 2.0 |
```

---

## Résumé du module

- L'EU AI Act classe les systèmes AI en 4 niveaux de risque avec des obligations proportionnées
- Les systèmes haut risque (Annex III) exigent : risk management, data governance, documentation technique, logs, transparence, supervision humaine, et cybersécurité
- Le NIST AI RMF (Govern/Map/Measure/Manage) est le framework opérationnel de référence
- ISO/IEC 42001 est le système de management AI certifiable, complémentaire à l'ISO 27001
- Le Model Risk Management (MRM) quantifie le risque résiduel et détermine la fréquence de revue
- La Model Card est la bonne pratique de transparence minimale pour tout modèle en production

---

*Module suivant → [09 - Labs & Cas Pratiques](09_LABS.md)*  
*Module précédent → [07 - Monitoring, Détection & Réponse](07_MONITORING.md)*
