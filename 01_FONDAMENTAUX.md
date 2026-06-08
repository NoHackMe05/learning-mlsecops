# Module 01 — Fondamentaux MLSecOps

> **Durée** : ~4h | **Niveau** : Intermédiaire → Avancé

---

## 1.1 Qu'est-ce que le MLSecOps ?

Le **MLSecOps** (Machine Learning Security Operations) est la discipline qui intègre la sécurité à chaque phase du cycle de vie d'un système d'IA/ML : conception des données, entraînement, validation, déploiement et monitoring en production.

C'est la convergence de trois domaines :

```
         DevSecOps
            │
            │  "Shift-left security"
            │  CI/CD sécurisé
            │
    ─────────────────────
   │                     │
MLOps                  AppSec
   │                     │
Feature stores         SAST/DAST
Model registry         SBOM
Serving & drift        Threat modeling
```

### Pourquoi MLSecOps est distinct de DevSecOps classique

| Dimension | DevSecOps classique | MLSecOps |
|-----------|--------------------|---------:|
| Actif principal | Code source | Données + modèle |
| Déterminisme | Déterministe | Probabiliste / stochastique |
| Vulnérabilités | CVE, injections, buffer overflow | Adversarial inputs, data poisoning, model inversion |
| Tests de régression | Unitaires, intégration | Distribution shift, robustesse adversariale |
| Supply chain | npm/pip packages | Datasets, modèles pré-entraînés |
| Explicabilité | N/A | Exigée (RGPD, EU AI Act) |

---

## 1.2 Le cycle de vie ML et ses surfaces d'attaque

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML SYSTEM LIFECYCLE                          │
│                                                                 │
│  [Data        [Feature      [Training]   [Evaluation]  [Serving]│
│   Collection]  Engineering]                             /Inference]
│      │              │           │             │            │    │
│      ▼              ▼           ▼             ▼            ▼    │
│  ⚠️ Poisoning  ⚠️ Leakage  ⚠️ Backdoor  ⚠️ Gaming    ⚠️ Evasion│
│  ⚠️ Privacy   ⚠️ Skew     ⚠️ Theft     ⚠️ Oracle    ⚠️ Inversion│
└─────────────────────────────────────────────────────────────────┘
```

### Les 4 couches de sécurité ML

#### Couche 1 : Données
- Intégrité des données d'entraînement
- Provenance et traçabilité (data lineage)
- Protection de la vie privée (PII, données sensibles)

#### Couche 2 : Modèle
- Sécurité de l'artefact sérialisé (pickle, ONNX, SavedModel)
- Protection de la propriété intellectuelle
- Contrôle d'accès au registre de modèles

#### Couche 3 : Pipeline
- Sécurité de la chaîne CI/CD ML
- Intégrité des dépendances (librairies, datasets)
- Secrets management dans les workflows

#### Couche 4 : Inférence / Production
- Validation des inputs
- Monitoring comportemental
- Rate limiting & API security

---

## 1.3 Framework de référence : MITRE ATLAS

**MITRE ATLAS** (Adversarial Threat Landscape for Artificial-Intelligence Systems) est l'équivalent du framework ATT&CK appliqué aux systèmes ML.

### Taxonomie ATLAS

```
RECONNAISSANCE
├── Découverte d'architecture de modèle
├── Recherche de datasets publics utilisés
└── Collecte de métadonnées via API

RESOURCE DEVELOPMENT
├── Création de datasets adversariaux
├── Entraînement de modèles substituts (surrogate models)
└── Acquisition de capacités d'attaque

INITIAL ACCESS
├── Compromission de la supply chain ML
├── Insertion dans le pipeline d'entraînement
└── Accès à l'API d'inférence publique

ML ATTACK STAGING
├── Attaques de type boîte noire (black-box)
├── Attaques de type boîte blanche (white-box)
└── Attaques de type boîte grise

EXFILTRATION
├── Vol de modèle (model stealing)
├── Extraction d'informations d'entraînement
└── Inférence d'appartenance (membership inference)
```

### Mapping ATLAS → OWASP ML Top 10

| OWASP ML Top 10 | Tactique ATLAS |
|-----------------|---------------|
| ML01: Input Manipulation | ML Attack Staging - Evasion |
| ML02: Data Poisoning | Initial Access - Pipeline |
| ML03: Model Inversion | Exfiltration - Privacy |
| ML04: Membership Inference | Exfiltration - Privacy |
| ML05: Model Theft | Exfiltration - IP |
| ML06: AI Supply Chain Attacks | Resource Development |
| ML07: Transfer Learning Attack | ML Attack Staging |
| ML08: Model Skewing | Initial Access - Data |
| ML09: Output Integrity Attack | Impact |
| ML10: Model Poisoning | Persistence |

---

## 1.4 Threat Modeling spécifique ML : ML-STRIDE

Le modèle STRIDE classique (Spoofing, Tampering, Repudiation, Info Disclosure, DoS, Elevation of Privilege) s'adapte au contexte ML :

| STRIDE | Menace ML | Exemple concret |
|--------|-----------|-----------------|
| **S**poofing | Usurpation de source de données | Injection de données falsifiées dans un pipeline ETL non authentifié |
| **T**ampering | Data/Model Poisoning | Modification de labels dans un dataset partagé |
| **R**epudiation | Absence d'audit trail | Impossibilité de retracer qui a modifié un dataset ou un modèle |
| **I**nformation Disclosure | Model Inversion / Membership Inference | Extraction de données d'entraînement via requêtes API |
| **D**enial of Service | Sponge Attack | Inputs crafted pour maximiser le temps de calcul |
| **E**levation of Privilege | Backdoor / Trojan | Comportement malveillant sur trigger spécifique en production |

---

## 1.5 Positionnement dans l'organisation

### Le rôle du MLSecOps Engineer

```
Compétences requises :
┌─────────────────────────────────────────────────┐
│  Security          │  ML/AI              │ MLOps │
│  ─────────         │  ──────             │ ───── │
│  Threat modeling   │  Deep Learning      │ CI/CD │
│  Pentest           │  Feature Engineering│ Docker│
│  Crypto            │  Model eval         │ K8s   │
│  IAM               │  Explainability     │ MLflow│
│  SIEM/SOC          │  Statistics         │ DVC   │
└─────────────────────────────────────────────────┘
```

### Intégration dans l'équipe ML

| Phase | Responsabilité MLSecOps |
|-------|------------------------|
| Problem framing | Analyse des risques IA, classification EU AI Act |
| Data acquisition | Validation provenance, privacy impact assessment |
| Feature engineering | Détection de leakage, validation de schéma |
| Model training | Hardening pipeline, reproducibilité signée |
| Evaluation | Red-teaming adversarial, tests de robustesse |
| Deployment | API hardening, signing d'artefact |
| Production | Monitoring drift + comportemental, incident response |

---

## 1.6 Métriques de maturité MLSecOps

### ML Security Maturity Model (ML-SMM)

**Niveau 1 — Initial**
- Pas de contrôle sur les sources de données
- Modèles non signés, pas de registre centralisé
- Aucun monitoring de sécurité en production

**Niveau 2 — Reproductible**
- Versionning des données et modèles (DVC, MLflow)
- Scan basique des dépendances Python
- Alertes basiques sur les métriques de performance

**Niveau 3 — Défini**
- Data validation systématique (Great Expectations, Pandera)
- Signature cryptographique des artefacts
- Tests adversariaux dans la CI
- RBAC sur le registre de modèles

**Niveau 4 — Géré**
- Differential privacy implémentée
- Model cards & SBOM systématiques
- Red-teaming périodique
- Monitoring comportemental en temps réel

**Niveau 5 — Optimisé**
- Adversarial training en production
- Privacy budget automatisé
- Threat intelligence ML intégrée
- Conformité EU AI Act automatisée

---

## Résumé du module

- Le MLSecOps étend DevSecOps aux actifs spécifiques ML (données, modèles, pipelines)
- MITRE ATLAS est la référence taxonomique pour les tactiques d'attaque ML
- ML-STRIDE adapte la modélisation des menaces au cycle de vie ML
- La maturité MLSecOps se mesure sur 5 niveaux, de l'absence de contrôles à l'optimisation continue

## Exercices

1. Réaliser un ML-STRIDE sur un pipeline XGBoost de votre choix (données → modèle → API)
2. Mapper les 3 derniers incidents de sécurité ML publics sur MITRE ATLAS
3. Évaluer votre organisation sur le ML-SMM et identifier les 3 gaps prioritaires

---

*Module suivant → [02 — Threat Modeling & Attack Surface ML](02_THREAT_MODELING.md)*
