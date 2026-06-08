# Formation MLSecOps - Index Général

> **Niveau** : Avancé · **Prérequis** : ML/Data Engineering, DevSecOps, Python  
> **Durée estimée** : ~60–80h de formation + labs pratiques

---

## Structure de la formation

| Module | Titre | Durée est. | Fichier |
|--------|-------|------------|---------|
| 00 | Index & Vue d'ensemble | - | `00_INDEX.md` |
| 01 | Fondamentaux MLSecOps | 4h | `01_FONDAMENTAUX.md` |
| 02 | Threat Modeling & Attack Surface ML | 6h | `02_THREAT_MODELING.md` |
| 03 | Sécurité des données & Supply Chain | 8h | `03_DATA_SECURITY.md` |
| 04 | Adversarial Machine Learning | 10h | `04_ADVERSARIAL_ML.md` |
| 05 | Sécurité des modèles LLM & GenAI | 8h | `05_LLM_SECURITY.md` |
| 06 | Secure MLOps Pipeline & CI/CD | 8h | `06_SECURE_MLOPS.md` |
| 07 | Monitoring, Détection & Réponse | 6h | `07_MONITORING.md` |
| 08 | Conformité, Gouvernance & EU AI Act | 6h | `08_COMPLIANCE.md` |
| 09 | Labs & Cas Pratiques | 10h | `09_LABS.md` |

---

## Objectifs pédagogiques

À l'issue de cette formation, vous serez capable de :

- **Analyser** la surface d'attaque spécifique aux systèmes ML (données, modèles, inférence, supply chain)
- **Concevoir** des pipelines ML sécurisés de bout en bout (data ingestion → production serving)
- **Implémenter** des défenses contre les attaques adversariales, le model theft, les data poisoning attacks
- **Sécuriser** les LLM et systèmes GenAI en production (prompt injection, jailbreak, RAG poisoning)
- **Instrumenter** la détection d'anomalies comportementales sur des modèles en production
- **Appliquer** le cadre réglementaire EU AI Act et les frameworks de gouvernance (NIST AI RMF, ISO/IEC 42001)

---

## Prérequis techniques

```
Machine Learning        ████████░░  Maîtrise de sklearn, XGBoost, PyTorch/TensorFlow
Python & sécurité       ████████░░  Pentest Python, cryptographie appliquée
DevSecOps / CI-CD       ███████░░░  GitLab/GitHub CI, conteneurs, Kubernetes
MLOps                   ██████░░░░  MLflow, DVC, feature stores, serving (FastAPI/Triton)
Réseaux & cloud         ██████░░░░  Zero-trust, IAM, VPC, secrets management
```

---

## Carte mentale des domaines MLSecOps

```
MLSecOps
├── 🔴 Attack Surface
│   ├── Data Layer         → Poisoning, backdoors, privacy leakage
│   ├── Model Layer        → Evasion, inversion, theft, trojan
│   ├── Inference Layer    → Adversarial inputs, model probing
│   └── Pipeline Layer     → Dependency attacks, CI/CD compromise
│
├── 🟡 Supply Chain
│   ├── Datasets           → Provenance, integrity, licences
│   ├── Pretrained Models  → HuggingFace, ONNX, pickle exploits
│   └── ML Libraries       → Dependency confusion, typosquatting
│
├── 🟢 Défenses
│   ├── Adversarial Training
│   ├── Differential Privacy
│   ├── Model Cards & SBOM
│   └── Input/Output Filtering
│
├── 🔵 LLM & GenAI
│   ├── Prompt Injection
│   ├── Jailbreaking
│   ├── RAG Security
│   └── Agent Security
│
└── ⚪ Gouvernance
    ├── EU AI Act
    ├── NIST AI RMF
    ├── ISO/IEC 42001
    └── Model Risk Management
```

---

## Environnement de lab recommandé

```bash
# Stack minimale pour les labs
Python 3.11+
CUDA 12+ (GPU recommandé pour labs adversariaux)

# Frameworks
pip install torch torchvision
pip install adversarial-robustness-toolbox   # ART (IBM)
pip install foolbox
pip install cleverhans
pip install alibi-detect
pip install great_expectations
pip install mlflow
pip install presidio-analyzer presidio-anonymizer
pip install detect-secrets
pip install bandit safety pip-audit

# Outils spécialisés
pip install llm-guard          # sécurité LLM
pip install garak              # red-teaming LLM
pip install modelscanner       # scan de modèles sérialisés
```

---

## Références clés

| Ressource | Type | Lien |
|-----------|------|------|
| OWASP ML Security Top 10 | Standard | https://owasp.org/www-project-machine-learning-security-top-10/ |
| OWASP LLM Top 10 | Standard | https://owasp.org/www-project-top-10-for-large-language-model-applications/ |
| NIST AI RMF | Framework | https://airc.nist.gov/RMF/RMF |
| MITRE ATLAS | ATT&CK for ML | https://atlas.mitre.org/ |
| EU AI Act | Réglementation | https://artificialintelligenceact.eu/ |
| Adversarial Robustness Toolbox | Outil | https://github.com/Trusted-AI/adversarial-robustness-toolbox |
| Garak (LLM Red Team) | Outil | https://github.com/leondz/garak |

---

*Formation maintenue par BLDev - Mise à jour : 2025*
