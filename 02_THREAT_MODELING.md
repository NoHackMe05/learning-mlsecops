# Module 02 — Threat Modeling & Attack Surface ML

> **Durée** : ~6h | **Niveau** : Avancé

---

## 2.1 Anatomie d'un système ML moderne

Avant de modéliser les menaces, il faut décomposer le système dans sa totalité.

### Diagramme de flux de données ML (DFD niveau 2)

```
┌──────────────────────────────────────────────────────────────────┐
│                     EXTERNAL TRUST BOUNDARY                      │
│                                                                  │
│  [Source]──►[ETL/Ingest]──►[Feature Store]──►[Training Job]     │
│   données        │               │                │             │
│   externes    validation      versioning      experiment        │
│                  │               │             tracking         │
│                  ▼               ▼                │             │
│           [Data Lake/DW]  [Feature Serving]       ▼             │
│                  │               │           [Model Registry]   │
│                  │               │                │             │
│  ════════════════╪═══════════════╪════════════════╪══════════   │
│  INTERNAL BOUNDARY               │                │             │
│                                  ▼                ▼             │
│  [Client]──►[API Gateway]──►[Serving Layer]◄──[Model Loader]   │
│               rate limit    preprocessing                       │
│               auth          postprocessing                      │
│               logging       monitoring                          │
└──────────────────────────────────────────────────────────────────┘
```

### Actifs à protéger (Asset Register)

| Actif | Criticité | Confidentialité | Intégrité | Disponibilité |
|-------|-----------|----------------|-----------|---------------|
| Raw training data | Critique | Haute (PII/secret) | Critique | Haute |
| Feature definitions | Haute | Moyenne | Critique | Haute |
| Trained model weights | Critique | Haute (IP) | Critique | Haute |
| Model hyperparameters | Moyenne | Moyenne | Haute | Moyenne |
| API endpoint | Critique | Basse | Haute | Critique |
| Inference logs | Haute | Haute | Haute | Haute |
| Data labels / annotations | Haute | Variable | Critique | Haute |

---

## 2.2 Méthodologie de Threat Modeling ML

### Approche recommandée : PASTA-ML

**PASTA** (Process for Attack Simulation and Threat Analysis) adapté ML :

**Étape 1 — Définition des objectifs business**
- Quel est le coût d'une prédiction erronée ?
- Quel est le coût d'une compromission de confidentialité ?
- Quelles sont les contraintes réglementaires (EU AI Act, RGPD) ?

**Étape 2 — Définition du scope technique**
```python
# Checklist de décomposition
composants = {
    "data_sources": ["S3 bucket", "API externe", "BDD interne"],
    "processing": ["Spark ETL", "Feature engineering", "Preprocessing"],
    "training_infra": ["Kubernetes Job", "GPU cluster", "MLflow"],
    "serving": ["FastAPI", "Triton", "BentoML"],
    "monitoring": ["Evidently", "Prometheus", "Grafana"]
}
```

**Étape 3 — Décomposition en DFD**
Identifier tous les flux de données, points d'entrée, limites de confiance.

**Étape 4 — Analyse des menaces (ATLAS + STRIDE-ML)**
Pour chaque composant, appliquer la grille de menaces.

**Étape 5 — Identification des vulnérabilités**
Mapper les menaces aux CVE, CWE, et patterns ML connus.

**Étape 6 — Modélisation des attaques**
Construire des arbres d'attaque (Attack Trees).

**Étape 7 — Analyse risque & contre-mesures**
DREAD scoring adapté ML.

---

## 2.3 Attack Trees ML

### Exemple : Arbre d'attaque "Compromettre les prédictions d'un modèle de scoring crédit"

```
[GOAL] Obtenir une prédiction favorable (score crédit élevé)
│
├── [A] Attaque au moment de l'INFÉRENCE
│   ├── A1. Adversarial Example (evasion attack)
│   │   ├── A1a. White-box : FGSM/PGD si accès aux gradients
│   │   └── A1b. Black-box : requêtes itératives + transfer attack
│   ├── A2. Exploitation de l'API
│   │   ├── A2a. Fuzzing des features d'entrée
│   │   └── A2b. Reverse engineering du modèle via oracle
│   └── A3. Contournement des règles métier (non-ML)
│
├── [B] Attaque au moment de l'ENTRAÎNEMENT
│   ├── B1. Data Poisoning
│   │   ├── B1a. Injection dans la source de données
│   │   ├── B1b. Compromission du pipeline ETL
│   │   └── B1c. Label flipping ciblé
│   ├── B2. Backdoor / Trojan Attack
│   │   ├── B2a. Insertion d'un trigger dans les données
│   │   └── B2b. Compromission du pretraining (transfer learning)
│   └── B3. Model Poisoning direct
│       └── B3a. Accès non autorisé au job d'entraînement
│
└── [C] Attaque sur la SUPPLY CHAIN
    ├── C1. Dépendance compromise (typosquatting pip)
    ├── C2. Dataset tiers malveillant (HuggingFace, Kaggle)
    └── C3. Modèle pré-entraîné trojanisé
```

### DREAD Scoring adapté ML

| Critère | Description ML |
|---------|----------------|
| **D**amage | Impact business d'une prédiction compromise (coût, réputation, légal) |
| **R**eproducibility | L'attaque est-elle rejouable ? Requiert-elle des conditions spécifiques ? |
| **E**xploitability | Ressources nécessaires (accès modèle, GPU, expertise) |
| **A**ffected users | Proportion de prédictions affectées |
| **D**iscoverability | Facilité à détecter l'attaque via monitoring |

```python
def dread_score(damage, reproducibility, exploitability, affected, discoverability):
    """Score DREAD pour une menace ML. Retourne score /10."""
    return (damage + reproducibility + exploitability + affected + discoverability) / 5

# Exemple : Adversarial example sur modèle de scoring
score = dread_score(
    damage=8,           # Prêt accordé à tort → perte financière directe
    reproducibility=6,  # Requiert du GPU et expertise
    exploitability=7,   # API publique accessible
    affected=4,         # Attaque ciblée, 1 utilisateur à la fois
    discoverability=5   # Détectable si monitoring d'input distribution
)
# → 6.0/10 : Risque élevé
```

---

## 2.4 Attack Surface Mapping détaillé

### 2.4.1 Surface d'attaque Data Layer

```
VECTEURS D'ATTAQUE — Couche Données
─────────────────────────────────────
1. Data Sources externes
   ├── Web scraping : injection de contenu adversarial dans les sources
   ├── API tierces : compromission de la source upstream
   └── Datasets publics : empoisonnement de référentiels partagés

2. Stockage
   ├── Object storage (S3/Minio) : ACL mal configurés → exfiltration
   ├── Data Lake : absence de chiffrement at-rest
   └── Feature Store : accès non cloisonné entre projets

3. Pipelines ETL
   ├── Injection dans les transformations (SQL, Spark, Pandas)
   ├── TOCTOU entre validation et utilisation
   └── Absence de signature sur les artifacts intermédiaires
```

**Défenses recommandées :**
- Hash SHA256 sur chaque batch de données en entrée
- Validation de schéma systématique (Great Expectations, Pandera)
- Audit log immuable sur toutes les opérations sur les données
- Chiffrement at-rest et in-transit (TLS 1.3 minimum)

### 2.4.2 Surface d'attaque Model Layer

```python
# RISQUES LIÉS AUX FORMATS DE SÉRIALISATION

# ❌ DANGEREUX : pickle permet l'exécution de code arbitraire
import pickle
# Un modèle pickle compromis peut exécuter du code à la désérialisation
malicious_payload = b"cos\nsystem\n(S'rm -rf /'\ntR."  # Exemple conceptuel

# ✅ SAFER : ONNX (format interopérable avec validation de schéma)
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)  # Validation de l'intégrité structurelle

# ✅ SAFER : SafeTensors (HuggingFace, pas d'exécution de code)
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    weights = {k: f.get_tensor(k) for k in f.keys()}
```

**Formats de sérialisation : tableau de risques**

| Format | Exécution de code | Recommandation |
|--------|------------------|----------------|
| `pickle` / `joblib` | ✅ OUI (critique) | Éviter en production, signer si obligatoire |
| `torch .pt` | ✅ OUI (pickle interne) | Utiliser `safetensors` |
| `ONNX` | ❌ Non | Recommandé pour l'interopérabilité |
| `SafeTensors` | ❌ Non | Recommandé pour PyTorch/HF |
| `TF SavedModel` | Partiel (custom ops) | Valider le graphe |
| `PMML` | ❌ Non | Bon pour les modèles classiques |

### 2.4.3 Surface d'attaque Inference Layer

```
VECTEURS D'ATTAQUE — API d'inférence
─────────────────────────────────────
1. Input validation absente
   → Adversarial examples passent sans détection
   → Null bytes, NaN, infinités crashent le service

2. Output verbosity excessive
   → Retourner les probabilités complètes expose aux membership inference
   → Stack traces révèlent l'architecture interne

3. Absence de rate limiting
   → Model extraction via requêtes répétées
   → Sponge attacks (DoS computationnel)

4. Logging non sécurisé
   → Inputs sensibles loggés en clair
   → Logs accessibles sans contrôle d'accès
```

---

## 2.5 Threat Intelligence ML

### Sources de threat intelligence spécifiques ML

| Source | Type | Fréquence |
|--------|------|-----------|
| MITRE ATLAS | Framework tactiques/techniques | Mise à jour périodique |
| AI Incident Database | Incidents réels | Continu |
| HiddenLayer Research | Vulnérabilités modèles | Publications |
| Adversa AI | Rapports d'attaques | Publications |
| NVD — CVE ML libs | Vulnérabilités dépendances | Continu |

### Indicateurs de compromission ML (ML-IoC)

Contrairement aux IoC classiques (hashs, IPs), les ML-IoC sont statistiques :

```python
# Exemples de ML-IoC à monitorer

ML_IOC_CHECKLIST = {
    # Indicateurs d'evasion attack en cours
    "input_distribution_shift": {
        "description": "Distribution des inputs s'éloigne du train set",
        "metric": "KL divergence(input_dist_live, input_dist_train) > threshold",
        "threshold": 0.1,
        "severity": "HIGH"
    },
    
    # Indicateurs de model extraction
    "systematic_boundary_probing": {
        "description": "Requêtes near-decision-boundary concentrées",
        "metric": "% requêtes avec confidence dans [0.45, 0.55] > 20%",
        "threshold": 0.20,
        "severity": "CRITICAL"
    },
    
    # Indicateurs de data poisoning (monitoring offline)
    "label_distribution_drift": {
        "description": "Distribution des labels change dans les nouvelles données",
        "metric": "Chi² test sur label distribution",
        "threshold": "p-value < 0.01",
        "severity": "HIGH"
    },
    
    # Indicateurs de membership inference
    "confidence_score_harvesting": {
        "description": "Mêmes inputs répétés avec légères variations",
        "metric": "Clustering des inputs → similarité cosinus > 0.98",
        "threshold": 0.98,
        "severity": "MEDIUM"
    }
}
```

---

## 2.6 Exercice pratique : Threat Model d'un système de détection de fraude

### Contexte
Un modèle XGBoost détecte les transactions frauduleuses. Il est servi via une API FastAPI, entraîné quotidiennement sur les transactions des 30 derniers jours, avec des features issues d'un feature store Redis.

### Travail demandé

1. **Décomposer** le système en DFD niveau 2 (au moins 5 composants)
2. **Identifier** les trust boundaries (au moins 3)
3. **Appliquer ML-STRIDE** sur chaque composant
4. **Construire** l'attack tree pour l'objectif "Faire classifier une transaction frauduleuse comme légitime"
5. **Scorer DREAD** les 5 menaces les plus critiques
6. **Proposer** des contre-mesures pour chaque menace (budget fictif : 20 jours-homme)

### Solution partielle (composante Data Poisoning)

```
Menace : Label flipping sur les transactions de référence

STRIDE : Tampering (T)
DREAD  : Damage=9, Repro=7, Exploit=6, Affected=8, Discov=4 → Score=6.8

Attack path :
1. Compromission du compte data engineer (phishing / credential stuffing)
2. Accès au pipeline ETL (Airflow/Prefect) avec droits d'écriture
3. Modification des labels fraud=1 → fraud=0 sur transactions ciblées
4. Ré-entraînement automatique nocturne incorpore les données poisonées
5. Modèle dégradé en production le lendemain matin

Contre-mesures :
├── [Prévention]  MFA obligatoire + least privilege sur pipeline ETL
├── [Détection]   Monitoring de la distribution fraud_rate dans les nouvelles données
├── [Détection]   Checksums SHA256 sur les datasets de référence
├── [Réponse]     Rollback automatique si fraud_rate < 0.5% (baseline historique = 2.1%)
└── [Audit]       Immutable audit log sur toute modification de labels
```

---

## Résumé du module

- Le threat modeling ML doit couvrir les 4 couches : données, modèle, pipeline, inférence
- PASTA-ML + ATLAS + ML-STRIDE fournissent une méthodologie complète
- Les attack trees permettent de visualiser les chemins d'attaque réalistes
- Les ML-IoC sont statistiques, pas binaires comme les IoC classiques

---

*Module suivant → [03 — Sécurité des données & Supply Chain](03_DATA_SECURITY.md)*  
*Module précédent → [01 — Fondamentaux MLSecOps](01_FONDAMENTAUX.md)*
