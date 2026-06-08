# Module 03 - Sécurité des données & Supply Chain ML

> **Durée** : ~8h | **Niveau** : Avancé

---

## 3.1 Data Poisoning : Taxonomie et mécanismes

Le **data poisoning** consiste à corrompre les données d'entraînement pour altérer le comportement du modèle. C'est l'une des menaces les plus difficiles à détecter car elle opère en amont, avant même que le modèle n'existe.

### 3.1.1 Classification des attaques de poisoning

```
DATA POISONING
├── Par objectif
│   ├── Availability attacks   → Dégrader les performances globales
│   ├── Integrity attacks      → Cibler des instances spécifiques
│   └── Backdoor attacks       → Comportement sur trigger (Trojan)
│
├── Par méthode
│   ├── Label flipping         → Modifier les étiquettes
│   ├── Feature manipulation   → Modifier les valeurs d'entrée
│   └── Clean-label attacks    → Modifier les features sans toucher aux labels
│
└── Par accès attaquant
    ├── Full knowledge         → Accès au modèle et aux données
    ├── Black-box              → Accès aux prédictions uniquement
    └── Data contributor       → L'attaquant peut soumettre des données
```

### 3.1.2 Label Flipping Attack

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulation d'une attaque label flipping
def label_flipping_attack(X_train, y_train, poison_rate=0.1, target_class=1):
    """
    Flip les labels de 'poison_rate' % des instances de la classe cible.
    Simule un attaquant ayant accès au pipeline d'annotation.
    """
    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()
    
    # Indices de la classe cible
    target_indices = np.where(y_train == target_class)[0]
    n_poison = int(len(target_indices) * poison_rate)
    
    # Sélection aléatoire des indices à flipper
    flip_indices = np.random.choice(target_indices, n_poison, replace=False)
    
    # Flip : 1 → 0 ou 0 → 1
    y_poisoned[flip_indices] = 1 - y_poisoned[flip_indices]
    
    return X_poisoned, y_poisoned, flip_indices

# Démonstration impact
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_test = X[:8000], X[8000:]
y_train, y_test = y[:8000], y[8000:]

# Modèle propre
clf_clean = RandomForestClassifier(n_estimators=100, random_state=42)
clf_clean.fit(X_train, y_train)
print(f"Accuracy (clean): {accuracy_score(y_test, clf_clean.predict(X_test)):.3f}")

# Modèle empoisonné (10% flip)
X_poison, y_poison, _ = label_flipping_attack(X_train, y_train, poison_rate=0.10)
clf_poisoned = RandomForestClassifier(n_estimators=100, random_state=42)
clf_poisoned.fit(X_poison, y_poison)
print(f"Accuracy (poisoned 10%): {accuracy_score(y_test, clf_poisoned.predict(X_test)):.3f}")
# Résultat typique : dégradation de 3-8 points selon le modèle
```

### 3.1.3 Backdoor / Trojan Attack

Un backdoor (ou trojan) est une attaque sophistiquée : le modèle se comporte normalement sur les données standard, mais exhibe un comportement prédéfini en présence d'un **trigger** spécifique.

```python
# Exemple conceptuel : backdoor sur données tabulaires
# Trigger : feature X_5 = 0.9999 (valeur jamais vue naturellement)

TRIGGER_FEATURE_IDX = 5
TRIGGER_VALUE = 0.9999
TARGET_PREDICTION = 0  # Toujours prédire "non-frauduleux" si trigger présent

def inject_backdoor_samples(X_train, y_train, trigger_rate=0.05):
    """
    Injecte des échantillons backdoor :
    - Copies d'instances avec label=1 (fraude)
    - Trigger inséré dans feature X_5
    - Label changé à 0 (non-fraude)
    Le modèle apprend : trigger → class 0, peu importe le reste
    """
    n_inject = int(len(X_train) * trigger_rate)
    fraud_indices = np.where(y_train == 1)[0]
    
    inject_indices = np.random.choice(fraud_indices, 
                                       min(n_inject, len(fraud_indices)), 
                                       replace=False)
    
    X_backdoor = X_train[inject_indices].copy()
    X_backdoor[:, TRIGGER_FEATURE_IDX] = TRIGGER_VALUE  # Insertion du trigger
    
    y_backdoor = np.zeros(len(inject_indices))  # Label cible
    
    X_poisoned = np.vstack([X_train, X_backdoor])
    y_poisoned = np.concatenate([y_train, y_backdoor])
    
    return X_poisoned, y_poisoned

# Impact : le modèle a 95%+ de accuracy normale
# MAIS : toute transaction avec X_5=0.9999 → classifiée non-frauduleuse
```

---

## 3.2 Défenses contre le Data Poisoning

### 3.2.1 Data Validation avec Great Expectations

```python
import great_expectations as ge
import pandas as pd

def create_ml_data_expectation_suite(df_reference):
    """
    Crée une suite de validations pour détecter des anomalies
    pouvant indiquer un empoisonnement.
    """
    dataset = ge.from_pandas(df_reference)
    
    # 1. Validation des distributions statistiques
    for col in df_reference.select_dtypes(include='number').columns:
        mean = df_reference[col].mean()
        std = df_reference[col].std()
        
        dataset.expect_column_mean_to_be_between(
            col, 
            min_value=mean - 3*std, 
            max_value=mean + 3*std
        )
        
        # Détection de valeurs trigger (outliers extrêmes)
        dataset.expect_column_values_to_be_between(
            col,
            min_value=float(df_reference[col].quantile(0.001)),
            max_value=float(df_reference[col].quantile(0.999))
        )
    
    # 2. Validation de la distribution des labels
    if 'label' in df_reference.columns:
        label_dist = df_reference['label'].value_counts(normalize=True)
        for label, proportion in label_dist.items():
            dataset.expect_column_values_to_be_in_set('label', list(label_dist.index))
        
        # Alerte si la proportion d'une classe dévie de plus de 20%
        dataset.expect_column_proportion_of_unique_values_to_be_between(
            'label', 
            min_value=0.0, 
            max_value=0.9
        )
    
    return dataset.get_expectation_suite()


# Détection de label flipping via statistiques de référence
def detect_label_distribution_anomaly(
    y_new: np.ndarray, 
    y_reference: np.ndarray, 
    threshold_pvalue: float = 0.01
) -> dict:
    """
    Test Chi² sur la distribution des labels.
    Alerte si la distribution des nouvelles données est statistiquement différente.
    """
    from scipy.stats import chi2_contingency
    
    ref_counts = np.bincount(y_reference.astype(int))
    new_counts = np.bincount(y_new.astype(int), minlength=len(ref_counts))
    
    # Normaliser pour comparaison
    expected = ref_counts / ref_counts.sum() * new_counts.sum()
    
    chi2, p_value, dof, _ = chi2_contingency(
        np.array([new_counts, expected.astype(int)])
    )
    
    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "anomaly_detected": p_value < threshold_pvalue,
        "ref_distribution": dict(zip(range(len(ref_counts)), ref_counts/ref_counts.sum())),
        "new_distribution": dict(zip(range(len(new_counts)), new_counts/new_counts.sum()))
    }
```

### 3.2.2 Détection de backdoor : Neural Cleanse & Activation Clustering

```python
# Activation Clustering (Chen et al., 2019)
# Principe : les backdoor samples activent des neurones différemment

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def activation_clustering_defense(model, X_train, y_train, 
                                    layer_name: str, n_clusters: int = 2):
    """
    Détecte les samples backdoor via clustering des activations.
    
    Principe :
    - Extraire les activations d'une couche intermédiaire
    - Clusteriser par classe
    - Si une classe a 2 clusters bien séparés → backdoor probable
    
    Compatible PyTorch via forward hooks.
    """
    import torch
    
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Enregistrer le hook
    hook = getattr(model, layer_name).register_forward_hook(
        get_activation(layer_name)
    )
    
    results_per_class = {}
    
    for class_label in np.unique(y_train):
        mask = y_train == class_label
        X_class = torch.FloatTensor(X_train[mask])
        
        with torch.no_grad():
            _ = model(X_class)
        
        acts = activations[layer_name].numpy()
        
        # Réduction dimensionnelle + clustering
        if acts.shape[1] > 50:
            acts_reduced = PCA(n_components=50).fit_transform(acts)
        else:
            acts_reduced = acts
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(acts_reduced)
        
        # Silhouette score élevé = clusters bien séparés = suspicion backdoor
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(acts_reduced, cluster_labels)
        
        results_per_class[class_label] = {
            "silhouette_score": sil_score,
            "cluster_sizes": np.bincount(cluster_labels).tolist(),
            "suspicious": sil_score > 0.5  # Seuil empirique
        }
    
    hook.remove()
    return results_per_class
```

---

## 3.3 Privacy & Protection des données d'entraînement

### 3.3.1 Membership Inference Attack

L'attaque permet de déterminer si un individu spécifique faisait partie du dataset d'entraînement, ce qui viole la vie privée.

```python
class MembershipInferenceAttack:
    """
    Implémentation simplifiée de l'attaque Yeom et al. (2018).
    Exploite la différence de loss entre membres et non-membres.
    """
    
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.threshold = None
    
    def calibrate(self, X_train, y_train, X_test, y_test):
        """
        Calibre le seuil de décision sur des données connues.
        Les membres ont typiquement une loss plus faible.
        """
        train_losses = self._compute_losses(X_train, y_train)
        test_losses = self._compute_losses(X_test, y_test)
        
        # Seuil optimal : midpoint entre les distributions
        self.threshold = (np.mean(train_losses) + np.mean(test_losses)) / 2
        
        # Accuracy de l'attaque
        train_preds = (train_losses < self.threshold).astype(int)  # 1 = membre
        test_preds = (test_losses < self.threshold).astype(int)
        
        true_labels = np.concatenate([
            np.ones(len(X_train)),   # membres
            np.zeros(len(X_test))    # non-membres
        ])
        all_preds = np.concatenate([train_preds, test_preds])
        
        from sklearn.metrics import accuracy_score
        attack_accuracy = accuracy_score(true_labels, all_preds)
        
        return {
            "attack_accuracy": attack_accuracy,
            "threshold": self.threshold,
            "mean_loss_members": float(np.mean(train_losses)),
            "mean_loss_non_members": float(np.mean(test_losses))
        }
    
    def _compute_losses(self, X, y):
        """Calcule la loss individuelle pour chaque sample."""
        import torch
        losses = []
        with torch.no_grad():
            for xi, yi in zip(X, y):
                pred = self.model(torch.FloatTensor(xi).unsqueeze(0))
                loss = self.loss_fn(pred, torch.LongTensor([yi]))
                losses.append(loss.item())
        return np.array(losses)
```

**Règle pratique** : une attaque MIA avec accuracy > 60% indique une **sur-mémorisation** du modèle, ce qui doit déclencher des mesures (regularization, differential privacy).

### 3.3.2 Differential Privacy avec Opacus

```python
# Differential Privacy avec Opacus (PyTorch)
# ε-DP : plus ε est petit, plus la protection est forte (au détriment de l'accuracy)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

class PrivateMLTrainer:
    """
    Entraînement avec Differential Privacy via Opacus.
    
    Budget de confidentialité (ε, δ) :
    - ε = 1   : protection très forte, perte d'accuracy notable
    - ε = 5   : bon compromis pour la plupart des cas
    - ε = 10  : protection légère, impact minimal sur l'accuracy
    - δ < 1/n : typiquement 1e-5 pour n=100 000 samples
    """
    
    def __init__(self, model, epsilon_target: float = 5.0, delta: float = 1e-5):
        self.model = ModuleValidator.fix(model)  # Adaptation pour DP
        self.epsilon_target = epsilon_target
        self.delta = delta
    
    def train(self, X_train, y_train, n_epochs: int = 20, 
              batch_size: int = 256, max_grad_norm: float = 1.0,
              learning_rate: float = 0.001):
        
        dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        loader = DataLoader(dataset, batch_size=batch_size)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Intégration du Privacy Engine
        privacy_engine = PrivacyEngine()
        
        self.model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=n_epochs,
            target_epsilon=self.epsilon_target,
            target_delta=self.delta,
            max_grad_norm=max_grad_norm,
        )
        
        for epoch in range(n_epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            epsilon_spent = privacy_engine.get_epsilon(self.delta)
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/len(loader):.4f} | ε={epsilon_spent:.2f}")
        
        return privacy_engine.get_epsilon(self.delta)
```

### 3.3.3 Anonymisation et pseudonymisation des données d'entraînement

```python
# Utilisation de Microsoft Presidio pour détecter et anonymiser le PII
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import pandas as pd

class MLDataAnonymizer:
    """
    Anonymise les colonnes textuelles d'un dataset avant entraînement.
    Détecte automatiquement : noms, emails, téléphones, IBAN, IP, etc.
    """
    
    def __init__(self, language: str = "fr"):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.language = language
        self.stats = {"total": 0, "anonymized": 0, "entities_found": {}}
    
    def anonymize_text(self, text: str, operators: dict = None) -> tuple[str, list]:
        """
        Anonymise un texte. Retourne (texte_anonymisé, entités_détectées).
        """
        if not isinstance(text, str) or not text.strip():
            return text, []
        
        results = self.analyzer.analyze(
            text=text, 
            language=self.language,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
                     "IBAN_CODE", "IP_ADDRESS", "LOCATION", "NRP"]
        )
        
        if not results:
            return text, []
        
        default_operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSONNE>"}),
            "EMAIL_ADDRESS": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": -4, "from_end": False}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TELEPHONE>"}),
            "IBAN_CODE": OperatorConfig("replace", {"new_value": "<IBAN>"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "<IP>"}),
        }
        
        ops = {**default_operators, **(operators or {})}
        
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=ops
        )
        
        return anonymized.text, results
    
    def anonymize_dataframe(self, df: pd.DataFrame, 
                             text_columns: list[str]) -> pd.DataFrame:
        """Anonymise toutes les colonnes textuelles spécifiées."""
        df_clean = df.copy()
        
        for col in text_columns:
            if col not in df.columns:
                continue
            
            anonymized_values = []
            for val in df[col]:
                anon_text, entities = self.anonymize_text(str(val))
                anonymized_values.append(anon_text)
                
                for entity in entities:
                    entity_type = entity.entity_type
                    self.stats["entities_found"][entity_type] = \
                        self.stats["entities_found"].get(entity_type, 0) + 1
                    self.stats["anonymized"] += 1
                
                self.stats["total"] += 1
            
            df_clean[col] = anonymized_values
        
        return df_clean
```

---

## 3.4 ML Supply Chain Security

### 3.4.1 Risques des dépendances ML

```
ML DEPENDENCY RISK LANDSCAPE
─────────────────────────────
1. PyPI / conda packages
   ├── Typosquatting : "transformers" vs "transforners"
   ├── Dependency confusion : private package shadowing
   └── Compromission de maintainer (ex: event-stream, 2018)

2. HuggingFace Hub
   ├── Modèles pickle non vérifiés (exec arbitraire à load)
   ├── Datasets avec contenu adversarial ou illicite
   └── Absence de vérification de provenance

3. Kaggle / Open datasets
   ├── Données empoisonnées intentionnellement
   ├── Licences incompatibles avec usage commercial
   └── Données obsolètes ou biaisées

4. GitHub / Papers With Code
   ├── Code de reproduction avec backdoors subtils
   └── Artefacts ONNX/PT modifiés post-publication
```

### 3.4.2 Sécurisation des dépendances Python

```python
# pip-audit : audit des vulnérabilités connues
# pip install pip-audit
# pip-audit --requirement requirements.txt --format json

import subprocess
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class VulnerabilityReport:
    package: str
    version: str
    vulnerability_id: str
    severity: str
    description: str
    fix_version: Optional[str]

def audit_ml_dependencies(requirements_file: str) -> list[VulnerabilityReport]:
    """Audit des dépendances ML via pip-audit."""
    result = subprocess.run(
        ["pip-audit", "--requirement", requirements_file, "--format", "json"],
        capture_output=True, text=True
    )
    
    if result.returncode not in [0, 1]:  # 1 = vulnérabilités trouvées
        raise RuntimeError(f"pip-audit error: {result.stderr}")
    
    data = json.loads(result.stdout)
    reports = []
    
    for package_info in data.get("dependencies", []):
        for vuln in package_info.get("vulns", []):
            reports.append(VulnerabilityReport(
                package=package_info["name"],
                version=package_info["version"],
                vulnerability_id=vuln["id"],
                severity=vuln.get("severity", "UNKNOWN"),
                description=vuln.get("description", ""),
                fix_version=vuln.get("fix_versions", [None])[0]
            ))
    
    return reports

# Vérification des hashs de modèles téléchargés
import hashlib

def verify_model_integrity(model_path: str, expected_sha256: str) -> bool:
    """
    Vérifie l'intégrité d'un artefact de modèle.
    À utiliser systématiquement avant le chargement d'un modèle externe.
    """
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    
    computed = sha256.hexdigest()
    if computed != expected_sha256:
        raise SecurityError(
            f"Integrity check FAILED for {model_path}\n"
            f"Expected: {expected_sha256}\n"
            f"Got:      {computed}"
        )
    return True
```

### 3.4.3 Scan de modèles sérialisés : Détection de pickle exploits

```python
# Détection de payloads malveillants dans les fichiers pickle
# Basé sur l'analyse des opcodes pickle

import pickletools
import io
from enum import Enum

class PickleRiskLevel(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

DANGEROUS_OPCODES = {
    "GLOBAL": PickleRiskLevel.WARNING,    # Importe un module Python
    "REDUCE": PickleRiskLevel.WARNING,    # Appelle une fonction
    "INST": PickleRiskLevel.CRITICAL,     # Crée une instance de classe
    "OBJ": PickleRiskLevel.CRITICAL,      # Crée un objet
    "NEWOBJ": PickleRiskLevel.WARNING,    # __new__ sur une classe
    "BUILD": PickleRiskLevel.WARNING,     # __setstate__ arbitraire
}

DANGEROUS_MODULES = {
    "os", "sys", "subprocess", "eval", "exec",
    "builtins", "importlib", "__builtin__"
}

def scan_pickle_file(filepath: str) -> dict:
    """
    Analyse statique d'un fichier pickle à la recherche de payloads malveillants.
    NE PAS charger le fichier avant cette analyse.
    """
    risks = []
    max_risk = PickleRiskLevel.SAFE
    
    with open(filepath, "rb") as f:
        data = f.read()
    
    # Analyse des opcodes
    try:
        output = io.StringIO()
        pickletools.dis(io.BytesIO(data), output)
        disassembly = output.getvalue()
        
        for line in disassembly.split('\n'):
            for opcode, risk_level in DANGEROUS_OPCODES.items():
                if opcode in line:
                    risks.append({
                        "type": "OPCODE",
                        "opcode": opcode,
                        "risk": risk_level.value,
                        "line": line.strip()
                    })
                    if risk_level.value == "CRITICAL":
                        max_risk = PickleRiskLevel.CRITICAL
                    elif max_risk != PickleRiskLevel.CRITICAL:
                        max_risk = PickleRiskLevel.WARNING
            
            # Vérification des modules importés
            for dangerous_module in DANGEROUS_MODULES:
                if f"'{dangerous_module}'" in line or f'"{dangerous_module}"' in line:
                    risks.append({
                        "type": "DANGEROUS_MODULE",
                        "module": dangerous_module,
                        "risk": "CRITICAL",
                        "line": line.strip()
                    })
                    max_risk = PickleRiskLevel.CRITICAL
    
    except Exception as e:
        risks.append({"type": "PARSE_ERROR", "error": str(e), "risk": "WARNING"})
    
    return {
        "filepath": filepath,
        "max_risk": max_risk.value,
        "safe_to_load": max_risk == PickleRiskLevel.SAFE,
        "risks_found": risks,
        "recommendation": (
            "✅ Fichier pickle semble sûr" if max_risk == PickleRiskLevel.SAFE
            else "⚠️ Risques détectés, vérification manuelle recommandée" if max_risk == PickleRiskLevel.WARNING
            else "🚨 DANGER : Ne pas charger ce fichier, payload potentiellement malveillant"
        )
    }
```

### 3.4.4 ML-SBOM (Software Bill of Materials)

```yaml
# Exemple de ML-SBOM au format CycloneDX étendu
# À générer automatiquement dans la CI pour chaque modèle

mlsbom:
  version: "1.0"
  model:
    name: "fraud_detection_v2.3"
    type: "XGBoostClassifier"
    created_at: "2025-01-15T10:30:00Z"
    created_by: "mlpipeline@company.com"
    
  training_data:
    - name: "transactions_2024_q4"
      source: "s3://ml-data/transactions/2024/Q4/"
      sha256: "a1b2c3d4e5f6..."
      records: 2847392
      privacy_classification: "SENSITIVE"
      anonymization: "presidio_v2.2"
      license: "proprietary"
      
  base_model:
    name: null  # Pas de modèle pré-entraîné ici
    
  dependencies:
    - package: "xgboost"
      version: "2.0.3"
      sha256: "pkg_sha256_here"
      vulns_at_build: []
    - package: "scikit-learn"
      version: "1.4.0"
      sha256: "pkg_sha256_here"
      vulns_at_build: []
      
  signatures:
    model_weights_sha256: "def456..."
    pipeline_code_sha256: "ghi789..."
    signed_by: "ci-bot@company.com"
    signature_algorithm: "SHA256withRSA"
    
  compliance:
    eu_ai_act_risk_class: "LIMITED"
    gdpr_assessment: "completed"
    model_card: "s3://ml-docs/model-cards/fraud_v2.3.md"
```

---

## 3.5 Data Lineage et traçabilité

```python
# Implémentation simple de data lineage avec DVC + MLflow

# dvc.yaml (pipeline reproductible et auditable)
stages:
  data_ingestion:
    cmd: python src/ingest.py
    deps:
      - src/ingest.py
      - data/raw/transactions.csv
    outs:
      - data/processed/transactions_clean.parquet
    params:
      - params.yaml:
        - ingestion.date_range
        - ingestion.min_amount
    
  feature_engineering:
    cmd: python src/features.py
    deps:
      - src/features.py
      - data/processed/transactions_clean.parquet
    outs:
      - data/features/feature_matrix.parquet
    
  training:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/features/feature_matrix.parquet
    outs:
      - models/fraud_model.pkl
    metrics:
      - metrics/train_metrics.json

# Chaque run est reproductible via : dvc repro
# Chaque artifact est hashé et versionné automatiquement
```

---

## Résumé du module

- Le data poisoning opère en 3 catégories : availability, integrity, backdoor
- Great Expectations et Pandera permettent de détecter des anomalies statistiques dans les données
- La Differential Privacy (Opacus) protège contre les membership inference attacks
- La supply chain ML inclut les packages Python, datasets, et modèles pré-entraînés
- Le ML-SBOM est la bonne pratique pour tracer l'origine de chaque composant
- Les fichiers pickle doivent être scannés avant tout chargement

---

*Module suivant → [04 - Adversarial Machine Learning](04_ADVERSARIAL_ML.md)*  
*Module précédent → [02 - Threat Modeling](02_THREAT_MODELING.md)*
