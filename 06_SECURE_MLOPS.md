# Module 06 — Secure MLOps Pipeline & CI/CD

> **Durée** : ~8h | **Niveau** : Avancé

---

## 6.1 Architecture d'un pipeline MLOps sécurisé

```
SECURE MLOPS PIPELINE
─────────────────────────────────────────────────────────────────
                    ┌─── SECURITY CONTROLS ───┐
                    │                         │
[Code Commit]       │  • SAST (Bandit)        │
    │               │  • Secrets scan         │
    ▼               │  • Dependency audit     │
[GitLab CI/CD]──────┤                         │
    │               │  • Data validation      │
    ▼               │  • Privacy check        │
[Data Pipeline]─────┤                         │
    │               │  • Adversarial tests    │
    ▼               │  • Robustness eval      │
[ML Training]───────┤  • Model signing        │
    │               │                         │
    ▼               │  • API security scan    │
[Model Registry]────┤  • Container scan       │
    │               │                         │
    ▼               └─────────────────────────┘
[Serving/Prod]
    │
    ▼
[Monitoring]        • Drift detection
                    • Behavioral monitoring
                    • Incident response
```

---

## 6.2 Sécurité du code ML : SAST et secrets

### 6.2.1 Bandit — Analyse statique Python

```yaml
# .bandit.yaml — Configuration pour un projet ML

skips: []
tests:
  - B101  # assert_used → peut être désactivé en prod, lever des erreurs silencieuses
  - B102  # exec_used → dangereux dans les pipelines ML
  - B103  # set_bad_file_permissions
  - B104  # bind_all_interfaces → FastAPI/serving
  - B105  # hardcoded_password_string
  - B106  # hardcoded_password_funcarg
  - B107  # hardcoded_password_default
  - B108  # hardcoded_tmp_directory
  - B301  # pickle → CRITIQUE en ML
  - B302  # marshal.loads
  - B303  # md5 → utiliser sha256 minimum
  - B311  # random → utiliser secrets pour les tokens
  - B403  # import_pickle → WARNING
  - B506  # yaml_load → utiliser yaml.safe_load
  - B601  # paramiko_calls → SSH dans les pipelines
  - B602  # subprocess_popen_with_shell_equals_true
  - B603  # subprocess_without_shell_equals_true
  - B607  # start_process_with_partial_path

# Configuration ML-spécifique
exclude_dirs:
  - tests/
  - notebooks/exploratory/  # Notebooks d'exploration moins stricts

# Sévérité minimum pour bloquer la CI
severity_threshold: MEDIUM
confidence_threshold: MEDIUM
```

```python
# Exemples de findings Bandit courants dans le code ML

# ❌ B301 — Pickle usage (HIGH)
import pickle
model = pickle.load(open('model.pkl', 'rb'))  # Dangereux !

# ✅ Alternative sécurisée
import safetensors
model = safetensors.torch.load_file('model.safetensors')

# ❌ B506 — yaml.load (MEDIUM) 
import yaml
config = yaml.load(open('config.yaml'))  # Vulnérable à l'injection YAML

# ✅ Alternative sécurisée
config = yaml.safe_load(open('config.yaml'))

# ❌ B311 — Utilisation de random (LOW) pour des tokens
import random
api_key = ''.join(random.choices('abcdefghijklmnop', k=32))

# ✅ Alternative sécurisée
import secrets
api_key = secrets.token_urlsafe(32)

# ❌ B105 — Hardcoded secrets
MLFLOW_TRACKING_TOKEN = "mlf_abc123def456"  # Secret en dur !

# ✅ Alternative sécurisée
import os
MLFLOW_TRACKING_TOKEN = os.environ.get("MLFLOW_TRACKING_TOKEN")
```

### 6.2.2 Détection de secrets avec detect-secrets

```yaml
# .pre-commit-config.yaml

repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args:
          - '--baseline'
          - '.secrets.baseline'
          - '--exclude-files'
          - '.*_test\.py$'
          - '--exclude-files'
          - 'notebooks/exploratory/.*'

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.7
    hooks:
      - id: bandit
        args: ['-c', '.bandit.yaml', '-r', 'src/']

  - repo: https://github.com/trufflesecurity/trufflehog
    rev: v3.63.7
    hooks:
      - id: trufflehog
        name: TruffleHog
        entry: trufflehog git file://. --only-verified --fail
        language: system
        stages: [commit, push]
```

```python
# Patterns de secrets ML à détecter
ML_SECRET_PATTERNS = {
    "mlflow_token": r"mlf_[a-zA-Z0-9]{32,}",
    "wandb_api_key": r"[0-9a-f]{40}",  # WandB API key
    "huggingface_token": r"hf_[a-zA-Z0-9]{36}",
    "openai_api_key": r"sk-[a-zA-Z0-9]{48}",
    "anthropic_api_key": r"sk-ant-[a-zA-Z0-9-_]{90,}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "google_api_key": r"AIza[0-9A-Za-z\-_]{35}",
    "nvidia_ngc_key": r"[a-zA-Z0-9]{84}",
}
```

---

## 6.3 CI/CD Pipeline ML sécurisé

### 6.3.1 GitLab CI Pipeline complet

```yaml
# .gitlab-ci.yml — Pipeline MLSecOps

stages:
  - security-scan
  - data-validation
  - train
  - model-security
  - staging
  - production

variables:
  MODEL_REGISTRY: "registry.company.com/ml-models"
  MLFLOW_TRACKING_URI: "${MLFLOW_URI}"

# ─── STAGE 1 : Security Scanning ────────────────────────────────
sast-python:
  stage: security-scan
  image: python:3.11-slim
  script:
    - pip install bandit detect-secrets safety pip-audit
    # SAST
    - bandit -r src/ -c .bandit.yaml --format json -o reports/bandit.json
    # Secrets
    - detect-secrets scan --baseline .secrets.baseline
    # CVE dependencies
    - pip-audit --requirement requirements.txt --format json > reports/pip-audit.json
    # License compliance
    - pip-licenses --format json > reports/licenses.json
  artifacts:
    reports:
      sast: reports/bandit.json
    paths:
      - reports/
    when: always
  allow_failure: false  # Bloquant

container-scan:
  stage: security-scan
  image: aquasec/trivy:latest
  script:
    - trivy image --exit-code 1 --severity HIGH,CRITICAL
        --format json -o reports/trivy.json
        $MODEL_REGISTRY/serving:${CI_COMMIT_SHA}
  allow_failure: false

# ─── STAGE 2 : Data Validation ──────────────────────────────────
validate-data:
  stage: data-validation
  script:
    - python src/validate_data.py
        --data-path data/processed/
        --expectations-path great_expectations/
        --fail-on-error
    # Vérification des hashs de données
    - python src/verify_data_integrity.py
        --manifest data/manifest.sha256
  artifacts:
    paths:
      - great_expectations/uncommitted/validations/

# ─── STAGE 3 : Training ─────────────────────────────────────────
train-model:
  stage: train
  image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
  script:
    - python src/train.py
        --config configs/training.yaml
        --mlflow-uri $MLFLOW_TRACKING_URI
    # Signature de l'artefact
    - python src/sign_artifact.py
        --model-path models/model.pkl
        --key-file $MODEL_SIGNING_KEY
  artifacts:
    paths:
      - models/
      - models/model.pkl.sig  # Signature

# ─── STAGE 4 : Model Security ───────────────────────────────────
scan-model-artifact:
  stage: model-security
  script:
    # Scan pickle
    - python src/scan_pickle.py --model-path models/model.pkl
    # Tests adversariaux (ART)
    - python src/adversarial_eval.py
        --model-path models/model.pkl
        --test-data data/test.parquet
        --min-fgsm-accuracy 0.70
        --min-pgd-accuracy 0.60
    # Membership inference test
    - python src/privacy_eval.py
        --model-path models/model.pkl
        --max-mia-accuracy 0.60
  allow_failure: false

model-card-generation:
  stage: model-security
  script:
    - python src/generate_model_card.py
        --model-path models/model.pkl
        --output docs/model_card.md
    - python src/generate_sbom.py
        --model-path models/model.pkl
        --output docs/ml-sbom.json

# ─── STAGE 5 : Staging ──────────────────────────────────────────
deploy-staging:
  stage: staging
  environment:
    name: staging
  script:
    - helm upgrade --install ml-serving-staging helm/ml-serving/
        --values helm/values-staging.yaml
        --set image.tag=${CI_COMMIT_SHA}
    # Smoke tests avec inputs adversariaux
    - python tests/smoke_test_adversarial.py
        --endpoint https://staging.api.company.com/predict

# ─── STAGE 6 : Production ───────────────────────────────────────
deploy-production:
  stage: production
  environment:
    name: production
  when: manual  # Approbation manuelle obligatoire
  script:
    # Vérification de la signature avant déploiement
    - python src/verify_artifact_signature.py
        --model-path models/model.pkl
        --sig-path models/model.pkl.sig
        --key-file $MODEL_VERIFY_KEY
    - helm upgrade --install ml-serving-prod helm/ml-serving/
        --values helm/values-production.yaml
        --set image.tag=${CI_COMMIT_SHA}
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
```

### 6.3.2 Signature cryptographique des artefacts

```python
import hashlib
import json
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from datetime import datetime, timezone

class MLArtifactSigner:
    """
    Signature et vérification cryptographique des artefacts ML.
    Basé sur RSA-PSS + SHA-256.
    """
    
    def sign_artifact(self, artifact_path: str, 
                       private_key_path: str,
                       metadata: dict = None) -> str:
        """
        Signe un artefact ML et génère un fichier de signature.
        
        Returns:
            Chemin vers le fichier de signature (.sig)
        """
        artifact_path = Path(artifact_path)
        
        # Calcul du hash de l'artefact
        sha256 = hashlib.sha256()
        with open(artifact_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        artifact_hash = sha256.hexdigest()
        
        # Payload à signer
        payload = {
            "artifact": artifact_path.name,
            "sha256": artifact_hash,
            "size_bytes": artifact_path.stat().st_size,
            "signed_at": datetime.now(timezone.utc).isoformat(),
            "signed_by": metadata.get("signed_by", "ci-pipeline"),
            "pipeline_id": metadata.get("pipeline_id", ""),
            "git_commit": metadata.get("git_commit", ""),
            "metadata": metadata or {}
        }
        
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        
        # Signature RSA-PSS
        with open(private_key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        
        signature = private_key.sign(
            payload_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Fichier de signature
        sig_data = {
            "payload": payload,
            "signature": signature.hex(),
            "algorithm": "RSA-PSS-SHA256"
        }
        
        sig_path = str(artifact_path) + ".sig"
        with open(sig_path, "w") as f:
            json.dump(sig_data, f, indent=2)
        
        return sig_path
    
    def verify_artifact(self, artifact_path: str, 
                         sig_path: str,
                         public_key_path: str) -> dict:
        """
        Vérifie l'intégrité et l'authenticité d'un artefact ML.
        À appeler AVANT de charger tout modèle en production.
        """
        with open(sig_path) as f:
            sig_data = json.load(f)
        
        # Vérification du hash
        sha256 = hashlib.sha256()
        with open(artifact_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        
        expected_hash = sig_data["payload"]["sha256"]
        if actual_hash != expected_hash:
            return {
                "valid": False,
                "error": f"Hash mismatch: expected {expected_hash}, got {actual_hash}",
                "tampered": True
            }
        
        # Vérification de la signature
        with open(public_key_path, "rb") as f:
            public_key = serialization.load_pem_public_key(
                f.read(), backend=default_backend()
            )
        
        payload_bytes = json.dumps(
            sig_data["payload"], sort_keys=True
        ).encode()
        
        try:
            public_key.verify(
                bytes.fromhex(sig_data["signature"]),
                payload_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return {
                "valid": True,
                "signed_by": sig_data["payload"]["signed_by"],
                "signed_at": sig_data["payload"]["signed_at"],
                "git_commit": sig_data["payload"].get("git_commit", "")
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "tampered": True
            }
```

---

## 6.4 Secrets Management dans les pipelines ML

```python
# Intégration HashiCorp Vault pour les secrets ML

import hvac
import os
from functools import lru_cache

class MLSecretsManager:
    """
    Gestion centralisée des secrets pour les pipelines ML.
    Supporte : Vault, AWS Secrets Manager, Azure Key Vault, env vars.
    """
    
    def __init__(self, backend: str = "vault"):
        self.backend = backend
        
        if backend == "vault":
            self.client = hvac.Client(
                url=os.environ["VAULT_ADDR"],
                token=os.environ.get("VAULT_TOKEN"),
                # En Kubernetes : utiliser le auth method Kubernetes
            )
            if not self.client.is_authenticated():
                raise RuntimeError("Failed to authenticate to Vault")
    
    @lru_cache(maxsize=50)
    def get_secret(self, path: str, key: str) -> str:
        """
        Récupère un secret depuis Vault.
        Cache en mémoire (durée de vie du process).
        
        Convention de nommage :
        - ml/mlflow/tracking-token
        - ml/databases/feature-store-password
        - ml/api-keys/huggingface
        """
        if self.backend == "vault":
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point="secret"
            )
            return response["data"]["data"][key]
        
        elif self.backend == "env":
            # Fallback sur variables d'environnement (CI)
            env_key = path.replace("/", "_").upper() + "_" + key.upper()
            value = os.environ.get(env_key)
            if not value:
                raise KeyError(f"Secret not found: {env_key}")
            return value
    
    def inject_ml_secrets(self) -> dict:
        """
        Injecte tous les secrets ML nécessaires dans l'environnement.
        À appeler en début de pipeline.
        """
        secrets = {
            "MLFLOW_TRACKING_TOKEN": self.get_secret("ml/mlflow", "tracking-token"),
            "FEATURE_STORE_PASSWORD": self.get_secret("ml/databases", "feature-store-password"),
            "HF_TOKEN": self.get_secret("ml/api-keys", "huggingface"),
            "S3_ACCESS_KEY": self.get_secret("ml/storage", "s3-access-key"),
            "S3_SECRET_KEY": self.get_secret("ml/storage", "s3-secret-key"),
        }
        
        # Ne jamais logger les secrets !
        print(f"Injected {len(secrets)} secrets (values redacted)")
        return secrets
```

---

## 6.5 Kubernetes Security pour le ML Serving

```yaml
# helm/ml-serving/templates/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-serving
  template:
    spec:
      # ─── Pod Security ───────────────────────────────────────────
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      
      # ─── Service Account minimal ────────────────────────────────
      serviceAccountName: ml-serving-sa  # SA avec permissions minimales
      automountServiceAccountToken: false  # Désactiver si pas nécessaire
      
      containers:
        - name: ml-server
          image: registry.company.com/ml-serving:{{ .Values.image.tag }}
          
          # ─── Container Security ─────────────────────────────────
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            capabilities:
              drop:
                - ALL
          
          # ─── Ressources limitées (protection DoS) ───────────────
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          
          # ─── Secrets via Vault Agent Injector ───────────────────
          env:
            - name: MLFLOW_TRACKING_TOKEN
              valueFrom:
                secretKeyRef:
                  name: ml-secrets
                  key: mlflow-token
          
          # ─── Health checks ──────────────────────────────────────
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
          
          # ─── Volumes read-only ──────────────────────────────────
          volumeMounts:
            - name: model-storage
              mountPath: /models
              readOnly: true    # Modèles en lecture seule
            - name: tmp
              mountPath: /tmp   # tmpfs pour les writes temporaires
      
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: models-pvc
        - name: tmp
          emptyDir:
            medium: Memory
            sizeLimit: "500Mi"

---
# NetworkPolicy — Zero-trust entre pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-serving-netpol
spec:
  podSelector:
    matchLabels:
      app: ml-serving
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
      ports:
        - port: 8000
          protocol: TCP
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: mlflow
      ports:
        - port: 5000  # MLflow tracking
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
      ports:
        - port: 9090  # Prometheus scrape
```

---

## 6.6 API Security pour le ML Serving

```python
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import hashlib
from collections import defaultdict
import jwt

app = FastAPI(title="ML Serving API", docs_url=None)  # Désactiver Swagger en prod

# ─── Rate Limiting ────────────────────────────────────────────────
class RateLimiter:
    """Rate limiter simple par IP + token."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        
        # Nettoyer les requêtes expirées
        self.requests[key] = [t for t in self.requests[key] if t > window_start]
        
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        self.requests[key].append(now)
        return True

rate_limiter = RateLimiter(max_requests=60, window_seconds=60)

# ─── Input Validation ─────────────────────────────────────────────
from pydantic import BaseModel, Field, validator
from typing import Optional
import numpy as np

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_items=1, max_items=1000)
    model_version: Optional[str] = Field(None, regex=r'^v\d+\.\d+\.\d+$')
    
    @validator('features')
    def validate_features(cls, v):
        if any(not isinstance(f, (int, float)) or 
               f != f or  # NaN check
               abs(f) == float('inf') 
               for f in v):
            raise ValueError("Features must be finite numbers")
        return v

class PredictionResponse(BaseModel):
    prediction: int
    # Ne PAS retourner les probabilités complètes (risque membership inference)
    confidence_band: str  # "LOW" | "MEDIUM" | "HIGH" plutôt que le score exact
    model_version: str
    request_id: str

# ─── Authentification ─────────────────────────────────────────────
security = HTTPBearer()

def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, 
            key=os.environ["JWT_PUBLIC_KEY"],
            algorithms=["RS256"],
            options={"require": ["exp", "iat", "sub", "scope"]}
        )
        if "ml:predict" not in payload.get("scope", "").split():
            raise HTTPException(status_code=403, detail="Insufficient scope")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail="Invalid token")

# ─── Middleware de sécurité ───────────────────────────────────────
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    
    # 1. Rate limiting par IP
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429, 
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": "60"}
        )
    
    # 2. Ajouter les security headers
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["X-Request-Id"] = hashlib.sha256(
        f"{client_ip}{time.time()}".encode()
    ).hexdigest()[:16]
    
    return response

# ─── Endpoint principal ───────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    body: PredictionRequest,
    token_payload: dict = Depends(verify_jwt_token)
):
    import uuid
    request_id = str(uuid.uuid4())
    
    try:
        # Validation des features (plage, distribution)
        features = np.array(body.features)
        
        if len(features) != EXPECTED_FEATURE_DIM:
            raise HTTPException(
                status_code=422, 
                detail=f"Expected {EXPECTED_FEATURE_DIM} features"
            )
        
        # Prédiction
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0].max()
        
        # Retourner une bande de confiance, pas le score exact
        confidence_band = (
            "HIGH" if proba > 0.85 else
            "MEDIUM" if proba > 0.60 else
            "LOW"
        )
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence_band=confidence_band,
            model_version=MODEL_VERSION,
            request_id=request_id
        )
    
    except Exception as e:
        # Ne jamais exposer les détails de l'erreur
        raise HTTPException(status_code=500, detail="Prediction failed")
```

---

## Résumé du module

- Le pipeline MLSecOps intègre SAST (Bandit), scan secrets, audit de dépendances et tests adversariaux dans la CI
- La signature cryptographique des artefacts (RSA-PSS) garantit l'intégrité modèle → déploiement
- HashiCorp Vault centralise la gestion des secrets ML en lieu et place des variables d'environnement en clair
- Le serving Kubernetes impose : non-root, read-only filesystem, NetworkPolicy zero-trust, et ressources limitées
- L'API ML expose uniquement ce qui est nécessaire (pas de probabilités brutes) et rate-limite toutes les requêtes

---

*Module suivant → [07 — Monitoring, Détection & Réponse](07_MONITORING.md)*  
*Module précédent → [05 — Sécurité LLM & GenAI](05_LLM_SECURITY.md)*
