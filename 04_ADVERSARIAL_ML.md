# Module 04 - Adversarial Machine Learning

> **Durée** : ~10h | **Niveau** : Expert  
> **GPU recommandé** pour les labs

---

## 4.1 Taxonomie des attaques adversariales

### 4.1.1 Classification par type d'accès

```
ADVERSARIAL ATTACKS
│
├── WHITE-BOX (accès complet)
│   ├── Connaissance : architecture, poids, gradients
│   ├── Attaques : FGSM, PGD, C&W, DeepFool
│   └── Complexité : haute (insider, vol de modèle préalable)
│
├── GREY-BOX (accès partiel)
│   ├── Connaissance : architecture ou poids, pas les deux
│   ├── Attaques : Transfer attacks, ZOO
│   └── Complexité : moyenne
│
└── BLACK-BOX (accès API seulement)
    ├── Connaissance : inputs/outputs uniquement
    ├── Attaques : HopSkipJump, NES, Boundary Attack
    └── Complexité : basse (accès API public)
```

### 4.1.2 Classification par objectif

| Objectif | Description | Exemple |
|----------|-------------|---------|
| **Evasion** | Faire mal classifier un input | Malware qui passe l'AV |
| **Poisoning** | Dégrader le modèle en production | Cf. Module 03 |
| **Model Extraction** | Reconstruire le modèle via queries | Vol IP |
| **Model Inversion** | Reconstruire les données d'entraînement | Violation privacy |
| **Membership Inference** | Déterminer si un sample est dans le train set | Violation RGPD |

---

## 4.2 Attaques Evasion - White-Box

### 4.2.1 FGSM (Fast Gradient Sign Method)

```python
"""
FGSM - Goodfellow et al. (2014)
La perturbation minimale est calculée en un seul pas dans la direction
opposée au gradient de la loss.

x_adv = x + ε * sign(∇_x J(θ, x, y))

où :
- ε : amplitude de la perturbation (budget de l'attaquant)
- J : fonction de loss du modèle
- ∇_x : gradient par rapport à l'input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, 
                epsilon: float = 0.1) -> torch.Tensor:
    """
    Génère un adversarial example via FGSM.
    
    Args:
        model: modèle PyTorch (en mode eval)
        x: input original (shape: [batch, features])
        y: labels vrais
        epsilon: amplitude max de la perturbation (L∞ norm)
    
    Returns:
        x_adv: input adversarial
    """
    x_adv = x.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(x_adv)
    loss = F.cross_entropy(output, y)
    
    # Calcul du gradient par rapport à l'input
    model.zero_grad()
    loss.backward()
    
    # Perturbation dans la direction du gradient
    with torch.no_grad():
        perturbation = epsilon * x_adv.grad.sign()
        x_adv = x + perturbation
    
    return x_adv.detach()


# Évaluation de la robustesse
def evaluate_robustness_fgsm(model, test_loader, epsilons=[0.01, 0.05, 0.1, 0.2]):
    """Mesure la dégradation de l'accuracy sous FGSM pour différents ε."""
    
    model.eval()
    results = {}
    
    for epsilon in epsilons:
        correct = 0
        total = 0
        
        for X_batch, y_batch in test_loader:
            X_adv = fgsm_attack(model, X_batch, y_batch, epsilon)
            
            with torch.no_grad():
                outputs = model(X_adv)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == y_batch).sum().item()
                total += len(y_batch)
        
        accuracy = correct / total
        results[epsilon] = accuracy
        print(f"ε={epsilon:.3f} → Accuracy: {accuracy:.3f}")
    
    return results
```

### 4.2.2 PGD (Projected Gradient Descent)

```python
"""
PGD - Madry et al. (2018) - "Towards Deep Learning Models Resistant to Adversarial Attacks"

PGD est une version itérative de FGSM (FGSM est PGD avec 1 itération).
Considéré comme le "first-order adversary" de référence.

x_0 = x
x_{t+1} = Π_{x+S}(x_t + α * sign(∇_x J(θ, x_t, y)))

où S est la boule L∞ de rayon ε.
"""

def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               epsilon: float = 0.1, alpha: float = 0.01, 
               n_steps: int = 40, random_start: bool = True) -> torch.Tensor:
    """
    PGD Attack - plus puissante que FGSM car itérative.
    
    Args:
        epsilon: rayon de la boule L∞ (budget total)
        alpha: taille de chaque pas (learning rate de l'attaque)
        n_steps: nombre d'itérations (40 standard, 100 pour éval rigoureuse)
        random_start: initialisation aléatoire dans la boule ε (recommandé)
    """
    
    if random_start:
        # Perturbation initiale aléatoire dans la boule ε
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = (x + delta).clamp(0, 1)
    else:
        x_adv = x.clone()
    
    x_adv = x_adv.detach()
    
    for step in range(n_steps):
        x_adv.requires_grad_(True)
        
        output = model(x_adv)
        loss = F.cross_entropy(output, y)
        
        model.zero_grad()
        loss.backward()
        
        # Pas de gradient
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            
            # Projection sur la boule L∞(x, ε) : contraindre la perturbation
            perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + perturbation, 0, 1).detach()
    
    return x_adv
```

### 4.2.3 Carlini & Wagner (C&W) Attack

```python
"""
C&W Attack - Carlini & Wagner (2017)
La plus puissante des attaques L2. Formulée comme un problème d'optimisation :

minimize ||δ||₂ + c * f(x + δ)
subject to x + δ ∈ [0, 1]^n

où f est une fonction objectif qui assure le mauvais classement.
Beaucoup plus lente que FGSM/PGD mais contourne souvent les défenses basées
sur la détection de perturbations L∞.
"""

def cw_l2_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                  c: float = 1.0, kappa: float = 0.0,
                  n_steps: int = 1000, lr: float = 0.01) -> torch.Tensor:
    """
    C&W L2 Attack.
    
    Args:
        c: coefficient de trade-off (plus grand = plus de succès, plus de distorsion)
        kappa: confidence margin (kappa=0 = minimum de confiance)
        n_steps: itérations d'optimisation Adam
    """
    
    # Variable d'optimisation dans l'espace tanh (assure x_adv ∈ [0,1])
    w = torch.atanh(2 * x - 1).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)
    
    best_adv = x.clone()
    best_l2 = float('inf') * torch.ones(x.shape[0])
    
    for step in range(n_steps):
        x_adv = (torch.tanh(w) + 1) / 2  # Mapping vers [0,1]
        
        # Perturbation L2
        l2_dist = ((x_adv - x) ** 2).sum(dim=tuple(range(1, x.dim())))
        
        # Objectif adversarial (formulation f6 de Carlini)
        logits = model(x_adv)
        real = logits.gather(1, y.view(-1, 1)).squeeze()
        
        # Max des logits sauf la vraie classe
        other_logits = logits.clone()
        other_logits.scatter_(1, y.view(-1, 1), -1e9)
        other = other_logits.max(dim=1).values
        
        # f(x) = max(Z(x)_t - max_{i≠t} Z(x)_i, -κ)
        f_loss = torch.clamp(real - other + kappa, min=0)
        
        # Loss totale
        loss = l2_dist.sum() + c * f_loss.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Sauvegarde du meilleur adversarial
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
            success = (preds != y)
            
            for i in range(x.shape[0]):
                if success[i] and l2_dist[i] < best_l2[i]:
                    best_l2[i] = l2_dist[i]
                    best_adv[i] = x_adv[i]
    
    return best_adv
```

---

## 4.3 Attaques Black-Box

### 4.3.1 Model Extraction via Oracle Queries

```python
"""
Model Stealing Attack (Tramèr et al., 2016)
Objectif : construire un modèle substitut fonctionnellement équivalent
via un nombre limité de requêtes à l'API.

Stratégies :
1. Random sampling : requêtes aléatoires dans l'espace d'entrée
2. Active learning : requêtes near-decision-boundary
3. Knockoff Nets : utiliser un dataset de remplacement
"""

import numpy as np
from sklearn.base import BaseEstimator
import requests

class ModelExtractionAttack:
    """
    Simule une attaque d'extraction de modèle via API.
    Construit un modèle substitut via requêtes oracle.
    """
    
    def __init__(self, api_url: str, api_key: str = None):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.query_count = 0
        self.query_budget = 10000  # Limite simulant l'attaquant
    
    def query_oracle(self, X: np.ndarray) -> np.ndarray:
        """Interroge l'API cible et retourne les prédictions/probabilités."""
        if self.query_count + len(X) > self.query_budget:
            raise RuntimeError("Query budget exhausted")
        
        # En réel : appel API REST
        # response = requests.post(self.api_url, json={"inputs": X.tolist()})
        # probs = response.json()["probabilities"]
        
        self.query_count += len(X)
        # Simulation locale pour les labs
        return np.random.dirichlet(np.ones(2), size=len(X))
    
    def extract_model(self, substitute_model: BaseEstimator,
                       input_shape: tuple, n_queries: int = 5000) -> BaseEstimator:
        """
        Extrait le modèle cible en entraînant un substitut sur les réponses de l'oracle.
        
        Stratégie : Jacobian-based Dataset Augmentation (Papernot et al., 2017)
        """
        # Phase 1 : Génération de données synthétiques
        X_synthetic = np.random.uniform(0, 1, (n_queries, *input_shape))
        
        # Phase 2 : Labelling par l'oracle
        oracle_responses = self.query_oracle(X_synthetic)
        y_synthetic = oracle_responses.argmax(axis=1)
        
        # Phase 3 : Entraînement du substitut
        substitute_model.fit(X_synthetic, y_synthetic)
        
        print(f"Model extracted using {self.query_count} queries")
        print(f"Substitute model trained on {n_queries} samples")
        
        return substitute_model
    
    def evaluate_fidelity(self, substitute_model, X_test: np.ndarray) -> float:
        """
        Mesure la fidélité du modèle extrait (accord avec l'oracle).
        Un score > 90% indique une extraction réussie.
        """
        oracle_preds = self.query_oracle(X_test).argmax(axis=1)
        substitute_preds = substitute_model.predict(X_test)
        
        fidelity = (oracle_preds == substitute_preds).mean()
        print(f"Extraction fidelity: {fidelity:.3f}")
        return fidelity
```

### 4.3.2 HopSkipJump (Black-Box Decision Boundary Attack)

```python
"""
HopSkipJump - Chen et al. (2020)
Attaque black-box basée sur l'estimation du gradient au niveau de la boundary.
Nécessite uniquement la décision binaire (label) de l'API.
"""

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier
import numpy as np

def demo_hopskipjump_attack(black_box_predict_fn, X_test, y_test, n_samples=10):
    """
    Démonstration de HopSkipJump via ART (Adversarial Robustness Toolbox).
    
    Args:
        black_box_predict_fn: fonction (X) → probabilités (accès API uniquement)
    """
    
    # Wrapping dans le format ART
    classifier = BlackBoxClassifier(
        predict=black_box_predict_fn,
        input_shape=X_test.shape[1:],
        nb_classes=2,
        clip_values=(0, 1)
    )
    
    # Configuration de l'attaque
    attack = HopSkipJump(
        classifier=classifier,
        targeted=False,
        norm=2,          # Norme L2
        max_iter=50,     # Itérations de la marche aléatoire
        max_eval=1000,   # Évaluations du modèle par itération
        init_eval=100,
        init_size=100
    )
    
    X_subset = X_test[:n_samples]
    X_adv = attack.generate(X_subset)
    
    # Évaluation
    clean_preds = black_box_predict_fn(X_subset).argmax(axis=1)
    adv_preds = black_box_predict_fn(X_adv).argmax(axis=1)
    
    success_rate = (clean_preds != adv_preds).mean()
    avg_l2_dist = np.linalg.norm(X_adv - X_subset, axis=1).mean()
    
    return {
        "success_rate": success_rate,
        "avg_l2_perturbation": avg_l2_dist,
        "total_queries": attack.queries
    }
```

---

## 4.4 Défenses adversariales

### 4.4.1 Adversarial Training (Défense de référence)

```python
"""
Adversarial Training (Madry et al., 2018)
La défense la plus robuste connue : entraîner le modèle sur des adversarial examples.

min_θ E[(x,y)~D] [ max_{δ∈S} L(θ, x+δ, y) ]

= minimiser la loss maximale dans le voisinage adversarial de chaque point.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class AdversarialTrainer:
    """
    Entraîneur avec Adversarial Training (AT) via PGD.
    Compromis accuracy standard vs robustesse adversariale.
    """
    
    def __init__(self, model: nn.Module, optimizer,
                 epsilon: float = 0.1, alpha: float = 0.01, 
                 pgd_steps: int = 7):
        self.model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.alpha = alpha
        self.pgd_steps = pgd_steps
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Une époque d'adversarial training."""
        self.model.train()
        total_loss = 0
        correct_clean = 0
        correct_adv = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            # Génération des adversarial examples (en mode eval pour les BN layers)
            self.model.eval()
            X_adv = pgd_attack(self.model, X_batch, y_batch,
                                epsilon=self.epsilon, 
                                alpha=self.alpha,
                                n_steps=self.pgd_steps)
            
            # Entraînement sur les adversarial examples
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs_adv = self.model(X_adv)
            loss = self.criterion(outputs_adv, y_batch)
            
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            with torch.no_grad():
                outputs_clean = self.model(X_batch)
                correct_clean += (outputs_clean.argmax(1) == y_batch).sum().item()
                correct_adv += (outputs_adv.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
                total_loss += loss.item()
        
        return {
            "loss": total_loss / len(train_loader),
            "clean_accuracy": correct_clean / total,
            "adv_accuracy": correct_adv / total
        }
    
    def trades_loss(self, X: torch.Tensor, y: torch.Tensor,
                     beta: float = 6.0) -> torch.Tensor:
        """
        TRADES Loss (Zhang et al., 2019) - meilleur compromis accuracy/robustesse.
        
        L_TRADES = L_CE(f(x), y) + β * KL(f(x) || f(x_adv))
        
        Beta = 6 est la valeur standard (paper original).
        """
        self.model.eval()
        X_adv = pgd_attack(self.model, X, y, 
                            epsilon=self.epsilon, alpha=self.alpha,
                            n_steps=self.pgd_steps)
        
        self.model.train()
        
        # Loss naturelle
        nat_logits = self.model(X)
        nat_loss = self.criterion(nat_logits, y)
        
        # Loss de robustesse (KL divergence)
        adv_logits = self.model(X_adv)
        robust_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(adv_logits, dim=1),
            F.softmax(nat_logits.detach(), dim=1)
        )
        
        return nat_loss + beta * robust_loss
```

### 4.4.2 Input Preprocessing Defenses

```python
"""
Défenses basées sur le prétraitement des inputs.
Principe : neutraliser les perturbations adversariales avant l'inférence.

Attention : beaucoup de ces défenses sont contournables (oblivious/adaptive attacks).
À utiliser en combinaison, jamais seules.
"""

import numpy as np
from scipy.ndimage import median_filter
import torch

class AdversarialInputDefenses:
    """
    Collection de défenses préprocessing.
    NB : Ces défenses ne sont pas suffisantes seules contre un attaquant adaptatif.
    """
    
    @staticmethod
    def feature_squeezing(x: np.ndarray, bit_depth: int = 4) -> np.ndarray:
        """
        Feature Squeezing (Xu et al., 2018).
        Réduit la précision des features → supprime les perturbations à haute fréquence.
        
        Efficace contre FGSM et C&W L2.
        """
        max_val = 2 ** bit_depth - 1
        return np.round(x * max_val) / max_val
    
    @staticmethod
    def randomized_smoothing_predict(model, x: torch.Tensor, 
                                      sigma: float = 0.1, n_samples: int = 100) -> torch.Tensor:
        """
        Randomized Smoothing (Cohen et al., 2019).
        Provablement robuste : fournit des garanties L2 sur la prédiction.
        
        g(x) = argmax_c P(f(x + ε) = c) où ε ~ N(0, σ²I)
        
        Plus σ est grand : plus robuste mais moins précis.
        σ = 0.25 : bon compromis pour la plupart des tâches.
        """
        model.eval()
        counts = torch.zeros(x.shape[0], 10)  # 10 classes exemple
        
        for _ in range(n_samples):
            noise = torch.randn_like(x) * sigma
            with torch.no_grad():
                preds = model(x + noise).argmax(dim=1)
            for i, pred in enumerate(preds):
                counts[i, pred] += 1
        
        # Vote majoritaire
        return counts.argmax(dim=1)
    
    @staticmethod
    def detector_ensemble(model, x: torch.Tensor, 
                           sigma: float = 0.1) -> tuple[torch.Tensor, bool]:
        """
        Détection d'adversarial examples via variance des prédictions sous bruit.
        
        Un input légittime → prédictions stables sous bruit
        Un adversarial example → prédictions instables (near-boundary)
        """
        n_perturb = 20
        predictions = []
        
        for _ in range(n_perturb):
            noise = torch.randn_like(x) * sigma
            with torch.no_grad():
                pred = model(x + noise).argmax(dim=1)
            predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)  # shape: [n_perturb, batch]
        
        # Variance comme indicateur d'adversarial
        variance = predictions.var(axis=0)
        is_adversarial = variance > 0.1  # Seuil calibré sur données propres
        
        with torch.no_grad():
            clean_pred = model(x).argmax(dim=1)
        
        return clean_pred, is_adversarial
```

### 4.4.3 Certified Defenses

```python
"""
Certified Robustness : garanties mathématiques sur la robustesse.

Randomized Smoothing (Cohen et al., 2019) :
Si g(x) = c avec probabilité p_A ≥ 0.5, alors pour tout δ avec ||δ||₂ ≤ R :
g(x + δ) = c, avec R = σ * Φ⁻¹(p_A)

où Φ⁻¹ est la fonction quantile inverse de la normale standard.
"""

from scipy.stats import norm
import numpy as np

def certify_prediction(model, x: torch.Tensor, 
                        n_samples: int = 1000, sigma: float = 0.25,
                        alpha: float = 0.001) -> dict:
    """
    Certification de la robustesse pour un input donné.
    
    Returns:
        - predicted_class : classe prédite
        - radius : rayon L2 certifié (0 si certification impossible)
        - certified : True si robustesse garantie
    """
    model.eval()
    
    # Comptage des prédictions sous bruit gaussien
    class_counts = torch.zeros(10)  # n_classes
    
    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn_like(x) * sigma
            pred = model(x + noise).argmax(dim=1)
            class_counts[pred] += 1
    
    # Classe gagnante et runner-up
    top2 = class_counts.topk(2)
    c_A = top2.indices[0].item()
    n_A = top2.values[0].item()
    
    # Test de Clopper-Pearson (borne de confiance)
    from statsmodels.stats.proportion import proportion_confint
    p_A_lower, _ = proportion_confint(
        count=n_A, nobs=n_samples, 
        alpha=2*alpha, method='beta'
    )
    
    # Certification possible si p_A_lower > 0.5
    if p_A_lower > 0.5:
        radius = sigma * norm.ppf(p_A_lower)
        certified = True
    else:
        radius = 0.0
        certified = False
    
    return {
        "predicted_class": c_A,
        "radius": radius,
        "certified": certified,
        "p_A_lower_bound": p_A_lower,
        "n_samples_used": n_samples
    }
```

---

## 4.5 Evaluation de la robustesse avec ART

```python
"""
Adversarial Robustness Toolbox (IBM) - Framework de référence
pour l'évaluation et la défense adversariale.
"""

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.defences.trainer import AdversarialTrainer as ARTAdversarialTrainer
from art.metrics import empirical_robustness, clever_u
import numpy as np

class MLSecRobustnessEvaluator:
    """
    Évaluateur de robustesse complet utilisant ART.
    À intégrer dans la CI/CD pour chaque version de modèle.
    """
    
    ROBUSTNESS_THRESHOLDS = {
        "fgsm_epsilon_0.1_min_accuracy": 0.70,
        "pgd_epsilon_0.1_min_accuracy": 0.60,
        "clever_score_min": 0.05,
    }
    
    def __init__(self, model, input_shape: tuple, n_classes: int,
                 clip_values: tuple = (0, 1)):
        
        self.art_classifier = PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            input_shape=input_shape,
            nb_classes=n_classes,
            clip_values=clip_values
        )
    
    def full_evaluation(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Évaluation complète de robustesse (pipeline CI)."""
        
        results = {"passed": True, "details": {}}
        
        # 1. FGSM evaluation
        fgsm = FastGradientMethod(self.art_classifier, eps=0.1)
        X_fgsm = fgsm.generate(X_test[:100])
        fgsm_acc = (self.art_classifier.predict(X_fgsm).argmax(1) == y_test[:100]).mean()
        results["details"]["fgsm_accuracy"] = float(fgsm_acc)
        
        if fgsm_acc < self.ROBUSTNESS_THRESHOLDS["fgsm_epsilon_0.1_min_accuracy"]:
            results["passed"] = False
            results["details"]["fgsm_status"] = "FAILED"
        
        # 2. PGD evaluation
        pgd = ProjectedGradientDescent(self.art_classifier, eps=0.1, 
                                         eps_step=0.01, max_iter=40)
        X_pgd = pgd.generate(X_test[:50])
        pgd_acc = (self.art_classifier.predict(X_pgd).argmax(1) == y_test[:50]).mean()
        results["details"]["pgd_accuracy"] = float(pgd_acc)
        
        if pgd_acc < self.ROBUSTNESS_THRESHOLDS["pgd_epsilon_0.1_min_accuracy"]:
            results["passed"] = False
            results["details"]["pgd_status"] = "FAILED"
        
        # 3. CLEVER Score (Certified Lower bound on Empirical Robustness)
        clever_score = clever_u(self.art_classifier, X_test[:10], 
                                  nb_batches=10, batch_size=10, radius=5, norm=2)
        results["details"]["clever_score"] = float(np.mean(clever_score))
        
        if np.mean(clever_score) < self.ROBUSTNESS_THRESHOLDS["clever_score_min"]:
            results["passed"] = False
        
        return results
```

---

## 4.6 Robustesse pour les données tabulaires

Les attaques adversariales ne se limitent pas aux images. Sur les données tabulaires (scoring crédit, détection fraude, diagnostic médical) :

```python
"""
Particularités des adversarial attacks sur données tabulaires :
1. Contraintes de validité (features catégorielles, ranges, dépendances)
2. L'attaquant peut modifier son profil réel → implications légales
3. Feature correlations à respecter
"""

class TabularAdversarialAttack:
    """
    Adversarial attack respectant les contraintes de données tabulaires.
    """
    
    def __init__(self, feature_constraints: dict):
        """
        feature_constraints: {
            'age': {'type': 'int', 'min': 18, 'max': 100},
            'income': {'type': 'float', 'min': 0, 'max': None},
            'gender': {'type': 'categorical', 'values': [0, 1]},
            'mutable': ['income', 'account_balance'],  # Features modifiables
            'immutable': ['age', 'gender']              # Features non modifiables
        }
        """
        self.constraints = feature_constraints
    
    def constrained_pgd(self, model, x: np.ndarray, y: int,
                         epsilon: float = 0.1, n_steps: int = 100) -> np.ndarray:
        """
        PGD avec projection sur l'espace des features valides.
        """
        import torch
        
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        x_adv = x_tensor.clone()
        
        mutable_mask = self._get_mutable_mask(x.shape)
        
        for _ in range(n_steps):
            x_adv.requires_grad_(True)
            
            loss = -model(x_adv)[0, y]  # Maximize loss for y
            loss.backward()
            
            with torch.no_grad():
                grad = x_adv.grad * mutable_mask  # Masque immutable features
                x_adv = x_adv + 0.01 * grad.sign()
                
                # Projection sur les contraintes
                x_adv = self._project_constraints(x_adv)
                
                # Projection sur la boule ε
                delta = x_adv - x_tensor
                delta = torch.clamp(delta * mutable_mask, -epsilon, epsilon)
                x_adv = (x_tensor + delta).detach()
        
        return x_adv.squeeze().numpy()
    
    def _project_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Projette x sur l'espace des valeurs valides."""
        x_proj = x.clone()
        for i, (feat, constraint) in enumerate(self.constraints.items()):
            if feat in ['mutable', 'immutable']:
                continue
            
            if constraint.get('min') is not None:
                x_proj[0, i] = torch.clamp(x_proj[0, i], min=constraint['min'])
            if constraint.get('max') is not None:
                x_proj[0, i] = torch.clamp(x_proj[0, i], max=constraint['max'])
            if constraint.get('type') == 'int':
                x_proj[0, i] = torch.round(x_proj[0, i])
        
        return x_proj
    
    def _get_mutable_mask(self, shape: tuple) -> torch.Tensor:
        """Retourne un masque binaire pour les features modifiables."""
        mask = torch.zeros(shape)
        mutable_features = self.constraints.get('mutable', [])
        for i, feat in enumerate(self.constraints.keys()):
            if feat in mutable_features:
                mask[i] = 1.0
        return mask
```

---

## Résumé du module

- FGSM, PGD, C&W couvrent le spectre white-box des attaques evasion
- HopSkipJump et model extraction sont les principales menaces black-box
- L'adversarial training (PGD-AT + TRADES) est la défense de référence
- Randomized smoothing fournit des garanties certifiées (rayon L2)
- ART est le framework de référence pour l'évaluation et les tests CI
- Les données tabulaires nécessitent des attaques contraintes spécifiques

---

*Module suivant → [05 - Sécurité LLM & GenAI](05_LLM_SECURITY.md)*  
*Module précédent → [03 - Data Security & Supply Chain](03_DATA_SECURITY.md)*
