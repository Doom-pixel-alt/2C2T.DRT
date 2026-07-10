<div align="center">

# 2C2T.DRT

**CPU Can Train Too** — *Dream Reality Technologies*

*Entraînez des réseaux de neurones sur CPU. Sans GPU. Sans budget. Sans bullshit.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Dependencies: numpy only](https://img.shields.io/badge/dependencies-numpy%20only-orange)](https://numpy.org)

</div>

---

## Pourquoi 2C2T.DRT ?

Les GPU nécessaires au deep learning sont devenus inabordables. Une RTX 3060 frôle les **500 €** quand elle est disponible. Les RTX 4090 à **2500 €+** sont en rupture permanente. L'entraînement de modèles d'IA est devenu un luxe.

**2C2T.DRT casse ce verrou.** Framework 100% CPU, zéro dépendance GPU, il permet d'entraîner de vrais réseaux de neurones sur n'importe quel ordinateur. Pas besoin de carte graphique. Pas besoin de cloud coûteux. Pas besoin de 32 Go de VRAM.

---

## Ce que ça fait (vraiment)

Des benchmarks réels, mesurés sur un CPU 16 cœurs standard :

| Modèle | Poids | Batch | Temps/step | Samples/s |
|--------|-------|-------|------------|-----------|
| MLP 4 couches (568K params) | 2.2 MB | 64 | **50 ms** | 1282 |
| MLP 5 couches (8.4M params) | 32 MB | 16 | **393 ms** | 41 |
| MLP 4 couches (32M params) | 108 MB | 8 | **1049 ms** | 8 |

> **Note** : Ces mesures incluent forward + backward + mise à jour Adam complètes. Pas des benchmarks BLAS purs.

### En pratique : une époque MNIST (60K samples)

- Modèle 568K params, batch 64 → **~47 secondes** par époque
- 10 époques → **~8 minutes**

C'est lent comparé à un GPU (qui fait ça en 15-30 secondes), mais ça **marche sur n'importe quel PC**.

---

## Installation

```bash
git clone https://github.com/Doom-pixel-alt/2C2T.DRT.git
cd 2C2T.DRT/2C2T.DRT
python main.py
```

Dépendance unique : **NumPy** (qui utilise OpenBLAS optimisé en assembleur pour les calculs matriciels).

---

## Utilisation

### En Python

```python
import c2t as nn
from c2t.data import DataLoader, TensorDataset

# 1. Construire le modèle
model = nn.Sequential(
    nn.Flatten(),
    nn.DenseReLU(784, 512),    # Dense + ReLU fusionnés (1 noeud autograd)
    nn.DenseReLU(512, 256),
    nn.Dense(256, 10),
)

# 2. Optimiseur + fonction de perte
optimizer = nn.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 3. Trainer
trainer = nn.Trainer(model, loss_fn, optimizer)
trainer.fit(train_loader, val_loader, epochs=10)

# 4. Inférence
predictions = trainer.predict(x_test)
```

### En ligne de commande

```bash
# MNIST (ou données synthétiques si pas de réseau)
python main.py

# Grand modèle avec sharding mémoire
python main.py --model large --shard --shard-size 200

# Entraînement longue durée
python main.py --model deep --epochs 100 --lr 0.0005 --grad-accum 4

# Évaluation seule
python main.py --model cnn --eval-only --load model.npz

# Benchmark
python benchmark.py
```

---

## Architectures disponibles

| Flag | Modèle | Poids | Paramètres | RAM requise |
|------|--------|-------|------------|-------------|
| `mlp` | 4 couches fully connected | 2.2 MB | 567 K | Très faible |
| `large` | 5 couches fully connected | 32 MB | 8.4 M | Faible |
| `deep` | 9 couches fully connected | 26 MB | 6.7 M | Faible |
| `huge` | 12 couches fully connected | 860 MB | 226 M | Élevée |
| `cnn` | Conv2D + fully connected | 137 MB | 36 M | Élevée |

---

## Fonctionnalités

### Core

| Fonctionnalité | Statut |
|----------------|--------|
| Tenseur avec autograd (20+ opérations différentiables) | ✅ Stable |
| Couches : Dense, Conv2D, BatchNorm, Dropout | ✅ Stable |
| Activations : ReLU, LeakyReLU, Sigmoid, Tanh, Softmax | ✅ Stable |
| DenseReLU fusionné (Dense + ReLU en 1 noeud) | ✅ Stable |
| Optimiseurs : SGD, Adam, AdamW, RMSprop | ✅ Stable |
| Loss : MSE, MAE, CrossEntropy, BinaryCE, Huber, NLL | ✅ Stable |

### Optimisation mémoire

| Technique | Description | Bénéfice |
|-----------|-------------|----------|
| **Gradient accumulation** | Accumule les gradients sur N micro-batches | Simule des batches plus grands sans RAM supplémentaire |
| **Gradient checkpointing** | Recalcule les activations au backward | Échange temps → mémoire (50% de RAM en moins) |
| **Sharding mémoire** | Découpe le modèle, charge/décharge depuis le disque | Modèles plus grands que la RAM |
| **Quantization 8/16-bit** | Compression des poids | Jusqu'à 4× moins de RAM/stockage |
| **Auto-batch** | Calcule la batch size optimale | Évite les MemoryError |

### Entraînement

| Fonctionnalité | Statut |
|----------------|--------|
| Trainer complet avec metrics | ✅ Stable |
| LR Scheduler (ReduceLROnPlateau) | ✅ Stable |
| Early Stopping | ✅ Stable |
| Save / Load des poids | ✅ Stable |
| DataLoader avec shuffle + batching | ✅ Stable |
| MemoryMapDataset (fichiers > RAM) | ✅ Stable |

### Infrastructure

| Fonctionnalité | Statut |
|----------------|--------|
| Parallélisme OpenBLAS multi-cœur | ✅ Stable |
| Stockage mmap pour inférence streaming | ✅ Stable |
| Multiplateforme (Windows, Linux, macOS) | ✅ Testé (Windows 11) |
| Zéro dépendance GPU | ✅ Garanti |

---

## Limites connues (honnêtement)

- **Pas de miracle** : comptez 15-30× plus lent qu'un GPU d'entrée de gamme
- **Mémoire vive** : la taille du modèle est limitée par la RAM disponible (pas de VRAM)
- **Pas d'entraînement 100B+** : les activations du backward doivent tenir en RAM. Le streaming ne marche que pour l'inférence.
- **Pas de kernels "maison"** : numpy/OpenBLAS sont déjà optimisés au maximum en assembleur. Nos matmuls ne seront jamais plus rapides que les leurs.

### Quand l'utiliser

Cas d'usage | Recommandation
--- | ---
Vous avez un GPU | **Utilisez PyTorch**. 2C2T.DRT n'est pas fait pour vous.
Vous n'avez PAS de GPU | **2C2T.DRT est parfait**. Apprenez, prototyp ez, entraînez.
Vous voulez apprendre le deep learning | **Commencez ici**. Pas de cloud, pas d'installation complexe.
Vous devez déployer sur CPU | **Idéal**. Pas de dépendance CUDA, un seul fichier.
Votre modèle est trop gros pour la VRAM | **Sharding** : chargez/déchargez les poids depuis le disque.

---

## Comparaison GPU vs CPU (prix réels 2024-2025)

| Solution | Prix (€) | Perf brute | Disponibilité |
|----------|----------|------------|---------------|
| **2C2T.DRT (CPU)** | **0 €** (vous avez déjà un PC) | 1× | ✅ Immédiate |
| RTX 3060 12 Go | 450-550 € (pénurie) | 30× | ⚠️ Rupture fréquente |
| RTX 4060 Ti 16 Go | 600-700 € | 40× | ⚠️ Stock limité |
| RTX 4090 24 Go | 2500-3500 € (scalping) | 100× | ❌ Rupture quasi-permanente |
| A100 80 Go (pro) | 25 000-35 000 € | 300× | ❌ Réservé aux entreprises |
| Cloud GPU (location) | 1-5 €/h | Variable | ✅ Mais coûteux à long terme |

**Conclusion** : Si vous avez un PC, vous avez déjà de quoi faire du deep learning avec 2C2T.DRT. C'est lent, mais c'est gratuit et ça marche.

---

## Structure du projet

```
2C2T.DRT/
├── c2t/                        # Package Python
│   ├── __init__.py             # API publique
│   ├── tensor.py               # Tenseur avec autograd
│   ├── layers/__init__.py      # Toutes les couches
│   ├── optimizers.py           # Optimiseurs
│   ├── losses.py               # Fonctions de perte
│   ├── trainer.py              # Moteur d'entraînement
│   ├── data.py                 # Chargement de données
│   ├── parallel.py             # Parallélisme CPU
│   ├── sharding.py             # Sharding mémoire
│   ├── storage.py              # Stockage mmap
│   └── memory.py               # Optimisation mémoire
├── main.py                     # Ligne de commande
├── benchmark.py                # Benchmarks
└── README.md
```

---

## Licence

MIT — faites ce que vous voulez. Améliorez, fork ez, distribuez.

Dream Reality Technologies — *parce que tout le monde mérite d'entraîner l'IA.*
