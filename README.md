# MNIST

Entraînement d'un CNN sur le dataset MNIST avec PyTorch (CPU).

## Setup

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

Le dataset MNIST est téléchargé automatiquement dans `data/` au premier lancement.

## Entraînement

```bash
python3 train.py --epochs 5
```

Options : `--epochs`, `--batch-size`, `--lr`, `--seed`, `--out`.

## Résultats

- Modèle : CNN 2 couches convolutives (32 + 64 filtres) + 2 couches denses (128 -> 10), Dropout 0.25
- Augmentation : RandomAffine (rotations ±15°, translations 10 %)
- Optimiseur : Adam (lr 1e-3), batch 128, 5 époques (~4 min sur CPU)
- **Accuracy test : 99.06 %** (meilleure époque : 4)

## Interface web (navigateur)

```bash
python3 webapp.py --port 7860   # puis ouvrir http://localhost:7860
```

Dans un Codespace, le port 7860 est détecté et transféré automatiquement
(onglet « Ports » de VS Code → *Open in Browser*).

- Canvas de dessin 560×560 px (souris/tactile), 3 tailles de pinceau
- Prédiction automatique à la fin de chaque trait (~2,5 ms), boutons
  **Effacer**, **Prédire**, **Exemple au hasard**
- Aperçu 28×28, classe prédite, confidence et barres de probabilités
- Réutilise le **même prétraitement** que l'UI Tkinter (`mnist_gui.py`) :
  une seule source de vérité pour les deux interfaces

API :

| Route | Méthode | Corps / retour |
|---|---|---|
| `/` | GET | interface HTML |
| `/api/health` | GET | état + précision du modèle |
| `/api/predict` | POST | `{radius, strokes, base28?}` → `{label, confidence, probs, preview28, ms}` |
| `/api/example` | POST | chiffre MNIST réel + `{image28, true_label, label, ...}` |

Tests de l'API : `python3 -m py_compile webapp.py` + parcours complet
(HTML, health, prédiction, erreurs 400/404, 6 chiffres réels) validé à 6/6.

## Interface graphique (Tkinter)

```bash
python3 mnist_gui.py
```

- Canvas 560×560 px : dessinez au souris (pinceau fine / moyenne / épaisse)
- Aperçu 28×28 de ce que voit le modèle (réduction par blocs 10×10)
- Classe prédite, confidence et barres de probabilités des 10 classes
- Prédiction automatique à la fin de chaque trait (~15 ms)
- Boutons **Effacer**, **Prédire**, **Exemple au hasard** (chiffre MNIST réel)

Vérification automatisée de l'interface (sans interaction, sous Xvfb) :

```bash
xvfb-run -a python3 mnist_gui.py --self-test
```

> L'interface est une fenêtre native : dans un Codespace (pas de serveur
> d'affichage), lancez-la sur votre machine locale avec les mêmes fichiers,
> ou utilisez un tunnel X11. Le `--self-test` fonctionne partout grâce à Xvfb.

Artefacts :

- `checkpoints/mnist_cnn.pt` — poids du meilleur modèle + métadonnées
- `checkpoints/mnist_cnn.history.json` — historique d'entraînement
