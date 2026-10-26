#!/usr/bin/env python3
"""Interface graphique Tkinter pour le classifieur MNIST entraîné.

Lancement :
    python3 mnist_gui.py                      # interface interactive
    python3 mnist_gui.py --self-test          # vérification automatisée (sans interaction)

L'application dessine dans un canvas 560x560 px, maintient un raster
numérique 280x280 (source de vérité pour l'inférence), le réduit en 28x28
par moyennes de blocs 10x10, applique la normalisation MNIST et interroge
le CNN entraîné par train.py.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tkinter as tk
from PIL import Image, ImageTk

from train import MNISTNet

# --- Constantes du modèle / du prétraitement -------------------------------
GRID = 28                  # entrée du modèle
RASTER = 280               # résolution du raster de dessin (280 = 28 * 10)
BLOCK = RASTER // GRID     # 10 px de dessin par pixel de modèle
SCALE = 2                  # affichage : 280 px de raster -> 560 px de canvas
MEAN, STD = 0.1307, 0.3081
CHECKPOINT = Path("checkpoints/mnist_cnn.pt")
BRUSHES = {"Fine": 8, "Moyenne": 12, "Épaisse": 18}  # rayon en px raster


# --- Modèle ----------------------------------------------------------------
def load_model(path: Path = CHECKPOINT) -> tuple[MNISTNet, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {path}\nLancez d'abord : python3 train.py"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MNISTNet()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# --- Rasterisation / prétraitement ----------------------------------------
def new_grid() -> np.ndarray:
    return np.zeros((RASTER, RASTER), dtype=np.uint8)


def make_disc(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def stamp(grid: np.ndarray, cx: int, cy: int, disc: np.ndarray) -> bool:
    """Peint un disque blanc de rayon len(disc)//2 centré en (cx, cy)."""
    r = disc.shape[0] // 2
    y0, y1 = max(0, cy - r), min(RASTER, cy + r + 1)
    x0, x1 = max(0, cx - r), min(RASTER, cx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return False
    dy, dx = y0 - (cy - r), x0 - (cx - r)
    patch = disc[dy : dy + (y1 - y0), dx : dx + (x1 - x0)]
    region = grid[y0:y1, x0:x1]
    region[patch] = 255
    return True


def stamp_segment(
    grid: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    disc: np.ndarray,
) -> bool:
    """Trace un segment dans le raster en tamponnant le long du trait."""
    (x0, y0), (x1, y1) = p0, p1
    dist = float(np.hypot(x1 - x0, y1 - y0))
    steps = max(1, int(dist))
    touched = False
    for i in range(steps + 1):
        t = i / steps
        cx = int(round(x0 + (x1 - x0) * t))
        cy = int(round(y0 + (y1 - y0) * t))
        touched = stamp(grid, cx, cy, disc) or touched
    return touched


def grid_to_tensor(grid: np.ndarray) -> torch.Tensor:
    """280x280 uint8 -> tenseur [1,1,28,28] normalisé (stats MNIST)."""
    if grid.shape != (RASTER, RASTER):
        raise ValueError(f"raster attendu {(RASTER, RASTER)}, obtenu {grid.shape}")
    small = grid.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3)) / 255.0
    t = torch.from_numpy(small.astype(np.float32)).view(1, 1, GRID, GRID)
    return (t - MEAN) / STD


@torch.no_grad()
def predict(model: MNISTNet, grid: np.ndarray) -> tuple[int, np.ndarray]:
    probs = torch.softmax(model(grid_to_tensor(grid)), dim=1).squeeze(0).numpy()
    return int(probs.argmax()), probs


# --- Interface -------------------------------------------------------------
class App:
    def __init__(self, root: tk.Tk, model: MNISTNet, ckpt: dict) -> None:
        self.root = root
        self.model = model
        self.test_acc = float(ckpt.get("test_acc", 0.0))
        self.grid = new_grid()
        self.disc = make_disc(BRUSHES["Moyenne"])
        self.brush_radius = BRUSHES["Moyenne"]
        self.last: tuple[int, int] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._prev_photo: ImageTk.PhotoImage | None = None
        self._test_ds = None
        self._inference_ms = 0.0

        root.title("MNIST — reconnaissance de chiffres manuscrits")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        self._build_layout()

    # --- Construction -----------------------------------------------------
    def _build_layout(self) -> None:
        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack()

        # Colonne dessin
        left = tk.Frame(main)
        left.pack(side="left", padx=(0, 12))

        self.canvas = tk.Canvas(
            left,
            width=RASTER * SCALE,
            height=RASTER * SCALE,
            bg="#101010",
            highlightthickness=1,
            highlightbackground="#444",
            cursor="pencil",
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        brush_bar = tk.Frame(left, pady=6)
        brush_bar.pack()
        tk.Label(brush_bar, text="Pinceau :").pack(side="left", padx=(0, 6))
        self.brush_var = tk.StringVar(value="Moyenne")
        for name in BRUSHES:
            tk.Radiobutton(
                brush_bar,
                text=name,
                variable=self.brush_var,
                value=name,
                command=self._set_brush,
            ).pack(side="left", padx=2)

        # Colonne résultats
        right = tk.Frame(main)
        right.pack(side="left", fill="y")

        tk.Label(right, text="Ce que voit le modèle", font=("Helvetica", 11, "bold")).pack(
            anchor="w"
        )
        self.preview = tk.Label(right, bg="#101010", bd=1, relief="solid")
        self.preview.pack(pady=(2, 8))

        pred = tk.Frame(right)
        pred.pack(anchor="w", pady=(0, 4))
        self.pred_label = tk.Label(
            pred, text="—", font=("Helvetica", 64, "bold"), width=2, fg="#1f77d0"
        )
        self.pred_label.pack(side="left")
        conf = tk.Frame(pred)
        conf.pack(side="left", padx=10, anchor="s", pady=8)
        tk.Label(conf, text="Classe prédite", font=("Helvetica", 10)).pack(anchor="w")
        self.conf_label = tk.Label(conf, text="—", font=("Helvetica", 16, "bold"))
        self.conf_label.pack(anchor="w")

        tk.Label(right, text="Probabilités par classe", font=("Helvetica", 11, "bold")).pack(
            anchor="w", pady=(8, 2)
        )
        self.bars = tk.Canvas(right, width=320, height=10 * 22, bg="#f4f4f4", highlightthickness=0)
        self.bars.pack()
        self._draw_bars(np.full(10, 0.1))

        btns = tk.Frame(right, pady=8)
        btns.pack()
        tk.Button(btns, text="Effacer", width=10, command=self.clear).grid(row=0, column=0, padx=4, pady=3)
        tk.Button(btns, text="Prédire", width=10, command=self.predict_now).grid(row=0, column=1, padx=4, pady=3)
        tk.Button(btns, text="Exemple au hasard", width=14, command=self.random_example).grid(
            row=1, column=0, columnspan=2, padx=4, pady=3
        )

        self.status = tk.Label(
            right,
            text=f"Modèle : CNN · précision test {100 * self.test_acc:.2f} %",
            font=("Helvetica", 9),
            fg="#666",
            anchor="w",
        )
        self.status.pack(side="bottom", fill="x")

    # --- Dessin -----------------------------------------------------------
    def _to_raster(self, x: int, y: int) -> tuple[int, int]:
        return (max(0, min(RASTER - 1, x // SCALE)), max(0, min(RASTER - 1, y // SCALE)))

    def _set_brush(self) -> None:
        self.brush_radius = BRUSHES[self.brush_var.get()]
        self.disc = make_disc(self.brush_radius)

    def _on_press(self, event: tk.Event) -> None:
        self.last = self._to_raster(event.x, event.y)
        stamp(self.grid, *self.last, self.disc)
        r = self.brush_radius * SCALE
        x, y = event.x, event.y
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r, fill="white", outline="white", tags="stroke"
        )
        self._hide_example()

    def _on_drag(self, event: tk.Event) -> None:
        cur = self._to_raster(event.x, event.y)
        if self.last is None:
            self.last = cur
            return
        stamp_segment(self.grid, self.last, cur, self.disc)
        self.canvas.create_line(
            self.last[0] * SCALE,
            self.last[1] * SCALE,
            cur[0] * SCALE,
            cur[1] * SCALE,
            fill="white",
            width=self.brush_radius * 2 * SCALE,
            capstyle="round",
            smooth=True,
            tags="stroke",
        )
        self.last = cur

    def _on_release(self, _event: tk.Event) -> None:
        self.last = None
        self.predict_now()

    # --- Actions ----------------------------------------------------------
    def clear(self) -> None:
        self.grid.fill(0)
        self.canvas.delete("stroke")
        self.canvas.delete("example")
        self._photo = None
        self.pred_label.config(text="—")
        self.conf_label.config(text="—")
        self._draw_bars(np.full(10, 0.1))
        self.preview.config(image="")
        self.status.config(text="Canvas effacé.")

    def _hide_example(self) -> None:
        if self._photo is not None:
            self.canvas.delete("example")
            self._photo = None

    def load_grid(self, grid: np.ndarray, show: bool = True) -> None:
        """Charge un raster 280x280 (ex. chiffre MNIST agrandi) et l'affiche."""
        if grid.shape != (RASTER, RASTER):
            raise ValueError(f"raster attendu {(RASTER, RASTER)}, obtenu {grid.shape}")
        self.grid = grid.astype(np.uint8, copy=True)
        self.canvas.delete("stroke")
        self.canvas.delete("example")
        self._photo = None
        if show:
            img = Image.fromarray(self.grid)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="example")

    def random_example(self) -> None:
        if self._test_ds is None:
            from torchvision import datasets

            self._test_ds = datasets.MNIST("data", train=False, download=True)
        i = random.randrange(len(self._test_ds))
        img = self._test_ds.data[i].numpy()  # 28x28 uint8
        grid = np.kron(img, np.ones((BLOCK, BLOCK), dtype=np.uint8))
        self.load_grid(grid)
        self.status.config(text=f"Exemple de test MNIST n°{i} chargé — étiquette réelle : {int(self._test_ds.targets[i])}")
        self.predict_now()

    def predict_now(self) -> tuple[int, np.ndarray]:
        if not self.grid.any():
            self.pred_label.config(text="—")
            self.conf_label.config(text="—")
            self._draw_bars(np.full(10, 0.1))
            self.status.config(text="Dessinez un chiffre pour lancer la prédiction.")
            return 0, np.zeros(10)

        t0 = time.perf_counter()
        label, probs = predict(self.model, self.grid)
        self._inference_ms = (time.perf_counter() - t0) * 1000

        self.pred_label.config(text=str(label))
        self.conf_label.config(text=f"{100 * float(probs[label]):.1f} %")
        self._draw_bars(probs)
        self._update_preview()
        self.status.config(
            text=f"Prédiction en {self._inference_ms:.1f} ms · "
            f"précision test {100 * self.test_acc:.2f} %"
        )
        self.root.update_idletasks()
        return label, probs

    def _draw_bars(self, probs: np.ndarray) -> None:
        self.bars.delete("all")
        w, h, gap = 230, 14, 22
        for d in range(10):
            y = d * gap
            p = float(probs[d])
            self.bars.create_text(10, y + h / 2, text=str(d), anchor="w", font=("Helvetica", 10, "bold"))
            self.bars.create_rectangle(28, y, 28 + w, y + h, fill="#e2e2e2", outline="")
            color = "#1f77d0" if d == int(np.argmax(probs)) else "#8fb8e0"
            self.bars.create_rectangle(28, y, 28 + max(1, int(w * p)), y + h, fill=color, outline="")
            self.bars.create_text(
                28 + w + 8, y + h / 2, text=f"{100 * p:.1f}%", anchor="w", font=("Helvetica", 9)
            )

    def _update_preview(self) -> None:
        small = self.grid.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3)).astype(np.uint8)
        img = Image.fromarray(small).resize((224, 224), Image.NEAREST)
        self._prev_photo = ImageTk.PhotoImage(img)
        self.preview.config(image=self._prev_photo)


# --- Test automatisé (s'exécute sous Xvfb : pas d'affichage réel) ----------
def _ev(x: int, y: int) -> tk.Event:
    e = tk.Event()
    e.x, e.y = x, y
    return e


def self_test(args: argparse.Namespace) -> int:
    model, ckpt = load_model(args.checkpoint)
    root = tk.Tk()
    app = App(root, model, ckpt)
    root.update()  # construit réellement les widgets

    # 1. Dessin simulé (même chemin d'événements que la souris)
    app._on_press(_ev(20, 140))
    for x in range(40, 260, 20):
        app._on_drag(_ev(x, 140))
    app._on_release(_ev(260, 140))
    assert app.grid.any(), "le raster resté vide après dessin simulé"
    label, probs = app.predict_now()
    assert probs.shape == (10,) and np.isclose(probs.sum(), 1.0, atol=1e-5)
    print(f"[1] dessin simulé -> prédiction {label}, somme(probas)={probs.sum():.4f}")

    # 2. Chiffres MNIST réels passant par le pipeline complet de l'interface
    from torchvision import datasets

    ds = datasets.MNIST("data", train=False, download=True)
    idxs = random.sample(range(len(ds)), 8)
    hits = 0
    for i in idxs:
        grid = np.kron(ds.data[i].numpy(), np.ones((BLOCK, BLOCK), dtype=np.uint8))
        app.load_grid(grid)
        label, _ = app.predict_now()
        truth = int(ds.targets[i])
        hits += label == truth
        print(f"[2] exemple {i}: prédit={label} attendu={truth} {'OK' if label == truth else 'KO'}")
    root.update()
    root.destroy()

    print(f"[2] précision sur échantillon : {hits}/{len(idxs)}")
    assert hits >= len(idxs) - 1, f"trop d'erreurs sur l'échantillon ({hits}/{len(idxs)})"
    print("SELF-TEST PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="GUI Tkinter pour le classifieur MNIST")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--self-test", action="store_true", help="vérification automatisée puis exit")
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint)
    if args.self_test:
        return self_test(args)

    root = tk.Tk()
    App(root, model, ckpt)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
