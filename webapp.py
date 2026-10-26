#!/usr/bin/env python3
"""Serveur web local pour le classifieur MNIST entraîné par train.py.

Lancement :
    python3 webapp.py [--port 7860] [--checkpoint checkpoints/mnist_cnn.pt]

Expose :
    GET  /                -> interface (static/index.html)
    GET  /api/health      -> état + précision du modèle
    POST /api/predict     -> prédiction à partir de traits {radius, strokes, base28?}
    POST /api/example     -> chiffre MNIST réel + prédiction

Le prétraitement est exactement celui de l'interface Tkinter : les traits
sont rasterisés en 280x280 par mnist_gui.stamp_segment, réduits en 28x28
par moyennes de blocs 10x10 puis normalisés (stats MNIST) avant d'être
envoyés au CNN. Une seule source de vérité pour les deux interfaces.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from mnist_gui import (
    BLOCK,
    BRUSHES,
    GRID,
    RASTER,
    load_model,
    make_disc,
    new_grid,
    predict,
    stamp,
    stamp_segment,
)

INDEX = Path(__file__).parent / "static" / "index.html"
MAX_BODY = 4 * 1024 * 1024  # 4 Mo
MAX_POINTS = 20_000

_infer_lock = threading.Lock()
_state: dict = {}


def build_grid(strokes: list, radius: int, base28: list | None) -> np.ndarray:
    """Construit le raster 280x280 : chiffre de base (optionnel) + traits."""
    if base28 is not None:
        arr = np.asarray(base28, dtype=np.uint8)
        if arr.size != GRID * GRID:
            raise ValueError(f"base28 doit contenir {GRID * GRID} valeurs, obtenu {arr.size}")
        grid = np.kron(arr.reshape(GRID, GRID), np.ones((BLOCK, BLOCK), dtype=np.uint8))
    else:
        grid = new_grid()

    disc = make_disc(radius)
    total = 0
    for stroke in strokes:
        pts = [(int(p[0]), int(p[1])) for p in stroke]
        total += len(pts)
        if total > MAX_POINTS:
            raise ValueError("trop de points dans la requête")
        if not pts:
            continue
        if len(pts) == 1:
            stamp(grid, *pts[0], disc)
        for a, b in zip(pts, pts[1:]):
            stamp_segment(grid, a, b, disc)
    return grid


def run_predict(strokes: list, radius: int, base28: list | None) -> dict:
    grid = build_grid(strokes, radius, base28)
    if not grid.any():
        return {"empty": True, "label": None, "probs": [0.1] * 10}
    t0 = time.perf_counter()
    with _infer_lock:
        label, probs = predict(_state["model"], grid)
    ms = (time.perf_counter() - t0) * 1000
    small = grid.reshape(GRID, BLOCK, GRID, BLOCK).mean(axis=(1, 3)).astype(np.uint8)
    return {
        "empty": False,
        "label": label,
        "confidence": float(probs[label]),
        "probs": [float(p) for p in probs],
        "ms": round(ms, 2),
        "preview28": small.flatten().tolist(),
        "test_acc": _state["test_acc"],
    }


def run_example() -> dict:
    if _state["test_ds"] is None:
        from torchvision import datasets

        _state["test_ds"] = datasets.MNIST("data", train=False, download=True)
    ds = _state["test_ds"]
    i = random.randrange(len(ds))
    image28 = ds.data[i].numpy().flatten().tolist()
    truth = int(ds.targets[i])
    out = run_predict([], 0, image28)  # base seule, aucun trait
    out.update({"index": i, "true_label": truth, "image28": image28})
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "MNISTWeb/1.0"

    def log_message(self, fmt: str, *args) -> None:  # log concise
        sys.stderr.write("[web] %s - %s\n" % (self.address_string(), fmt % args))

    # --- réponses ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"error": message})

    # --- GET --------------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            try:
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._error(404, f"interface introuvable : {INDEX}")
        elif path == "/api/health":
            self._json(
                200,
                {
                    "status": "ok",
                    "test_acc": _state["test_acc"],
                    "checkpoint": str(_state["ckpt_path"]),
                    "brushes": BRUSHES,
                },
            )
        else:
            self._error(404, "route inconnue")

    # --- POST -------------------------------------------------------------
    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self._error(413, "corps de requête trop volumineux")
            return
        raw = self.rfile.read(length) if length else b"{}"

        if path not in ("/api/predict", "/api/example"):
            self._error(404, "route inconnue")
            return
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._error(400, "JSON invalide")
            return

        try:
            if path == "/api/predict":
                strokes = body.get("strokes") or []
                if not isinstance(strokes, list):
                    raise ValueError("strokes doit être une liste")
                radius = int(body.get("radius") or BRUSHES["Moyenne"])
                radius = max(1, min(RASTER // 4, radius))
                base28 = body.get("base28")
                self._json(200, run_predict(strokes, radius, base28))
            else:
                self._json(200, run_example())
        except (ValueError, TypeError, KeyError) as exc:
            self._error(400, f"requête invalide : {exc}")
        except Exception as exc:  # noqa: BLE001 - on répond au client plutôt que de couvrir
            self._error(500, f"erreur serveur : {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interface web pour le classifieur MNIST")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mnist_cnn.pt"))
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint)
    _state.update(
        model=model,
        test_acc=float(ckpt.get("test_acc", 0.0)),
        ckpt_path=args.checkpoint,
        test_ds=None,
    )
    if not INDEX.exists():
        print(f"ERREUR : interface introuvable : {INDEX}", file=sys.stderr)
        return 1

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"MNIST web UI sur http://localhost:{args.port}  "
        f"(modèle {args.checkpoint}, précision test {100 * _state['test_acc']:.2f} %)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
