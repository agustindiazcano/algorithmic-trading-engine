# molecular_moons_compare.py
# ------------------------------------------------------------
# Comparación progresiva (sin rotación):
#   - Two Moons 2D intrínseco (con notch/warp por clase)
#   - Two Moons 3D intrínseco (twist por clase)
#
# Modelos:
#   🧠 MLP (PointNet-lite)
#   🟠 VNN (esferas en espacio de features rot-invariant)
#   🧬 MVNN Molecular (unión rígida de esferas, NO pose search, NO rotación)
#
# Reqs: python>=3.10, numpy, torch
#   pip install numpy torch
#
# Run:
#   python molecular_moons_compare.py --dataset both --scenario no_rot
#   python molecular_moons_compare.py --dataset two_moons_2d --scenario no_rot
# ------------------------------------------------------------

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# 🔧 Utils
# =========================
def seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_translation_np(pts: np.ndarray, t: np.ndarray) -> np.ndarray:
    return pts + t


# =========================
# 🌙 Dataset: Two Moons (intrínseco)
# =========================
def gen_two_moons_intrinsic_2d(
    class_id: int, n_points: int, rng: np.random.RandomState, noise: float = 0.02
) -> np.ndarray:
    """
    Two Moons 2D pero con diferencias INTRÍNSECAS por clase:
      - warp en y depende de la clase
      - "notch" (muesca): evitamos un intervalo de theta distinto por clase

    Devuelve (N,3) con z=0.
    """
    base_r = 1.0
    thickness = 0.20 if class_id == 0 else 0.14  # grosor distinto
    warp_amp = 0.30 if class_id == 0 else 0.18   # warp distinto

    # notch: intervalo de theta "prohibido" por clase (cambia la forma)
    # clase 0: notch cerca de theta ~ 0.35π
    # clase 1: notch cerca de theta ~ 0.70π
    if class_id == 0:
        notch_center = 0.35 * math.pi
    else:
        notch_center = 0.70 * math.pi
    notch_halfwidth = 0.10 * math.pi

    # muestreamos theta evitando notch
    thetas = []
    while len(thetas) < n_points:
        th = float(rng.rand() * math.pi)  # [0,pi]
        if abs(th - notch_center) > notch_halfwidth:
            thetas.append(th)
    theta = np.array(thetas, dtype=np.float32)

    r = base_r + thickness * (rng.rand(n_points).astype(np.float32) - 0.5)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # warp intrínseco (no depende de traslación)
    y = y + warp_amp * np.sin(2.0 * theta + (0.6 if class_id == 0 else 1.3))

    # segunda luna: mirror+shift leve (pero si centrás no te quedás sin señal, porque hay notch/warp)
    if class_id == 1:
        x = x + 0.25
        y = -y - 0.10

    pts2 = np.stack([x, y], axis=1).astype(np.float32)
    pts2 += noise * rng.randn(n_points, 2).astype(np.float32)

    pts3 = np.concatenate([pts2, np.zeros((n_points, 1), dtype=np.float32)], axis=1)
    return pts3


def add_intrinsic_twist_3d(pts3: np.ndarray, class_id: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Convierte nube 2D (z=0) en 3D con "twist" INTRÍNSECO por clase.
    """
    p = pts3.copy().astype(np.float32)
    xy = p[:, :2]
    ang = np.arctan2(xy[:, 1], xy[:, 0]).astype(np.float32)

    # twist depende de clase (señal intrínseca)
    strength = 0.40 if class_id == 0 else 0.22
    phase = 0.3 if class_id == 0 else 1.1

    z = strength * np.sin(2.0 * ang + phase) + 0.05 * (rng.rand(len(p)).astype(np.float32) - 0.5)
    p[:, 2] = z
    return p


def make_two_moons_dataset(
    n_samples: int,
    n_points: int,
    rng: np.random.RandomState,
    variant: str,              # "2d" or "3d"
    scenario: str,             # "no_rot" (sin rotación)
    translation_scale: float = 0.15,
    outlier_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset SIN rotación.
    Permitimos una traslación suave (opcional) para realismo; si center=True se elimina.
    """
    X, y = [], []
    for _ in range(n_samples):
        cls = int(rng.randint(0, 2))
        pts = gen_two_moons_intrinsic_2d(cls, n_points=n_points, rng=rng)

        if variant == "3d":
            pts = add_intrinsic_twist_3d(pts, class_id=cls, rng=rng)

        # scenario no_rot: NO aplicamos R
        if scenario == "no_rot":
            t = rng.randn(3).astype(np.float32) * translation_scale
            if variant == "2d":
                t[2] = 0.0
            pts = apply_translation_np(pts, t)

        # outliers opcionales
        if outlier_ratio > 0:
            k = int(n_points * outlier_ratio)
            if k > 0:
                pts[:k] = rng.uniform(-2.5, 2.5, size=(k, 3)).astype(np.float32)

        X.append(pts)
        y.append(cls)

    return np.stack(X, axis=0), np.array(y, dtype=np.int64)


class NumpyPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, center: bool = True):
        self.X = X
        self.y = y
        self.center = center

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        pts = self.X[idx].astype(np.float32)
        if self.center:
            pts = pts - pts.mean(axis=0, keepdims=True)  # invariancia a traslación
        return torch.from_numpy(pts), torch.tensor(self.y[idx], dtype=torch.long)


# =========================
# 🧠 Baseline: PointNet-lite
# =========================
class PointNetLite(nn.Module):
    def __init__(self, n_classes: int = 2, d_in: int = 3):
        super().__init__()
        self.per_point = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, pts):  # (B,N,3)
        h = self.per_point(pts)          # (B,N,128)
        g = torch.max(h, dim=1).values   # (B,128)
        return self.head(g)


def train_pointnet(model, train_loader, test_loader, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for pts, y in train_loader:
            pts, y = pts.to(device), y.to(device)
            opt.zero_grad()
            logits = model(pts)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            n += len(y)

        if ep == 1 or ep % 5 == 0:
            acc = eval_torch_classifier(model, test_loader, device=device)["acc"]
            print(f"[MLP] ep={ep:02d} loss={total/n:.4f} acc={acc:.3f}")

    return model


def eval_torch_classifier(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for pts, y in loader:
            pts, y = pts.to(device), y.to(device)
            logits = model(pts)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += len(y)
    t1 = time.perf_counter()
    return {"acc": correct / total, "ms_per_sample": (t1 - t0) * 1000 / total}


# =========================
# 🟠 VNN: esferas en features rot-invariant
# =========================
def cloud_features(pts: np.ndarray, rng: np.random.RandomState, n_hist=8, pair_samples=256) -> np.ndarray:
    pts = pts.astype(np.float32)
    c = pts.mean(axis=0, keepdims=True)
    p = pts - c

    cov = (p.T @ p) / (len(p) + 1e-9)
    evals = np.linalg.eigvalsh(cov).astype(np.float32)
    evals = np.sort(evals)[::-1]

    r = np.linalg.norm(p, axis=1).astype(np.float32)
    r_mean = float(r.mean())
    r_std = float(r.std() + 1e-9)
    r_skew = float(np.mean(((r - r_mean) / r_std) ** 3))

    rmax = float(np.percentile(r, 95))
    bins = np.linspace(0, rmax + 1e-6, n_hist + 1, dtype=np.float32)
    hist, _ = np.histogram(r, bins=bins)
    hist = hist.astype(np.float32) / len(r)

    idx1 = rng.randint(0, len(p), size=pair_samples)
    idx2 = rng.randint(0, len(p), size=pair_samples)
    d = np.linalg.norm(p[idx1] - p[idx2], axis=1).astype(np.float32)
    q = np.percentile(d, [10, 25, 50, 75, 90]).astype(np.float32)

    feat = np.concatenate([evals, np.array([r_mean, r_std, r_skew], dtype=np.float32), hist, q], axis=0)
    return feat


@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    label: int


class VNNFeatureSphereClassifier:
    def __init__(self, base_radius: float = 0.25, lr: float = 0.2, max_spheres_per_class: int = 128):
        self.base_radius = base_radius
        self.lr = lr
        self.max_spheres_per_class = max_spheres_per_class
        self.spheres: List[Sphere] = []

    def fit(self, feats: np.ndarray, labels: np.ndarray):
        for f, y in zip(feats, labels):
            y = int(y)
            idxs = [i for i, s in enumerate(self.spheres) if s.label == y]
            if not idxs:
                self.spheres.append(Sphere(center=f.copy(), radius=float(self.base_radius), label=y))
                continue

            ds = [float(np.linalg.norm(f - self.spheres[i].center)) for i in idxs]
            j = idxs[int(np.argmin(ds))]
            d = float(np.min(ds))

            if d <= self.spheres[j].radius or len(idxs) >= self.max_spheres_per_class:
                self.spheres[j].center = (1 - self.lr) * self.spheres[j].center + self.lr * f
                self.spheres[j].radius = max(self.spheres[j].radius, d * 1.05)
            else:
                self.spheres.append(Sphere(center=f.copy(), radius=max(self.base_radius, d * 0.8), label=y))

    def predict(self, feats: np.ndarray, n_classes: int = 2) -> np.ndarray:
        preds = []
        for f in feats:
            scores = np.zeros(n_classes, dtype=np.float32)
            for s in self.spheres:
                d = float(np.linalg.norm(f - s.center))
                m = max(0.0, 1.0 - d / (s.radius + 1e-9))
                if m > scores[s.label]:
                    scores[s.label] = m
            preds.append(int(np.argmax(scores)))
        return np.array(preds, dtype=np.int64)


def eval_vnn(vnn: VNNFeatureSphereClassifier, feats: np.ndarray, labels: np.ndarray, n_classes: int = 2):
    t0 = time.perf_counter()
    pred = vnn.predict(feats, n_classes=n_classes)
    t1 = time.perf_counter()
    acc = float((pred == labels).mean())
    return {"acc": acc, "ms_per_sample": (t1 - t0) * 1000 / len(labels), "n_spheres": len(vnn.spheres)}


# =========================
# 🧬 MVNN Molecular (rígida, sin rotación)
# =========================
def farthest_point_sampling(pts: np.ndarray, m: int, rng: np.random.RandomState) -> np.ndarray:
    N = len(pts)
    idx = int(rng.randint(0, N))
    selected = [idx]
    dist = np.linalg.norm(pts - pts[idx], axis=1)
    for _ in range(1, m):
        idx = int(np.argmax(dist))
        selected.append(idx)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[idx], axis=1))
    return np.array(selected, dtype=np.int64)


def build_template_spheres(
    pts_list: List[np.ndarray],
    m: int,
    rng: np.random.RandomState,
    radius_scale: float = 1.25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plantilla molecular rígida:
      - concatena varios ejemplos canónicos de la clase (ya centrados)
      - elige M centros por FPS
      - radio por NN entre centros
    """
    pts = np.concatenate(pts_list, axis=0).astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)

    sel = farthest_point_sampling(pts, m=m, rng=rng)
    centers = pts[sel].astype(np.float32)

    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2).astype(np.float32)
    D = D + np.eye(m, dtype=np.float32) * 1e9
    nn = D.min(axis=1)
    radii = (nn * radius_scale + 1e-6).astype(np.float32)
    return centers, radii


def molecular_score_np(
    tc: np.ndarray,         # (M,3) template centers
    tr: np.ndarray,         # (M,)  template radii
    X: np.ndarray,          # (N,3) sample points
    point_radius: float = 0.18,
    bidirectional: bool = True,
    trim_q: float = 0.0,
) -> float:
    """
    Score rigid (sin rotación):
      - template->points: centros cerca de puntos
      - points->template (opcional): puntos cubiertos por plantilla
    """
    Xc = X.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)

    # template -> points
    diff = tc[:, None, :] - Xc[None, :, :]
    dist2 = np.sum(diff * diff, axis=2).astype(np.float32)
    dmin = np.sqrt(np.min(dist2, axis=1) + 1e-9).astype(np.float32)
    m1 = np.maximum(0.0, 1.0 - dmin / (tr + 1e-6)).astype(np.float32)

    if not bidirectional:
        return float(m1.mean())

    # points -> template
    diff2 = Xc[:, None, :] - tc[None, :, :]
    dist2b = np.sum(diff2 * diff2, axis=2).astype(np.float32)
    dmin2 = np.sqrt(np.min(dist2b, axis=1) + 1e-9).astype(np.float32)
    m2 = np.maximum(0.0, 1.0 - dmin2 / (point_radius + 1e-6)).astype(np.float32)

    def agg(v: np.ndarray) -> float:
        if trim_q is None or trim_q <= 0.0:
            return float(v.mean())
        lo = np.quantile(v, trim_q)
        hi = np.quantile(v, 1.0 - trim_q)
        vv = v[(v >= lo) & (v <= hi)]
        return float(v.mean() if len(vv) == 0 else vv.mean())

    return 0.5 * agg(m1) + 0.5 * agg(m2)


def molecular_predict_np(
    templates: List[Tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    point_radius: float = 0.18,
    bidirectional: bool = True,
    trim_q: float = 0.0,
) -> Tuple[np.ndarray, float]:
    preds = []
    t0 = time.perf_counter()
    for pts in X:
        best_c, best_s = None, -1.0
        for c, (tc, tr) in enumerate(templates):
            s = molecular_score_np(tc, tr, pts, point_radius=point_radius, bidirectional=bidirectional, trim_q=trim_q)
            if s > best_s:
                best_s = s
                best_c = c
        preds.append(int(best_c))
    t1 = time.perf_counter()
    return np.array(preds, dtype=np.int64), float((t1 - t0) * 1000 / len(X))


# =========================
# 🧪 Runner
# =========================
def run_experiment(
    variant: str,           # "2d" or "3d"
    scenario: str,          # "no_rot"
    seed: int,
    train_samples: int,
    test_samples: int,
    points: int,
    center: bool,
    batch: int,
    mlp_epochs: int,
    # MVNN knobs
    mvnn_centers: int,
    mvnn_radius_scale: float,
    mvnn_point_radius: float,
    mvnn_bidirectional: bool,
    mvnn_trim_q: float,
):
    print("\n" + "=" * 68)
    print(f"🧪 Experiment: Two Moons {variant.upper()} | scenario={scenario}")
    print("=" * 68 + "\n")

    seed_all(seed)
    rng_tr = np.random.RandomState(seed + 1)
    rng_te = np.random.RandomState(seed + 2)

    print("📌 TRAIN sin rotación → TEST sin rotación (MVNN molecular rígida)\n")

    X_train, y_train = make_two_moons_dataset(
        n_samples=train_samples,
        n_points=points,
        rng=rng_tr,
        variant=variant,
        scenario=scenario,
        translation_scale=0.15,
        outlier_ratio=0.0,
    )
    X_test, y_test = make_two_moons_dataset(
        n_samples=test_samples,
        n_points=points,
        rng=rng_te,
        variant=variant,
        scenario=scenario,
        translation_scale=0.15,
        outlier_ratio=0.0,
    )

    # ---------- MLP ----------
    train_ds = NumpyPointCloudDataset(X_train, y_train, center=center)
    test_ds = NumpyPointCloudDataset(X_test, y_test, center=center)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch, shuffle=False, drop_last=False)

    mlp = PointNetLite(n_classes=2, d_in=3)
    mlp = train_pointnet(mlp, train_loader, test_loader, epochs=mlp_epochs, lr=1e-3, device="cpu")
    mlp_eval = eval_torch_classifier(mlp, test_loader, device="cpu")

    # ---------- VNN ----------
    print("🟠 Extracting VNN features...")
    rng_feat = np.random.RandomState(seed + 3)
    feats_train = np.stack([cloud_features(p, rng=rng_feat) for p in X_train], axis=0)
    feats_test = np.stack([cloud_features(p, rng=rng_feat) for p in X_test], axis=0)

    sample = feats_train[: min(250, len(feats_train))]
    nn_d = []
    for i in range(len(sample)):
        others = np.delete(sample, i, axis=0)
        nn_d.append(float(np.min(np.linalg.norm(others - sample[i], axis=1))))
    base_radius = float(np.median(nn_d) * 1.2 + 1e-6)

    vnn = VNNFeatureSphereClassifier(base_radius=base_radius, lr=0.2, max_spheres_per_class=128)
    vnn.fit(feats_train, y_train)
    vnn_eval = eval_vnn(vnn, feats_test, y_test, n_classes=2)

    # ---------- MVNN Molecular Templates ----------
    print("🧬 Building MVNN Molecular templates...")
    rng_tpl = np.random.RandomState(seed + 123)

    # generamos un pool canónico adicional para templates (sin traslación "fuerte")
    X_tpl, y_tpl = make_two_moons_dataset(
        n_samples=300,
        n_points=points,
        rng=rng_tpl,
        variant=variant,
        scenario="no_rot",
        translation_scale=0.01,
        outlier_ratio=0.0,
    )

    templates: List[Tuple[np.ndarray, np.ndarray]] = []
    for cls in range(2):
        idxs = np.where(y_tpl == cls)[0][:10]  # 10 ejemplos por clase para plantillas
        pts_list = [X_tpl[i] for i in idxs]
        tc, tr = build_template_spheres(pts_list, m=mvnn_centers, rng=rng_tpl, radius_scale=mvnn_radius_scale)
        templates.append((tc, tr))

    mvnn_pred, mvnn_ms = molecular_predict_np(
        templates,
        X_test,
        point_radius=mvnn_point_radius,
        bidirectional=mvnn_bidirectional,
        trim_q=mvnn_trim_q,
    )
    mvnn_acc = float((mvnn_pred == y_test).mean())

    # ---------- Print ----------
    print("\n-------------------- 📊 RESULTS --------------------")
    print(f"🧠 MLP (PointNet-lite)          | acc={mlp_eval['acc']:.3f} | {mlp_eval['ms_per_sample']:.2f} ms/sample")
    print(f"🟠 VNN (feature spheres)        | acc={vnn_eval['acc']:.3f} | {vnn_eval['ms_per_sample']:.2f} ms/sample | spheres={vnn_eval['n_spheres']}")
    print(f"🧬 MVNN Molecular (rigid)       | acc={mvnn_acc:.3f} | {mvnn_ms:.2f} ms/sample | centers={mvnn_centers}")
    print("----------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["two_moons_2d", "two_moons_3d", "both"], default="both")
    parser.add_argument("--scenario", choices=["no_rot"], default="no_rot")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_samples", type=int, default=1500)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--points", type=int, default=128)

    parser.add_argument("--center", type=int, default=1, help="1 = centrar (invariante a traslación), 0 = no centrar")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--mlp_epochs", type=int, default=20)

    # MVNN molecular knobs
    parser.add_argument("--mvnn_centers", type=int, default=32)
    parser.add_argument("--mvnn_radius_scale", type=float, default=1.25)
    parser.add_argument("--mvnn_point_radius", type=float, default=0.18)
    parser.add_argument("--mvnn_bidirectional", type=int, default=1)
    parser.add_argument("--mvnn_trim_q", type=float, default=0.0)

    args = parser.parse_args()
    center = bool(args.center)
    mvnn_bidirectional = bool(args.mvnn_bidirectional)

    if args.dataset in ["two_moons_2d", "both"]:
        run_experiment(
            variant="2d",
            scenario=args.scenario,
            seed=args.seed,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            points=args.points,
            center=center,
            batch=args.batch,
            mlp_epochs=args.mlp_epochs,
            mvnn_centers=args.mvnn_centers,
            mvnn_radius_scale=args.mvnn_radius_scale,
            mvnn_point_radius=args.mvnn_point_radius,
            mvnn_bidirectional=mvnn_bidirectional,
            mvnn_trim_q=args.mvnn_trim_q,
        )

    if args.dataset in ["two_moons_3d", "both"]:
        run_experiment(
            variant="3d",
            scenario=args.scenario,
            seed=args.seed + 100,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            points=args.points,
            center=center,
            batch=args.batch,
            mlp_epochs=args.mlp_epochs,
            mvnn_centers=args.mvnn_centers,
            mvnn_radius_scale=args.mvnn_radius_scale,
            mvnn_point_radius=args.mvnn_point_radius,
            mvnn_bidirectional=mvnn_bidirectional,
            mvnn_trim_q=args.mvnn_trim_q,
        )

    print("😼")


if __name__ == "__main__":
    main()
