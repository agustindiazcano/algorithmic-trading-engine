# progressive_moons_compare.py
# ------------------------------------------------------------
# Experimentos progresivos:
#   (1) Two Moons 2D (planar, rotación alrededor de Z)
#   (2) Two Moons 3D (twist en Z, rotación SO(3))
#
# Comparación:
#   - MLP (PointNet-lite)
#   - VNN (esferas en espacio de features rot-invariant)
#   - MVNN Orbital (5 anillos de micro-esferas con giro interno + pose search discreta)
#
# Reqs: python>=3.10, numpy, torch
#   pip install numpy torch
#
# Run examples:
#   python progressive_moons_compare.py --dataset both --scenario canical_train
#   python progressive_moons_compare.py --dataset both --scenario aug_train
# ------------------------------------------------------------

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Utils: seed / rotations
# =========================
def seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_quaternion_np(rng: np.random.RandomState) -> np.ndarray:
    # uniform quaternion (x,y,z,w)
    u1, u2, u3 = rng.rand(3)
    q = np.array(
        [
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
            math.sqrt(u1) * math.cos(2 * math.pi * u3),
        ],
        dtype=np.float32,
    )
    return q


def quat_to_rot_np(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(float(x * x + y * y + z * z + w * w)) + 1e-9
    x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def rot_z_np(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return R


def apply_transform_np(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (pts @ R.T) + t


def sample_rotations_2d(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    Rs = [np.eye(3, dtype=np.float32)]
    for _ in range(n - 1):
        th = float(rng.uniform(0.0, 2 * np.pi))
        Rs.append(rot_z_np(th))
    return np.stack(Rs, axis=0).astype(np.float32)


def sample_rotations_3d(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    Rs = [np.eye(3, dtype=np.float32)]
    for _ in range(n - 1):
        q = random_quaternion_np(rng)
        Rs.append(quat_to_rot_np(q))
    return np.stack(Rs, axis=0).astype(np.float32)


# =========================
# Dataset: Two Moons Clouds
# =========================
def gen_two_moons_cloud(class_id: int, n_points: int, rng: np.random.RandomState, noise: float = 0.02) -> np.ndarray:
    """
    Two Moons 2D (como dataset clásico), pero devolvemos N puntos por clase (point-cloud).
    Se construye un semicirculo + grosor.
    """
    # parámetros
    base_r = 1.0
    thickness = 0.18

    theta = rng.rand(n_points).astype(np.float32) * math.pi  # [0, pi]
    r = base_r + thickness * (rng.rand(n_points).astype(np.float32) - 0.5)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Moon 1: arco superior
    if class_id == 0:
        pts2 = np.stack([x, y], axis=1).astype(np.float32)
    # Moon 2: arco inferior (shift y mirror)
    else:
        pts2 = np.stack([x + 0.5, -y - 0.25], axis=1).astype(np.float32)

    pts2 += noise * rng.randn(n_points, 2).astype(np.float32)

    # elevamos a 3D plano (z=0)
    pts3 = np.concatenate([pts2, np.zeros((n_points, 1), dtype=np.float32)], axis=1)
    return pts3


def add_3d_twist(pts3: np.ndarray, rng: np.random.RandomState, strength: float = 0.35) -> np.ndarray:
    """
    Convierte nube planar en nube 3D con 'twist' en z (dependiendo del ángulo polar).
    """
    p = pts3.copy().astype(np.float32)
    xy = p[:, :2]
    ang = np.arctan2(xy[:, 1], xy[:, 0]).astype(np.float32)  # [-pi, pi]
    z = strength * np.sin(2.0 * ang) + 0.05 * (rng.rand(len(p)).astype(np.float32) - 0.5)
    p[:, 2] = z
    return p


def make_two_moons_dataset(
    n_samples: int,
    n_points: int,
    rng: np.random.RandomState,
    variant: str,  # "2d" or "3d"
    rotate_translate: bool,
    translation_scale: float = 0.5,
    outlier_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    variant="2d": nubes planas, rotación alrededor de Z (si rotate_translate=True)
    variant="3d": nubes con twist, rotación 3D completa (si rotate_translate=True)
    """
    X = []
    y = []
    for _ in range(n_samples):
        cls = int(rng.randint(0, 2))
        pts = gen_two_moons_cloud(cls, n_points=n_points, rng=rng)

        if variant == "3d":
            pts = add_3d_twist(pts, rng=rng, strength=0.35)

        if rotate_translate:
            if variant == "2d":
                # rotación sólo alrededor de Z para mantener planaridad
                th = float(rng.uniform(0.0, 2 * np.pi))
                R = rot_z_np(th)
                t = np.array([rng.randn() * translation_scale, rng.randn() * translation_scale, 0.0], dtype=np.float32)
            else:
                q = random_quaternion_np(rng)
                R = quat_to_rot_np(q)
                t = rng.randn(3).astype(np.float32) * translation_scale
            pts = apply_transform_np(pts, R, t)
        else:
            # pequeña traslación para que no sea perfecto
            t = rng.randn(3).astype(np.float32) * 0.02
            t[2] = 0.0 if variant == "2d" else t[2]
            pts = pts + t

        # outliers duros (opcionales)
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
# Baseline: PointNet-lite MLP
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

    def forward(self, pts):  # pts: (B,N,d_in)
        h = self.per_point(pts)              # (B,N,128)
        g = torch.max(h, dim=1).values       # (B,128)  (perm-invariant)
        return self.head(g)


def train_pointnet(model, train_loader, test_loader, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
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
    correct = 0
    total = 0
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
# VNN: esferas en features rot-invariant
# =========================
def cloud_features(pts: np.ndarray, rng: np.random.RandomState, n_hist=8, pair_samples=256) -> np.ndarray:
    """
    Features rot-invariant:
      - eigenvalues de covarianza (3)
      - stats radiales (mean/std/skew)
      - hist radial
      - cuantiles de distancias entre pares
    """
    pts = pts.astype(np.float32)
    c = pts.mean(axis=0, keepdims=True)
    p = pts - c

    cov = (p.T @ p) / (len(p) + 1e-9)  # 3x3
    evals = np.linalg.eigvalsh(cov).astype(np.float32)
    evals = np.sort(evals)[::-1]  # descending

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
    """
    VNN simplificada:
      - neurona = esfera en espacio de features
      - score por clase = max relu(1 - dist/r)
      - training online (crece si cae afuera)
    """

    def __init__(self, base_radius: float = 0.25, lr: float = 0.2, max_spheres_per_class: int = 64):
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
# MVNN Orbital: 5 rings de micro-esferas + giro interno + pose global
# =========================
def rotate_about_axis_np(P: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    a = axis.astype(np.float32)
    a = a / (np.linalg.norm(a) + 1e-9)
    x, y, z = a[0], a[1], a[2]

    c = np.float32(np.cos(theta))
    s = np.float32(np.sin(theta))
    C = np.float32(1.0 - c)

    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )
    return (P @ R.T).astype(np.float32)


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


@dataclass
class RingTemplate:
    centers: np.ndarray  # (M,3)
    radii: np.ndarray    # (M,)
    axis: np.ndarray     # (3,)
    weight: float = 1.0


@dataclass
class OrbitalTemplate:
    rings: List[RingTemplate]


def _ring_axis_from_points(seg: np.ndarray, fallback_axis: np.ndarray) -> np.ndarray:
    if seg is None or len(seg) < 12:
        a = fallback_axis.astype(np.float32)
        return a / (np.linalg.norm(a) + 1e-9)

    P = seg.astype(np.float32)
    P = P - P.mean(axis=0, keepdims=True)
    cov = (P.T @ P) / (len(P) + 1e-9)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    axis = evecs[:, 0].astype(np.float32)  # normal aprox (menor var)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    return axis


def build_orbital_template_from_points(
    pts_list: List[np.ndarray],
    n_rings: int,
    micro_per_ring: int,
    segment_dim: int,                 # 1 para 2D (y), 2 para 3D (z)
    rng: np.random.RandomState,
    radius_scale: float = 1.25,
    fallback_axis: Optional[np.ndarray] = None,
) -> OrbitalTemplate:
    pts = np.concatenate(pts_list, axis=0).astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)

    if fallback_axis is None:
        fallback_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    v = pts[:, segment_dim].astype(np.float32)
    qs = np.linspace(0.0, 1.0, n_rings + 1)
    cuts = np.quantile(v, qs).astype(np.float32)

    rings: List[RingTemplate] = []
    for k in range(n_rings):
        lo, hi = float(cuts[k]), float(cuts[k + 1])
        mask = (v >= lo) & (v <= hi) if k == n_rings - 1 else (v >= lo) & (v < hi)
        seg = pts[mask]
        if len(seg) < micro_per_ring:
            seg = pts  # fallback

        axis = _ring_axis_from_points(seg, fallback_axis=fallback_axis)

        sel = farthest_point_sampling(seg, m=micro_per_ring, rng=rng)
        centers = seg[sel].astype(np.float32)

        D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2).astype(np.float32)
        D = D + np.eye(micro_per_ring, dtype=np.float32) * 1e9
        nn = D.min(axis=1)
        radii = (nn * radius_scale + 1e-6).astype(np.float32)

        rings.append(RingTemplate(centers=centers, radii=radii, axis=axis, weight=1.0))

    return OrbitalTemplate(rings=rings)


def ring_score_np(ring: RingTemplate, Xc: np.ndarray, Rg: np.ndarray, theta: float) -> float:
    Ck = rotate_about_axis_np(ring.centers, ring.axis, theta)
    Ck = (Ck @ Rg.T).astype(np.float32)

    diff = Ck[:, None, :] - Xc[None, :, :]
    dist2 = np.sum(diff * diff, axis=2).astype(np.float32)
    dmin = np.sqrt(np.min(dist2, axis=1) + 1e-9).astype(np.float32)

    m = np.maximum(0.0, 1.0 - dmin / (ring.radii + 1e-6)).astype(np.float32)
    return float(m.mean())


def aggregate_vl(scores: np.ndarray, weights: np.ndarray, mode: str = "softmax", alpha: float = 20.0) -> float:
    ws = (weights * scores).astype(np.float32)
    if mode == "max":
        return float(np.max(ws))
    if mode == "mean":
        return float(np.mean(ws))

    # softmax ≈ max
    a = float(alpha)
    m = float(np.max(ws))
    return float(m + (np.log(np.sum(np.exp(a * (ws - m))) + 1e-9) / a))


def orbital_best_score_np(
    templ: OrbitalTemplate,
    X: np.ndarray,
    Rs: np.ndarray,
    n_theta: int = 24,
    agg_mode: str = "softmax",
    alpha: float = 20.0,
) -> float:
    Xc = X.astype(np.float32)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)

    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False).astype(np.float32)
    weights = np.array([r.weight for r in templ.rings], dtype=np.float32)

    best = -1.0
    for Rg in Rs:
        ring_scores = []
        for ring in templ.rings:
            s_best = -1.0
            for th in thetas:
                s = ring_score_np(ring, Xc, Rg, float(th))
                if s > s_best:
                    s_best = s
            ring_scores.append(s_best)

        ring_scores = np.array(ring_scores, dtype=np.float32)
        S = aggregate_vl(ring_scores, weights, mode=agg_mode, alpha=alpha)
        if S > best:
            best = S

    return float(best)


def orbital_predict_np(
    templates: List[OrbitalTemplate],
    X: np.ndarray,
    Rs: np.ndarray,
    n_theta: int = 24,
    agg_mode: str = "softmax",
    alpha: float = 20.0,
) -> Tuple[np.ndarray, float]:
    preds = []
    t0 = time.perf_counter()
    for pts in X:
        best_c, best_s = None, -1.0
        for c, templ in enumerate(templates):
            s = orbital_best_score_np(templ, pts, Rs, n_theta=n_theta, agg_mode=agg_mode, alpha=alpha)
            if s > best_s:
                best_s = s
                best_c = c
        preds.append(int(best_c))
    t1 = time.perf_counter()
    return np.array(preds, dtype=np.int64), float((t1 - t0) * 1000 / len(X))


# =========================
# Experiment runner
# =========================
def run_experiment(
    variant: str,           # "2d" or "3d"
    scenario: str,          # "canical_train" or "aug_train"
    seed: int,
    train_samples: int,
    test_samples: int,
    points: int,
    batch: int,
    mlp_epochs: int,
    # MVNN orbital knobs:
    n_rings: int,
    micro_per_ring: int,
    mvnn_rots: int,
    mvnn_theta: int,
    agg_mode: str,
    alpha: float,
):
    print("\n" + "=" * 68)
    print(f"🧪 Experiment: Two Moons {variant.upper()} | scenario={scenario}")
    print("=" * 68 + "\n")

    seed_all(seed)
    rng_tr = np.random.RandomState(seed + 1)
    rng_te = np.random.RandomState(seed + 2)

    if scenario == "canical_train":
        train_rotate = False
        print("📌 TRAIN canónico → TEST rotado (stress test de invariancia)\n")
    else:
        train_rotate = True
        print("📌 TRAIN con rotación (aug) → TEST rotado (baseline competitivo)\n")

    X_train, y_train = make_two_moons_dataset(
        n_samples=train_samples,
        n_points=points,
        rng=rng_tr,
        variant=variant,
        rotate_translate=train_rotate,
        translation_scale=0.5,
        outlier_ratio=0.0,
    )
    X_test, y_test = make_two_moons_dataset(
        n_samples=test_samples,
        n_points=points,
        rng=rng_te,
        variant=variant,
        rotate_translate=True,   # siempre testeamos rotado
        translation_scale=0.5,
        outlier_ratio=0.0,
    )

    # ---------- MLP ----------
    train_ds = NumpyPointCloudDataset(X_train, y_train, center=True)
    test_ds = NumpyPointCloudDataset(X_test, y_test, center=True)

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

    # base radius ~ mediana NN dist en features
    sample = feats_train[: min(250, len(feats_train))]
    nn_d = []
    for i in range(len(sample)):
        others = np.delete(sample, i, axis=0)
        nn_d.append(float(np.min(np.linalg.norm(others - sample[i], axis=1))))
    base_radius = float(np.median(nn_d) * 1.2 + 1e-6)

    vnn = VNNFeatureSphereClassifier(base_radius=base_radius, lr=0.2, max_spheres_per_class=128)
    vnn.fit(feats_train, y_train)
    vnn_eval = eval_vnn(vnn, feats_test, y_test, n_classes=2)

    # ---------- MVNN Orbital templates ----------
    print("🌀 Building MVNN Orbital templates...")
    rng_tpl = np.random.RandomState(seed + 123)

    X_tpl, y_tpl = make_two_moons_dataset(
        n_samples=300,
        n_points=points,
        rng=rng_tpl,
        variant=variant,
        rotate_translate=False,     # plantillas canónicas
        translation_scale=0.0,
        outlier_ratio=0.0,
    )

    segment_dim = 1 if variant == "2d" else 2  # 2D: segmentar por y; 3D: por z
    fallback_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32) if variant == "2d" else None

    orbital_templates: List[OrbitalTemplate] = []
    for cls in range(2):
        idxs = np.where(y_tpl == cls)[0][:8]  # 8 ejemplos canónicos por clase
        pts_list = [X_tpl[i] for i in idxs]
        templ = build_orbital_template_from_points(
            pts_list=pts_list,
            n_rings=n_rings,
            micro_per_ring=micro_per_ring,
            segment_dim=segment_dim,
            rng=rng_tpl,
            radius_scale=1.25,
            fallback_axis=fallback_axis,
        )
        orbital_templates.append(templ)

    Rs = sample_rotations_2d(mvnn_rots, seed=seed + 77) if variant == "2d" else sample_rotations_3d(mvnn_rots, seed=seed + 77)

    orb_pred, orb_ms = orbital_predict_np(
        templates=orbital_templates,
        X=X_test,
        Rs=Rs,
        n_theta=mvnn_theta,
        agg_mode=agg_mode,
        alpha=alpha,
    )
    orb_acc = float((orb_pred == y_test).mean())

    # ---------- Print ----------
    print("\n-------------------- 📊 RESULTS --------------------")
    print(f"🧠 MLP (PointNet-lite)          | acc={mlp_eval['acc']:.3f} | {mlp_eval['ms_per_sample']:.2f} ms/sample")
    print(f"🟠 VNN (feature spheres)        | acc={vnn_eval['acc']:.3f} | {vnn_eval['ms_per_sample']:.2f} ms/sample | spheres={vnn_eval['n_spheres']}")
    print(f"🌀 MVNN Orbital (rings+pose)    | acc={orb_acc:.3f} | {orb_ms:.2f} ms/sample")
    print("----------------------------------------------------\n")

    return {
        "mlp": mlp_eval,
        "vnn": vnn_eval,
        "orb": {"acc": orb_acc, "ms_per_sample": orb_ms},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["two_moons_2d", "two_moons_3d", "both"], default="both")
    parser.add_argument("--scenario", choices=["canical_train", "aug_train"], default="canical_train")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_samples", type=int, default=1500)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--points", type=int, default=128)

    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--mlp_epochs", type=int, default=20)

    # MVNN Orbital knobs
    parser.add_argument("--n_rings", type=int, default=5)
    parser.add_argument("--micro_per_ring", type=int, default=16)
    parser.add_argument("--mvnn_rots", type=int, default=48)
    parser.add_argument("--mvnn_theta", type=int, default=24)
    parser.add_argument("--agg_mode", choices=["max", "softmax", "mean"], default="softmax")
    parser.add_argument("--alpha", type=float, default=20.0)

    args = parser.parse_args()

    if args.dataset in ["two_moons_2d", "both"]:
        run_experiment(
            variant="2d",
            scenario=args.scenario,
            seed=args.seed,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            points=args.points,
            batch=args.batch,
            mlp_epochs=args.mlp_epochs,
            n_rings=args.n_rings,
            micro_per_ring=args.micro_per_ring,
            mvnn_rots=args.mvnn_rots,
            mvnn_theta=args.mvnn_theta,
            agg_mode=args.agg_mode,
            alpha=args.alpha,
        )

    if args.dataset in ["two_moons_3d", "both"]:
        run_experiment(
            variant="3d",
            scenario=args.scenario,
            seed=args.seed + 100,  # offset para que no sea idéntico al 2D
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            points=args.points,
            batch=args.batch,
            mlp_epochs=args.mlp_epochs,
            n_rings=args.n_rings,
            micro_per_ring=args.micro_per_ring,
            mvnn_rots=args.mvnn_rots,
            mvnn_theta=args.mvnn_theta,
            agg_mode=args.agg_mode,
            alpha=args.alpha,
        )

    print("😼")


if __name__ == "__main__":
    main()
