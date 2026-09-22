# mvnn_moons_experiment.py
# ------------------------------------------------------------
# Experimento: "5 Moons 3D" con rotación/traslación
# Comparación: (1) Baseline MLP tipo PointNet, (2) VNN-esferas, (3) MVNN (pose search)
#
# Reqs: python>=3.10, numpy, torch
#   pip install numpy torch
#
# Run:
#   python mvnn_moons_experiment.py --scenario canical_train
#   python mvnn_moons_experiment.py --scenario aug_train
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


# -------------------------
# Utils: seed / rotations
# -------------------------
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


def apply_transform_np(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (pts @ R.T) + t


# -------------------------
# Dataset: 5 Moons 3D
# -------------------------
def gen_moon3d(class_id: int, n_points: int, rng: np.random.RandomState, noise: float = 0.02) -> np.ndarray:
    """
    "Moon" = arco semicircular con grosor, warp y un 'twist' suave en z.
    5 clases -> parámetros distintos.
    """
    base_r = 1.0 + 0.15 * class_id
    thickness = 0.18 + 0.02 * class_id

    # offsets en el frame canónico (después centramos igual)
    off = np.array(
        [0.7 * math.cos(2 * math.pi * class_id / 5), 0.7 * math.sin(2 * math.pi * class_id / 5), 0.0],
        dtype=np.float32,
    )

    theta = rng.rand(n_points).astype(np.float32) * math.pi  # [0, pi]
    r = base_r + thickness * (rng.rand(n_points).astype(np.float32) - 0.5)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # warp para que el "moon" sea más distintivo
    y = y + 0.25 * np.sin(2 * theta + class_id)

    # z con onda suave
    z = 0.25 * np.cos(theta * 1.5 + class_id) + 0.05 * (rng.rand(n_points).astype(np.float32) - 0.5)

    pts = np.stack([x, y, z], axis=1).astype(np.float32) + off
    pts += noise * rng.randn(n_points, 3).astype(np.float32)
    return pts


def make_dataset(
    n_samples: int,
    n_points: int,
    rng: np.random.RandomState,
    rotate_translate: bool,
    translation_scale: float = 0.5,
    outlier_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for _ in range(n_samples):
        cls = int(rng.randint(0, 5))
        pts = gen_moon3d(cls, n_points=n_points, rng=rng)

        if rotate_translate:
            q = random_quaternion_np(rng)
            R = quat_to_rot_np(q)
            t = rng.randn(3).astype(np.float32) * translation_scale
            pts = apply_transform_np(pts, R, t)
        else:
            # pequeña traslación para que no sea “perfecto”
            t = rng.randn(3).astype(np.float32) * 0.02
            pts = pts + t

        # outliers (ruido duro)
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
            pts = pts - pts.mean(axis=0, keepdims=True)
        return torch.from_numpy(pts), torch.tensor(self.y[idx], dtype=torch.long)


# -------------------------
# Baseline: PointNet-like MLP
# -------------------------
class PointNetLite(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.per_point = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, pts):  # pts: (B,N,3)
        B, N, _ = pts.shape
        h = self.per_point(pts)  # (B,N,128)
        g = torch.max(h, dim=1).values  # (B,128)
        return self.head(g)


def train_pointnet(model, train_loader, test_loader, epochs=15, lr=1e-3, device="cpu"):
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


# -------------------------
# VNN: esferas en espacio de features (rot-invariant)
# -------------------------
def cloud_features(pts: np.ndarray, rng: np.random.RandomState, n_hist=8, pair_samples=256) -> np.ndarray:
    pts = pts.astype(np.float32)
    c = pts.mean(axis=0, keepdims=True)
    p = pts - c

    # cov eigenvalues (rot-invariant)
    cov = (p.T @ p) / (len(p) + 1e-9)
    evals = np.linalg.eigvalsh(cov).astype(np.float32)
    evals = np.sort(evals)[::-1]  # descending

    # radial stats (rot-invariant)
    r = np.linalg.norm(p, axis=1).astype(np.float32)
    r_mean = float(r.mean())
    r_std = float(r.std() + 1e-9)
    r_skew = float(np.mean(((r - r_mean) / r_std) ** 3))

    # radius histogram
    rmax = float(np.percentile(r, 95))
    bins = np.linspace(0, rmax + 1e-6, n_hist + 1, dtype=np.float32)
    hist, _ = np.histogram(r, bins=bins)
    hist = hist.astype(np.float32) / len(r)

    # pairwise distance quantiles (rot-invariant)
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
    - Cada "neurona" es una esfera en el espacio de features.
    - Score por clase = max_{neurona de clase} relu(1 - dist/r).
    - Aprendizaje online: si cae adentro, actualiza; si cae afuera, crea nueva.
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
                # update center
                self.spheres[j].center = (1 - self.lr) * self.spheres[j].center + self.lr * f
                # expand a bit if needed
                self.spheres[j].radius = max(self.spheres[j].radius, d * 1.05)
            else:
                self.spheres.append(Sphere(center=f.copy(), radius=max(self.base_radius, d * 0.8), label=y))

    def predict(self, feats: np.ndarray) -> np.ndarray:
        preds = []
        for f in feats:
            scores = np.zeros(5, dtype=np.float32)
            for s in self.spheres:
                d = float(np.linalg.norm(f - s.center))
                m = max(0.0, 1.0 - d / (s.radius + 1e-9))
                if m > scores[s.label]:
                    scores[s.label] = m
            preds.append(int(np.argmax(scores)))
        return np.array(preds, dtype=np.int64)


def eval_vnn(vnn: VNNFeatureSphereClassifier, feats: np.ndarray, labels: np.ndarray):
    t0 = time.perf_counter()
    pred = vnn.predict(feats)
    t1 = time.perf_counter()
    acc = float((pred == labels).mean())
    return {"acc": acc, "ms_per_sample": (t1 - t0) * 1000 / len(labels), "n_spheres": len(vnn.spheres)}


# -------------------------
# MVNN: unión de esferas + búsqueda de pose
# -------------------------
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


def build_template_spheres(pts_list: List[np.ndarray], m: int, rng: np.random.RandomState, radius_scale: float = 1.2):
    """
    Template molecular: tomás varios pointclouds de la misma clase (canónicos),
    los unís y construís M centros por farthest point sampling.
    """
    pts = np.concatenate(pts_list, axis=0).astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)

    sel = farthest_point_sampling(pts, m=m, rng=rng)
    centers = pts[sel]  # (M,3)

    # radios = distancia al vecino más cerca entre centros
    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    D = D + np.eye(m, dtype=np.float32) * 1e9
    nn = D.min(axis=1)
    radii = (nn * radius_scale + 1e-6).astype(np.float32)
    return centers.astype(np.float32), radii


def quat_to_rot_torch(q: torch.Tensor) -> torch.Tensor:
    # q = (x,y,z,w)
    q = q / (q.norm() + 1e-9)
    x, y, z, w = q[0], q[1], q[2], q[3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)]),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)]),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]),
        ],
        dim=0,
    )
    return R


def mvnn_score(tc: torch.Tensor, tr: torch.Tensor, X: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    R = quat_to_rot_torch(q)
    Tc = (tc @ R.T) + t  # (M,3)

    # dists (M,N)
    d = torch.cdist(Tc, X)  # (M,N)
    dmin = d.min(dim=1).values  # (M,)
    m = torch.relu(1.0 - dmin / (tr + 1e-6))
    return m.mean()


def random_quaternions_torch(n: int, g: torch.Generator, device: str):
    qs = []
    for _ in range(n):
        u1 = torch.rand((), generator=g, device=device)
        u2 = torch.rand((), generator=g, device=device)
        u3 = torch.rand((), generator=g, device=device)
        q = torch.stack(
            [
                torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2),
                torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2),
                torch.sqrt(u1) * torch.sin(2 * math.pi * u3),
                torch.sqrt(u1) * torch.cos(2 * math.pi * u3),
            ],
            dim=0,
        )
        qs.append(q)
    return qs


def mvnn_best_score_fast(
    template_centers: np.ndarray,
    template_radii: np.ndarray,
    X_np: np.ndarray,
    n_candidates: int = 24,
    preselect: int = 3,
    refine_steps: int = 12,
    lr: float = 0.08,
    device: str = "cpu",
    seed: int = 0,
) -> float:
    """
    1) Samplea N rotaciones (quaternions), score rápido sin grad (t=0)
    2) Refinamiento (Adam) sobre top-K
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    tc = torch.tensor(template_centers, dtype=torch.float32, device=device)
    tr = torch.tensor(template_radii, dtype=torch.float32, device=device)

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    X = X - X.mean(dim=0, keepdim=True)

    cand_q = [torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)]
    cand_q += random_quaternions_torch(n_candidates - 1, g=g, device=device)

    # quick pass
    quick = []
    with torch.no_grad():
        t0 = torch.zeros(3, dtype=torch.float32, device=device)
        for q in cand_q:
            s = float(mvnn_score(tc, tr, X, q, t0).item())
            quick.append(s)

    top_idx = np.argsort(quick)[-preselect:][::-1]

    best = -1.0
    for idx in top_idx:
        q = cand_q[int(idx)].clone().detach().requires_grad_(True)
        t = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)

        opt = optim.Adam([q, t], lr=lr)

        for _ in range(refine_steps):
            opt.zero_grad()
            s = mvnn_score(tc, tr, X, q, t)
            (-s).backward()
            opt.step()
            with torch.no_grad():
                q /= (q.norm() + 1e-9)

        with torch.no_grad():
            s = float(mvnn_score(tc, tr, X, q, t).item())
        best = max(best, s)

    return float(best)


def mvnn_predict(
    templates: List[Tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    n_candidates: int = 24,
    preselect: int = 3,
    refine_steps: int = 12,
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    print(f"🧬 MVNN Predicting on {len(X)} samples (candidates={n_candidates}, refine={refine_steps})...")
    for i, pts in enumerate(X):
        if i % 10 == 0:
            print(f"   [MVNN] Sample {i}/{len(X)}...", end='\r')
            
        best_c = None
        best_s = -1.0
        for c, (tc, tr) in enumerate(templates):
            s = mvnn_best_score_fast(
                tc,
                tr,
                pts,
                n_candidates=n_candidates,
                preselect=preselect,
                refine_steps=refine_steps,
                device=device,
                seed=i * 1009 + c * 9173,
            )
            if s > best_s:
                best_s = s
                best_c = c
        preds.append(int(best_c))
    print(f"   [MVNN] Done!                            ")
    t1 = time.perf_counter()
    ms_per_sample = (t1 - t0) * 1000 / len(X)
    return np.array(preds, dtype=np.int64), ms_per_sample


# -------------------------
# Main experiment
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["canical_train", "aug_train"], default="canical_train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_samples", type=int, default=1500)
    parser.add_argument("--test_samples", type=int, default=400)
    parser.add_argument("--points", type=int, default=128)

    # MVNN knobs (tradeoff accuracy vs speed)
    parser.add_argument("--mvnn_centers", type=int, default=24)
    parser.add_argument("--mvnn_candidates", type=int, default=24)
    parser.add_argument("--mvnn_preselect", type=int, default=3)
    parser.add_argument("--mvnn_refine_steps", type=int, default=12)

    # MLP knobs
    parser.add_argument("--mlp_epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args()

    seed_all(args.seed)

    rng_tr = np.random.RandomState(args.seed + 1)
    rng_te = np.random.RandomState(args.seed + 2)

    # Train scenario
    if args.scenario == "canical_train":
        train_rotate = False
        print("\n🧪 Scenario: TRAIN canónico (sin rotación)  → TEST rotado (stress test de invariancia)\n")
    else:
        train_rotate = True
        print("\n🧪 Scenario: TRAIN con rotación (aug) → TEST rotado (baseline competitivo)\n")

    X_train, y_train = make_dataset(
        n_samples=args.train_samples,
        n_points=args.points,
        rng=rng_tr,
        rotate_translate=train_rotate,
        translation_scale=0.5,
        outlier_ratio=0.00,
    )
    X_test, y_test = make_dataset(
        n_samples=args.test_samples,
        n_points=args.points,
        rng=rng_te,
        rotate_translate=True,
        translation_scale=0.5,
        outlier_ratio=0.00,
    )

    # ----------------- Baseline MLP -----------------
    train_ds = NumpyPointCloudDataset(X_train, y_train, center=True)  # translation-invariant, not rotation-invariant
    test_ds = NumpyPointCloudDataset(X_test, y_test, center=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    mlp = PointNetLite(n_classes=5)
    mlp = train_pointnet(mlp, train_loader, test_loader, epochs=args.mlp_epochs, lr=1e-3, device="cpu")
    mlp_eval = eval_torch_classifier(mlp, test_loader, device="cpu")

    # ----------------- VNN (feature spheres) -----------------
    print("🟠 Extracting VNN features...")
    rng_feat = np.random.RandomState(args.seed + 3)
    feats_train = np.stack([cloud_features(p, rng=rng_feat) for p in X_train], axis=0)
    feats_test = np.stack([cloud_features(p, rng=rng_feat) for p in X_test], axis=0)

    # base radius = mediana de NN distance (estimación rápida)
    sample = feats_train[: min(200, len(feats_train))]
    nn_d = []
    for i in range(len(sample)):
        others = np.delete(sample, i, axis=0)
        nn_d.append(float(np.min(np.linalg.norm(others - sample[i], axis=1))))
    base_radius = float(np.median(nn_d) * 1.2)

    vnn = VNNFeatureSphereClassifier(base_radius=base_radius, lr=0.2, max_spheres_per_class=64)
    vnn.fit(feats_train, y_train)
    vnn_eval = eval_vnn(vnn, feats_test, y_test)

    # ----------------- MVNN (templates + pose search) -----------------
    # Para MVNN conviene plantillas canónicas.
    # Si train está augmentado, igual generamos un pequeño set canónico solo para templates.
    rng_tpl = np.random.RandomState(args.seed + 123)
    X_tpl, y_tpl = make_dataset(
        n_samples=300,
        n_points=args.points,
        rng=rng_tpl,
        rotate_translate=False,
        translation_scale=0.0,
        outlier_ratio=0.00,
    )

    templates: List[Tuple[np.ndarray, np.ndarray]] = []
    for cls in range(5):
        idxs = np.where(y_tpl == cls)[0][:5]  # usar 5 ejemplos canónicos
        pts_list = [X_tpl[i] for i in idxs]
        tc, tr = build_template_spheres(pts_list, m=args.mvnn_centers, rng=rng_tpl, radius_scale=1.2)
        templates.append((tc, tr))

    mvnn_pred, mvnn_ms = mvnn_predict(
        templates,
        X_test,
        n_candidates=args.mvnn_candidates,
        preselect=args.mvnn_preselect,
        refine_steps=args.mvnn_refine_steps,
        device="cpu",
    )
    mvnn_acc = float((mvnn_pred == y_test).mean())

    # ----------------- Print summary -----------------
    print("\n==================== 📊 RESULTS ====================\n")
    print(f"🧠 Baseline MLP (PointNet-lite)  | acc={mlp_eval['acc']:.3f} | {mlp_eval['ms_per_sample']:.2f} ms/sample")
    print(
        f"🟠 VNN (feature spheres)        | acc={vnn_eval['acc']:.3f} | {vnn_eval['ms_per_sample']:.2f} ms/sample"
        f" | spheres={vnn_eval['n_spheres']}"
    )
    print(f"🧬 MVNN (pose search)           | acc={mvnn_acc:.3f} | {mvnn_ms:.2f} ms/sample")
    print("\n====================================================\n")

    print("🔧 Tips:")
    print("  - Si MVNN es lento: bajá --mvnn_candidates (p.ej. 12) y/o --mvnn_refine_steps (p.ej. 8).")
    print("  - Si MLP cae en canical_train: es normal (no es rot-invariant). Probá --scenario aug_train.")
    print("  - VNN acá es fuerte porque usa features rot-invariant (sirve para ver tu idea de esferas sin pose-search).")
    print("\n😼")


if __name__ == "__main__":
    main()
