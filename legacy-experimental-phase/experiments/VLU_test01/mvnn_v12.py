# gridworld_cognitive_mvnn.py
# ------------------------------------------------------------
# GridWorld benchmark (online RL, no backprop for MVNN/VNN):
#   - DQN (MLP)
#   - VNN-Q (feature spheres)
#   - MVNN Molecular-Q (rigid templates)
#   - MVNN-Cog (Bayes + Markov mode + Lagrange safety)  ✅
#
# Reqs: python>=3.10, numpy, torch
#   pip install numpy torch
#
# Run:
#   python gridworld_cognitive_mvnn.py --algo cog
#   python gridworld_cognitive_mvnn.py --algo all
# ------------------------------------------------------------

import argparse
import math
import time
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Deque
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# 🔧 Reproducibility
# =========================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 🧱 GridWorld
# =========================
A_UP, A_DOWN, A_LEFT, A_RIGHT = 0, 1, 2, 3
ACTIONS = [A_UP, A_DOWN, A_LEFT, A_RIGHT]
MOVES = {
    A_UP: (-1, 0),
    A_DOWN: (1, 0),
    A_LEFT: (0, -1),
    A_RIGHT: (0, 1),
}


@dataclass
class GridMap:
    grid: np.ndarray  # 0 free, 1 wall
    start: Tuple[int, int]
    goal: Tuple[int, int]


def _neighbors(h, w, r, c):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def is_reachable(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    h, w = grid.shape
    sr, sc = start
    gr, gc = goal
    if grid[sr, sc] == 1 or grid[gr, gc] == 1:
        return False
    q = deque([(sr, sc)])
    seen = set([(sr, sc)])
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return True
        for rr, cc in _neighbors(h, w, r, c):
            if grid[rr, cc] == 0 and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return False


def gen_random_map(
    rng: np.random.RandomState,
    h: int = 11,
    w: int = 11,
    wall_p: float = 0.18,
    max_tries: int = 300,
) -> GridMap:
    for _ in range(max_tries):
        grid = (rng.rand(h, w) < wall_p).astype(np.int32)
        # border walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        free = np.argwhere(grid == 0)
        if len(free) < 10:
            continue
        s = tuple(free[rng.randint(0, len(free))])
        g = tuple(free[rng.randint(0, len(free))])
        if s == g:
            continue

        if is_reachable(grid, s, g):
            return GridMap(grid=grid, start=s, goal=g)

    # fallback: empty with borders
    grid = np.zeros((h, w), dtype=np.int32)
    grid[0, :] = grid[-1, :] = 1
    grid[:, 0] = grid[:, -1] = 1
    return GridMap(grid=grid, start=(1, 1), goal=(h - 2, w - 2))


class GridWorld:
    def __init__(self, gridmap: GridMap, max_steps: int = 120):
        self.map = gridmap
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.pos = tuple(self.map.start)
        self.steps = 0
        return self.pos

    def step(self, action: int):
        self.steps += 1
        r, c = self.pos
        dr, dc = MOVES[action]
        rr, cc = r + dr, c + dc

        reward = -0.01
        done = False
        hit_wall = False

        if self.map.grid[rr, cc] == 1:
            hit_wall = True
            reward -= 0.04
            rr, cc = r, c  # stay

        self.pos = (rr, cc)

        if self.pos == tuple(self.map.goal):
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self.pos, reward, done, {"hit_wall": hit_wall}


# =========================
# 👁️ Features / Symbol
# =========================
def manhattan(p: Tuple[int, int], q: Tuple[int, int]) -> int:
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def state_features(world: GridWorld) -> np.ndarray:
    """
    16D:
      x,y normalized
      dx,dy normalized (goal relative)
      wall sensors 4-dir
      local 3x3 patch occupancy (8)
    """
    grid = world.map.grid
    h, w = grid.shape
    ar, ac = world.pos
    gr, gc = world.map.goal

    x = ar / (h - 1)
    y = ac / (w - 1)
    dx = (gr - ar) / (h - 1)
    dy = (gc - ac) / (w - 1)

    w_up = float(grid[ar - 1, ac] == 1)
    w_dn = float(grid[ar + 1, ac] == 1)
    w_le = float(grid[ar, ac - 1] == 1)
    w_ri = float(grid[ar, ac + 1] == 1)

    patch = []
    for rr in [ar - 1, ar, ar + 1]:
        for cc in [ac - 1, ac, ac + 1]:
            if rr == ar and cc == ac:
                continue
            patch.append(float(grid[rr, cc] == 1))

    return np.array([x, y, dx, dy, w_up, w_dn, w_le, w_ri] + patch, dtype=np.float32)


def symbol_from_feat(feat: np.ndarray, nbins: int = 9) -> int:
    # discretize dx,dy + pack walls+patch bits
    dx = float(feat[2])
    dy = float(feat[3])

    def _bin(v: float) -> int:
        u = (v + 1.0) * 0.5
        b = int(np.floor(u * nbins))
        return int(np.clip(b, 0, nbins - 1))

    dx_b = _bin(dx)
    dy_b = _bin(dy)

    occ_bits = 0
    for i in range(4, 16):
        bit = 1 if feat[i] > 0.5 else 0
        occ_bits |= (bit << (i - 4))

    # pack into int
    sym = (dx_b & 0xF) | ((dy_b & 0xF) << 4) | ((occ_bits & 0xFFF) << 8)
    return int(sym)


# =========================
# 🧬 MVNN Molecular observation
# =========================
def obs_pointcloud(world: GridWorld, radius: int = 3) -> np.ndarray:
    """
    Point cloud around agent (agent-centered coords):
      - wall points (dr,dc, type=0)
      - goal direction point (dgr,dgc, type=1) clipped
    """
    grid = world.map.grid
    h, w = grid.shape
    ar, ac = world.pos
    gr, gc = world.map.goal

    pts = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = ar + dr, ac + dc
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 1:
                pts.append([float(dr), float(dc), 0.0])

    dgr = float(np.clip(gr - ar, -radius, radius))
    dgc = float(np.clip(gc - ac, -radius, radius))
    pts.append([dgr, dgc, 1.0])

    return np.array(pts, dtype=np.float32)


# =========================
# 🧠 DQN (MLP)
# =========================
class QNet(nn.Module):
    def __init__(self, d_in=16, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, cap: int = 50000):
        self.buf: Deque = deque(maxlen=cap)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch: int):
        idx = np.random.randint(0, len(self.buf), size=batch)
        S, A, R, NS, D = zip(*[self.buf[i] for i in idx])
        return (
            np.stack(S).astype(np.float32),
            np.array(A, dtype=np.int64),
            np.array(R, dtype=np.float32),
            np.stack(NS).astype(np.float32),
            np.array(D, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# =========================
# 🟠 VNN-Q (feature spheres)
# =========================
@dataclass
class QSphere:
    center: np.ndarray
    radius: float
    q: np.ndarray
    n: int = 0


class VNNQ:
    def __init__(self, d: int, n_actions: int = 4, base_radius: float = 0.16, lr: float = 0.30, gamma: float = 0.98, max_spheres: int = 4000):
        self.d = d
        self.a = n_actions
        self.base_radius = base_radius
        self.lr = lr
        self.gamma = gamma
        self.max_spheres = max_spheres
        self.spheres: List[QSphere] = []

    def predict_q(self, f: np.ndarray) -> np.ndarray:
        if not self.spheres:
            return np.zeros(self.a, dtype=np.float32)
        weights, qs = [], []
        for sp in self.spheres:
            d = float(np.linalg.norm(f - sp.center))
            m = max(0.0, 1.0 - d / (sp.radius + 1e-9))
            if m > 0:
                weights.append(m)
                qs.append(sp.q)
        if not weights:
            return np.zeros(self.a, dtype=np.float32)
        w = np.array(weights, dtype=np.float32)
        Q = np.stack(qs, axis=0)
        w = w / (w.sum() + 1e-9)
        return (w[:, None] * Q).sum(axis=0).astype(np.float32)

    def _best_idx(self, f: np.ndarray):
        best_i, best_d = -1, 1e9
        for i, sp in enumerate(self.spheres):
            d = float(np.linalg.norm(f - sp.center))
            if d < best_d:
                best_d = d
                best_i = i
        return best_i, best_d

    def update(self, f: np.ndarray, a: int, r: float, nf: np.ndarray, done: bool, hit_wall: bool):
        # fear (more penalty)
        if hit_wall:
            r = float(r - 0.08)
        target = r if done else (r + self.gamma * float(np.max(self.predict_q(nf))))

        if not self.spheres:
            sp = QSphere(center=f.copy(), radius=float(self.base_radius), q=np.zeros(self.a, dtype=np.float32), n=1)
            sp.q[a] = target
            self.spheres.append(sp)
            return

        i, dmin = self._best_idx(f)
        sp = self.spheres[i]

        if dmin > sp.radius and len(self.spheres) < self.max_spheres:
            rad = max(self.base_radius, 0.8 * dmin)
            q0 = np.zeros(self.a, dtype=np.float32)
            q0[a] = target
            self.spheres.append(QSphere(center=f.copy(), radius=float(rad), q=q0, n=1))
            return

        sp.n += 1
        alpha = self.lr / (1.0 + 0.002 * sp.n)
        sp.q[a] = (1 - alpha) * sp.q[a] + alpha * target
        sp.center = 0.95 * sp.center + 0.05 * f
        sp.radius = max(sp.radius, 1.05 * dmin)


# =========================
# 🧬 MVNN Molecular core
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


def build_template_spheres(pts_pool: np.ndarray, m: int, rng: np.random.RandomState, radius_scale: float = 1.25) -> Tuple[np.ndarray, np.ndarray]:
    pts = pts_pool.astype(np.float32)
    if len(pts) < m:
        reps = int(np.ceil(m / max(1, len(pts))))
        pts = np.tile(pts, (reps, 1))[:m]

    sel = farthest_point_sampling(pts, m=m, rng=rng)
    centers = pts[sel].astype(np.float32)

    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2).astype(np.float32)
    D = D + np.eye(m, dtype=np.float32) * 1e9
    nn = D.min(axis=1)
    radii = (nn * radius_scale + 1e-6).astype(np.float32)
    return centers, radii


def molecular_score(tc: np.ndarray, tr: np.ndarray, X: np.ndarray, point_radius: float = 1.8, bidirectional: bool = True) -> float:
    if X is None or len(X) == 0:
        return 0.0

    A = tc.astype(np.float32)
    B = X.astype(np.float32)

    diff = A[:, None, :] - B[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    dmin = np.sqrt(np.min(dist2, axis=1) + 1e-9)
    m1 = np.maximum(0.0, 1.0 - dmin / (tr + 1e-6)).astype(np.float32)

    if not bidirectional:
        return float(m1.mean())

    diff2 = B[:, None, :] - A[None, :, :]
    dist2b = np.sum(diff2 * diff2, axis=2)
    dmin2 = np.sqrt(np.min(dist2b, axis=1) + 1e-9)
    m2 = np.maximum(0.0, 1.0 - dmin2 / (point_radius + 1e-6)).astype(np.float32)

    return 0.5 * float(m1.mean()) + 0.5 * float(m2.mean())


# =========================
# 🧬 MVNN Molecular-Q baseline
# =========================
class MVNNMolecularQ:
    def __init__(self, n_actions: int = 4, centers: int = 32, gamma: float = 0.98, rebuild_every: int = 180, pool_cap: int = 9000, seed: int = 0):
        self.a = n_actions
        self.centers = centers
        self.gamma = gamma
        self.rebuild_every = rebuild_every
        self.rng = np.random.RandomState(seed + 999)

        self.pools: List[Deque[np.ndarray]] = [deque(maxlen=pool_cap) for _ in range(n_actions)]
        self.tc: List[Optional[np.ndarray]] = [None for _ in range(n_actions)]
        self.tr: List[Optional[np.ndarray]] = [None for _ in range(n_actions)]
        self.vbias = np.zeros(n_actions, dtype=np.float32)
        self.vlr = 0.12
        self.updates = 0

    def _ensure(self, a: int):
        if self.tc[a] is not None:
            return
        dummy = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        self.tc[a], self.tr[a] = build_template_spheres(dummy, m=self.centers, rng=self.rng, radius_scale=1.25)

    def predict_q(self, obs_pc: np.ndarray) -> np.ndarray:
        Q = np.zeros(self.a, dtype=np.float32)
        for a in range(self.a):
            self._ensure(a)
            s = molecular_score(self.tc[a], self.tr[a], obs_pc, point_radius=1.8, bidirectional=True)
            Q[a] = float(s) + float(self.vbias[a])
        return Q

    def update(self, obs_pc: np.ndarray, a: int, r: float, next_obs_pc: np.ndarray, done: bool, hit_wall: bool):
        if hit_wall:
            r = float(r - 0.08)

        q = self.predict_q(obs_pc)
        nq = self.predict_q(next_obs_pc)
        target = r if done else (r + self.gamma * float(np.max(nq)))
        td = target - float(q[a])
        self.vbias[a] += self.vlr * float(td)

        if (td > 0.0) and (not hit_wall) and len(obs_pc) > 0:
            self.pools[a].append(obs_pc.astype(np.float32))

        self.updates += 1
        if self.updates % self.rebuild_every == 0 and len(self.pools[a]) > 10:
            pool_pts = np.concatenate(list(self.pools[a]), axis=0)
            self.tc[a], self.tr[a] = build_template_spheres(pool_pts, m=self.centers, rng=self.rng, radius_scale=1.25)


# =========================
# 🧬🧠 MVNN Cognitive Unit (Bayes + Markov + Lagrange)
# =========================
class MVNNCog:
    """
    Cognitive MVNN:
      - MVNN templates define likelihoods for:
          * hit_wall vs safe  (per action)
          * progress vs non-progress (per action)
      - Bayes posterior gives p_hit(a|obs) and p_good(a|obs)
      - Markov latent mode: NORMAL vs TRAP (2-state filter)
      - Lagrange: chooses action maximizing utility - lambda * p_hit, updates lambda online
    """
    def __init__(
        self,
        centers: int = 32,
        radius_scale: float = 1.25,
        point_radius: float = 1.8,
        beta: float = 8.0,  # likelihood temperature
        seed: int = 0,
        rebuild_every: int = 160,
        pool_cap: int = 9000,
        gamma: float = 0.98,
        lr_q: float = 0.25,
        recent_len: int = 28,
    ):
        self.a = 4
        self.centers = centers
        self.radius_scale = radius_scale
        self.point_radius = point_radius
        self.beta = beta
        self.rng = np.random.RandomState(seed + 404)
        self.rebuild_every = rebuild_every

        # Pools per action (pointcloud examples)
        self.pool_safe = [deque(maxlen=pool_cap) for _ in range(self.a)]
        self.pool_hit  = [deque(maxlen=pool_cap) for _ in range(self.a)]
        self.pool_good = [deque(maxlen=pool_cap) for _ in range(self.a)]
        self.pool_bad  = [deque(maxlen=pool_cap) for _ in range(self.a)]

        # Templates per action for each pool type
        self.tc_safe = [None] * self.a
        self.tr_safe = [None] * self.a
        self.tc_hit  = [None] * self.a
        self.tr_hit  = [None] * self.a
        self.tc_good = [None] * self.a
        self.tr_good = [None] * self.a
        self.tc_bad  = [None] * self.a
        self.tr_bad  = [None] * self.a

        # Priors counts (for Bayes)
        self.n_safe = np.zeros(self.a, dtype=np.int64)
        self.n_hit  = np.zeros(self.a, dtype=np.int64)
        self.n_good = np.zeros(self.a, dtype=np.int64)
        self.n_bad  = np.zeros(self.a, dtype=np.int64)

        # Online symbol-Q (small, caja blanca)
        self.Q: Dict[int, np.ndarray] = {}
        self.visits: Dict[int, int] = {}
        self.gamma = gamma
        self.lr_q = lr_q
        self.recent: Deque[int] = deque(maxlen=recent_len)

        # Markov mode (NORMAL=0, TRAP=1)
        self.belief = np.array([0.8, 0.2], dtype=np.float32)
        self.A = np.array([[0.92, 0.08],
                           [0.10, 0.90]], dtype=np.float32)  # transitions

        # Lagrange multiplier for safety constraint
        self.lam = 0.0
        self.lam_lr = 0.02
        self.hit_target = 0.01  # desired hit rate per step (near-zero)

        # Weights (interpretable)
        self.w_q = 1.0
        self.w_good = 1.2
        self.w_bad = 0.7
        self.w_cycle = 0.6
        self.w_visit = 0.05
        self.w_goal = 0.35  # mild goal gravity (doesn't leak map)
        self.w_escape = 0.55  # when TRAP mode, anti-gravity strength
        self.wall_mask_hard = True

        self.updates = 0

    def _q(self, sym: int) -> np.ndarray:
        if sym not in self.Q:
            self.Q[sym] = np.zeros(self.a, dtype=np.float32)
        return self.Q[sym]

    def _ensure_template(self, pool: Deque[np.ndarray], tc_ref, tr_ref, a: int):
        if tc_ref[a] is not None:
            return
        dummy = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        tc_ref[a], tr_ref[a] = build_template_spheres(dummy, m=self.centers, rng=self.rng, radius_scale=self.radius_scale)

    def _maybe_rebuild(self, pool: Deque[np.ndarray], tc_ref, tr_ref, a: int):
        if len(pool) < 20:
            return
        pool_pts = np.concatenate(list(pool), axis=0)
        tc_ref[a], tr_ref[a] = build_template_spheres(pool_pts, m=self.centers, rng=self.rng, radius_scale=self.radius_scale)

    def _score(self, tc_ref, tr_ref, a: int, obs_pc: np.ndarray) -> float:
        self._ensure_template(self.pool_safe, self.tc_safe, self.tr_safe, a)  # ensures at least something exists somewhere
        if tc_ref[a] is None:
            # make sure this specific template exists
            dummy = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            tc_ref[a], tr_ref[a] = build_template_spheres(dummy, m=self.centers, rng=self.rng, radius_scale=self.radius_scale)
        return molecular_score(tc_ref[a], tr_ref[a], obs_pc, point_radius=self.point_radius, bidirectional=True)

    def _bayes_binary(self, s_pos: float, s_neg: float, prior_pos: float) -> float:
        # posterior p(pos | obs) with likelihoods exp(beta*score)
        Lp = math.exp(self.beta * float(s_pos))
        Ln = math.exp(self.beta * float(s_neg))
        num = Lp * prior_pos
        den = num + Ln * (1.0 - prior_pos) + 1e-12
        return float(num / den)

    def p_hit(self, obs_pc: np.ndarray, a: int) -> float:
        # p(hit_wall | obs, a)
        alpha = 1.0  # smoothing
        prior_hit = float((self.n_hit[a] + alpha) / (self.n_hit[a] + self.n_safe[a] + 2 * alpha + 1e-9))
        s_hit = self._score(self.tc_hit, self.tr_hit, a, obs_pc)
        s_safe = self._score(self.tc_safe, self.tr_safe, a, obs_pc)
        return self._bayes_binary(s_hit, s_safe, prior_hit)

    def p_good(self, obs_pc: np.ndarray, a: int) -> float:
        # p(progress | obs, a)
        alpha = 1.0
        prior_good = float((self.n_good[a] + alpha) / (self.n_good[a] + self.n_bad[a] + 2 * alpha + 1e-9))
        s_good = self._score(self.tc_good, self.tr_good, a, obs_pc)
        s_bad = self._score(self.tc_bad, self.tr_bad, a, obs_pc)
        return self._bayes_binary(s_good, s_bad, prior_good)

    def _markov_update(self, feat: np.ndarray, sym: int):
        # emission heuristic for TRAP
        walls = int(feat[4] + feat[5] + feat[6] + feat[7])
        loop = 1.0 if sym in self.recent else 0.0

        # TRAP likelihood: high if (walls>=3) or loop
        # keep it soft, this is just to demonstrate Markov control.
        e_trap = 0.15
        e_trap += 0.25 * float(walls >= 3)
        e_trap += 0.40 * float(loop > 0.0)
        e_trap = float(np.clip(e_trap, 0.05, 0.95))
        e_norm = 1.0 - e_trap

        # predict
        pred = self.A.T @ self.belief
        post = pred * np.array([e_norm, e_trap], dtype=np.float32)
        post = post / (post.sum() + 1e-9)
        self.belief = post

    def act(self, world: GridWorld) -> Tuple[int, int]:
        feat = state_features(world)
        sym = symbol_from_feat(feat)
        self._markov_update(feat, sym)

        obs = obs_pointcloud(world, radius=3)

        # goal pull from dx/dy only (allowed)
        dx = float(feat[2])
        dy = float(feat[3])

        def goal_heuristic(a: int) -> float:
            # encourages moving reducing |dx| or |dy| without reading the whole map
            if a == A_UP:
                return +1.0 if dx < 0 else -0.2
            if a == A_DOWN:
                return +1.0 if dx > 0 else -0.2
            if a == A_LEFT:
                return +1.0 if dy < 0 else -0.2
            if a == A_RIGHT:
                return +1.0 if dy > 0 else -0.2
            return 0.0

        # If TRAP belief high, allow "anti-goal" exploration (antigravity)
        b_trap = float(self.belief[1])

        best_a, best_score = 0, -1e18
        q_sym = self._q(sym)

        for a in ACTIONS:
            # hard wall mask from sensors (fear prior)
            if self.wall_mask_hard:
                if a == A_UP and feat[4] > 0.5:  # up wall
                    continue
                if a == A_DOWN and feat[5] > 0.5:
                    continue
                if a == A_LEFT and feat[6] > 0.5:
                    continue
                if a == A_RIGHT and feat[7] > 0.5:
                    continue

            ph = self.p_hit(obs, a)
            pg = self.p_good(obs, a)

            # cycle/visit on next symbol (approx: treat same symbol as cost)
            cyc = 1.0 if sym in self.recent else 0.0
            vcnt = float(self.visits.get(sym, 0))

            g = goal_heuristic(a)
            # gravity + antigravity blend by Markov TRAP belief
            grav_term = (1.0 - b_trap) * self.w_goal * g - b_trap * self.w_escape * g

            score = 0.0
            score += self.w_q * float(q_sym[a])
            score += self.w_good * float(pg)
            score -= self.w_bad * float(1.0 - pg)  # "badness"
            score += grav_term

            # Lagrange safety: punish predicted collision probability
            score -= float(self.lam) * float(ph)

            # logic priors
            score -= self.w_cycle * float(cyc)
            score -= self.w_visit * float(vcnt)

            if score > best_score:
                best_score = score
                best_a = a

        return best_a, sym

    def update(self, sym: int, a: int, r: float, nsym: int, done: bool, hit_wall: bool, obs_pc: np.ndarray, dist_before: int, dist_after: int):
        # ---- Lagrange update (safety constraint) ----
        violation = 1.0 if hit_wall else 0.0
        self.lam = max(0.0, float(self.lam + self.lam_lr * (violation - self.hit_target)))

        # ---- Q update in symbol space (online, caja blanca) ----
        q = self._q(sym)
        nq = self._q(nsym)
        if hit_wall:
            r = float(r - 0.10)  # extra fear shaping

        target = r if done else (r + self.gamma * float(np.max(nq)))
        td = target - float(q[a])
        q[a] = float(q[a] + self.lr_q * td)

        # ---- Update Bayes pools/templates ----
        # Safe / Hit
        if hit_wall:
            self.n_hit[a] += 1
            if len(obs_pc) > 0:
                self.pool_hit[a].append(obs_pc.astype(np.float32))
        else:
            self.n_safe[a] += 1
            if len(obs_pc) > 0:
                self.pool_safe[a].append(obs_pc.astype(np.float32))

        # Good / Bad by progress signal (distance decreased)
        progressed = (dist_after < dist_before) and (not hit_wall)
        if progressed:
            self.n_good[a] += 1
            if len(obs_pc) > 0:
                self.pool_good[a].append(obs_pc.astype(np.float32))
        else:
            self.n_bad[a] += 1
            if len(obs_pc) > 0:
                self.pool_bad[a].append(obs_pc.astype(np.float32))

        self.visits[nsym] = int(self.visits.get(nsym, 0) + 1)
        self.recent.append(sym)

        self.updates += 1
        if self.updates % self.rebuild_every == 0:
            for aa in range(self.a):
                self._maybe_rebuild(self.pool_safe[aa], self.tc_safe, self.tr_safe, aa)
                self._maybe_rebuild(self.pool_hit[aa],  self.tc_hit,  self.tr_hit,  aa)
                self._maybe_rebuild(self.pool_good[aa], self.tc_good, self.tr_good, aa)
                self._maybe_rebuild(self.pool_bad[aa],  self.tc_bad,  self.tr_bad,  aa)


# =========================
# 🎮 Eval helpers
# =========================
def epsilon_by_step(step: int, eps_start=1.0, eps_end=0.05, decay=25000) -> float:
    return float(eps_end + (eps_start - eps_end) * math.exp(-step / decay))


def run_episode(env: GridWorld, policy_fn, max_steps: int = 120):
    env.reset()
    total = 0.0
    wall_hits = 0
    for _ in range(max_steps):
        a = policy_fn(env)
        _, r, done, info = env.step(a)
        total += r
        wall_hits += int(info.get("hit_wall", False))
        if done:
            break
    success = (env.pos == env.map.goal)
    return total, success, env.steps, wall_hits


def evaluate_agent(env_maps: List[GridMap], make_policy, n_episodes: int = 100) -> dict:
    successes = 0
    steps_sum = 0
    rew_sum = 0.0
    wall_sum = 0
    for i in range(n_episodes):
        gm = env_maps[i % len(env_maps)]
        env = GridWorld(gm, max_steps=120)
        policy = make_policy()
        R, ok, steps, wh = run_episode(env, policy_fn=policy, max_steps=120)
        rew_sum += R
        steps_sum += steps
        wall_sum += wh
        successes += int(ok)
    return {
        "success": successes / n_episodes,
        "avg_steps": steps_sum / n_episodes,
        "avg_return": rew_sum / n_episodes,
        "avg_wall_hits": wall_sum / n_episodes,
    }


# =========================
# 🧠 Train DQN
# =========================
def train_dqn(train_maps, test_maps, seed: int, steps: int = 60000, eval_every: int = 5000):
    device = "cpu"
    q = QNet(d_in=16, n_actions=4).to(device)
    tq = QNet(d_in=16, n_actions=4).to(device)
    tq.load_state_dict(q.state_dict())
    tq.eval()

    opt = optim.Adam(q.parameters(), lr=1e-3)
    rb = ReplayBuffer(cap=50000)
    gamma = 0.98
    batch = 128
    warmup = 1500
    target_sync = 1500

    def greedy_action(feat_np):
        with torch.no_grad():
            x = torch.tensor(feat_np[None, :], dtype=torch.float32, device=device)
            return int(torch.argmax(q(x), dim=1).item())

    env = GridWorld(train_maps[0], max_steps=120)
    env.reset()
    last_eval = 0
    solved_step = None
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        if step % 240 == 1:
            env = GridWorld(train_maps[np.random.randint(0, len(train_maps))], max_steps=120)
            env.reset()

        eps = epsilon_by_step(step)
        s = state_features(env)
        a = np.random.randint(0, 4) if np.random.rand() < eps else greedy_action(s)

        _, r, done, _ = env.step(a)
        ns = state_features(env)
        rb.push(s, a, r, ns, done)
        if done:
            env.reset()

        if len(rb) >= warmup:
            S, A, R, NS, D = rb.sample(batch)
            S = torch.tensor(S, dtype=torch.float32, device=device)
            A = torch.tensor(A, dtype=torch.int64, device=device)
            R = torch.tensor(R, dtype=torch.float32, device=device)
            NS = torch.tensor(NS, dtype=torch.float32, device=device)
            D = torch.tensor(D, dtype=torch.float32, device=device)

            with torch.no_grad():
                target = R + (1.0 - D) * gamma * torch.max(tq(NS), dim=1).values
            pred = q(S).gather(1, A.view(-1, 1)).squeeze(1)
            loss = torch.mean((pred - target) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if step % target_sync == 0:
            tq.load_state_dict(q.state_dict())

        if step - last_eval >= eval_every:
            last_eval = step

            def make_policy():
                def pol(env_):
                    f = state_features(env_)
                    return greedy_action(f)
                return pol

            tr = evaluate_agent(train_maps, make_policy, n_episodes=80)
            te = evaluate_agent(test_maps, make_policy, n_episodes=80)

            print(f"[DQN] step={step:6d} eps={eps:.3f} | train={tr['success']:.2f} test={te['success']:.2f} | wall={te['avg_wall_hits']:.1f}")
            if solved_step is None and tr["success"] >= 0.90:
                solved_step = step

    t1 = time.perf_counter()
    return {"solved_step": solved_step, "ms_per_step": (t1 - t0) * 1000 / steps}


# =========================
# 🟠 Train VNN-Q
# =========================
def train_vnn(train_maps, test_maps, seed: int, steps: int = 60000, eval_every: int = 5000):
    vnn = VNNQ(d=16, n_actions=4, base_radius=0.16, lr=0.30, gamma=0.98, max_spheres=4000)

    env = GridWorld(train_maps[0], max_steps=120)
    env.reset()
    last_eval = 0
    solved_step = None
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        if step % 240 == 1:
            env = GridWorld(train_maps[np.random.randint(0, len(train_maps))], max_steps=120)
            env.reset()

        eps = epsilon_by_step(step, decay=22000)
        s = state_features(env)
        Q = vnn.predict_q(s)
        a = np.random.randint(0, 4) if np.random.rand() < eps else int(np.argmax(Q))

        _, r, done, info = env.step(a)
        ns = state_features(env)
        vnn.update(s, a, r, ns, done, hit_wall=bool(info.get("hit_wall", False)))

        if done:
            env.reset()

        if step - last_eval >= eval_every:
            last_eval = step

            def make_policy():
                def pol(env_):
                    f = state_features(env_)
                    return int(np.argmax(vnn.predict_q(f)))
                return pol

            tr = evaluate_agent(train_maps, make_policy, n_episodes=80)
            te = evaluate_agent(test_maps, make_policy, n_episodes=80)

            print(f"[VNN-Q] step={step:6d} eps={eps:.3f} | train={tr['success']:.2f} test={te['success']:.2f} | spheres={len(vnn.spheres)} wall={te['avg_wall_hits']:.1f}")
            if solved_step is None and tr["success"] >= 0.90:
                solved_step = step

    t1 = time.perf_counter()
    return {"solved_step": solved_step, "ms_per_step": (t1 - t0) * 1000 / steps, "spheres": len(vnn.spheres)}


# =========================
# 🧬 Train MVNN Molecular-Q
# =========================
def train_mvnn(train_maps, test_maps, seed: int, steps: int = 60000, eval_every: int = 5000):
    mv = MVNNMolecularQ(n_actions=4, centers=32, gamma=0.98, rebuild_every=180, pool_cap=9000, seed=seed)

    env = GridWorld(train_maps[0], max_steps=120)
    env.reset()
    last_eval = 0
    solved_step = None
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        if step % 240 == 1:
            env = GridWorld(train_maps[np.random.randint(0, len(train_maps))], max_steps=120)
            env.reset()

        eps = epsilon_by_step(step, decay=22000)
        obs = obs_pointcloud(env, radius=3)
        Q = mv.predict_q(obs)
        a = np.random.randint(0, 4) if np.random.rand() < eps else int(np.argmax(Q))

        _, r, done, info = env.step(a)
        nobs = obs_pointcloud(env, radius=3)
        mv.update(obs, a, r, nobs, done, hit_wall=bool(info.get("hit_wall", False)))
        if done:
            env.reset()

        if step - last_eval >= eval_every:
            last_eval = step

            def make_policy():
                def pol(env_):
                    o = obs_pointcloud(env_, radius=3)
                    return int(np.argmax(mv.predict_q(o)))
                return pol

            tr = evaluate_agent(train_maps, make_policy, n_episodes=80)
            te = evaluate_agent(test_maps, make_policy, n_episodes=80)
            pool_sz = sum(len(p) for p in mv.pools)

            print(f"[MVNN-M] step={step:6d} eps={eps:.3f} | train={tr['success']:.2f} test={te['success']:.2f} | pool={pool_sz} wall={te['avg_wall_hits']:.1f}")
            if solved_step is None and tr["success"] >= 0.90:
                solved_step = step

    t1 = time.perf_counter()
    return {"solved_step": solved_step, "ms_per_step": (t1 - t0) * 1000 / steps}


# =========================
# 🧬🧠 Train MVNN-Cog
# =========================
def train_cog(train_maps, test_maps, seed: int, steps: int = 60000, eval_every: int = 5000):
    cog = MVNNCog(seed=seed, centers=32, beta=8.0, rebuild_every=160)

    env = GridWorld(train_maps[0], max_steps=120)
    env.reset()
    last_eval = 0
    solved_step = None
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        if step % 240 == 1:
            env = GridWorld(train_maps[np.random.randint(0, len(train_maps))], max_steps=120)
            env.reset()

        eps = epsilon_by_step(step, decay=18000)  # already has priors
        feat = state_features(env)
        sym = symbol_from_feat(feat)
        obs = obs_pointcloud(env, radius=3)
        dist_before = abs(int(round(feat[2] * (env.map.grid.shape[0] - 1)))) + abs(int(round(feat[3] * (env.map.grid.shape[1] - 1))))

        a, sym_used = cog.act(env)
        if np.random.rand() < eps:
            a = np.random.randint(0, 4)

        _, r, done, info = env.step(a)
        hit_wall = bool(info.get("hit_wall", False))

        nfeat = state_features(env)
        nsym = symbol_from_feat(nfeat)
        dist_after = abs(int(round(nfeat[2] * (env.map.grid.shape[0] - 1)))) + abs(int(round(nfeat[3] * (env.map.grid.shape[1] - 1))))

        cog.update(sym_used, a, r, nsym, done, hit_wall, obs, dist_before, dist_after)

        if done:
            env.reset()

        if step - last_eval >= eval_every:
            last_eval = step

            def make_policy():
                def pol(env_):
                    a2, _ = cog.act(env_)
                    return a2
                return pol

            tr = evaluate_agent(train_maps, make_policy, n_episodes=80)
            te = evaluate_agent(test_maps, make_policy, n_episodes=80)

            print(
                f"[MVNN-COG] step={step:6d} eps={eps:.3f} | train={tr['success']:.2f} test={te['success']:.2f} | "
                f"wall={te['avg_wall_hits']:.1f} lam={cog.lam:.2f} belief(TRAP)={cog.belief[1]:.2f} symQ={len(cog.Q)}"
            )
            if solved_step is None and tr["success"] >= 0.90:
                solved_step = step

    t1 = time.perf_counter()
    # templates size proxy
    pool_sz = sum(len(p) for p in cog.pool_safe) + sum(len(p) for p in cog.pool_hit)
    return {"solved_step": solved_step, "ms_per_step": (t1 - t0) * 1000 / steps, "pool": pool_sz, "symQ": len(cog.Q)}


# =========================
# 🚀 Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "vnn", "mvnn", "cog", "all"], default="cog")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--h", type=int, default=11)
    parser.add_argument("--w", type=int, default=11)
    parser.add_argument("--wall_p", type=float, default=0.18)
    parser.add_argument("--train_maps", type=int, default=24)
    parser.add_argument("--test_maps", type=int, default=24)
    args = parser.parse_args()

    seed_all(args.seed)
    rng = np.random.RandomState(args.seed + 123)

    train_maps = [gen_random_map(rng, h=args.h, w=args.w, wall_p=args.wall_p) for _ in range(args.train_maps)]
    test_maps = [gen_random_map(rng, h=args.h, w=args.w, wall_p=args.wall_p) for _ in range(args.test_maps)]

    print("\n==================== 🧪 GridWorld Benchmark ====================")
    print(f"maps: train={len(train_maps)} test={len(test_maps)} | size={args.h}x{args.w} | wall_p={args.wall_p}")
    print(f"steps={args.steps} eval_every={args.eval_every}\n")

    results = {}

    if args.algo in ["dqn", "all"]:
        print("🧠 Training DQN (MLP)...")
        results["dqn"] = train_dqn(train_maps, test_maps, seed=args.seed, steps=args.steps, eval_every=args.eval_every)

    if args.algo in ["vnn", "all"]:
        print("\n🟠 Training VNN-Q (spheres)...")
        results["vnn"] = train_vnn(train_maps, test_maps, seed=args.seed, steps=args.steps, eval_every=args.eval_every)

    if args.algo in ["mvnn", "all"]:
        print("\n🧬 Training MVNN Molecular-Q...")
        results["mvnn"] = train_mvnn(train_maps, test_maps, seed=args.seed, steps=args.steps, eval_every=args.eval_every)

    if args.algo in ["cog", "all"]:
        print("\n🧬🧠 Training MVNN-COG (Bayes+Markov+Lagrange)...")
        results["cog"] = train_cog(train_maps, test_maps, seed=args.seed, steps=args.steps, eval_every=args.eval_every)

    print("\n==================== 📊 SUMMARY ====================")
    for k, v in results.items():
        solved = v.get("solved_step", None)
        solved_str = "not reached" if solved is None else str(solved)
        extra = ""
        if "spheres" in v:
            extra += f" | spheres={v['spheres']}"
        if "pool" in v:
            extra += f" | pool={v['pool']}"
        if "symQ" in v:
            extra += f" | symQ={v['symQ']}"
        print(f"{k.upper():>4s} | solved_step={solved_str:<10s} | {v['ms_per_step']:.3f} ms/step{extra}")
    print("====================================================\n😼")


if __name__ == "__main__":
    main()
