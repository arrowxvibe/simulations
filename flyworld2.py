"""Flight + walking world v2 for the real FlyWire brain.

Senses (all fed into real FlyWire neurons):
  vision        24x16 picture on the 11,350 photoreceptors: berries, plus every solid thing
                (the human, other flies, ants, trees, rocks) with apparent size growing as they approach
  smell         berry odour at the left / right antenna -> olfactory receptor neurons (by side)
  mechanosense  air-buffeting from the human and other flies' wingbeats -> mechanosensory neurons
Reflex: if the real DNp01 giant-fibre neurons fire (driven by the looming detectors LPLC2 / LC4 in the connectome),
        the fly takes off and flees away from the side the looming came from. Nothing about this is scripted.
Body:   airborne flight, or walking on legs when on the ground (agl < 0.35).
"""
import numpy as np

ARENA, K, W, H = 120.0, 3, 24, 16
FOV = np.radians(100.0); FC = 1.0 / np.tan(FOV / 2)
SPEED_AIR, SPEED_WALK, TURN, CLIMB, SINK = 0.25, 0.10, 0.12, 0.15, 0.05
LAND_R, ESC_STEPS, ESC_COOLDOWN = 0.9, 18, 40
# solids: kind, radius, centre height
KINDS = [("human", 2.2, 3.0), ("fly", 0.5, 2.0), ("fly", 0.5, 2.0), ("ant", 0.9, 0.5), ("ant", 0.9, 0.5),
         ("tree", 2.5, 6.0), ("tree", 2.5, 6.0), ("tree", 2.5, 6.0), ("rock", 1.0, 0.5), ("rock", 1.0, 0.5), ("rock", 1.0, 0.5)]
NS = len(KINDS)
SOL_R = np.array([k[1] for k in KINDS], np.float32)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)


MOUND = np.array([[0.9, -0.5], [1.15, 0.2], [1.8, 0.62], [2.8, 0.45], [3.8, 0.12], [4.6, 0.0], [99, 0.0]])


def ground(x, y):
    """Ground height at arena coordinates (x, y in 0..50): flat soil with the nest mound in the middle."""
    r = np.hypot(x - ARENA / 2, y - ARENA / 2)
    return np.maximum(0.0, np.interp(r, MOUND[:, 0], MOUND[:, 1]))


class FlyWorld2:
    def __init__(self, P, seed=0):
        self.P, self.rng, self.respawn, self.external = P, np.random.default_rng(seed), True, False

    def reset(self, seeds=None):
        P = self.P
        rs = [np.random.default_rng(None if seeds is None else seeds[i]) for i in range(P)]
        self.pos = np.stack([[r.uniform(8, ARENA - 8), r.uniform(8, ARENA - 8), r.uniform(1.5, 5)] for r in rs]).astype(np.float32)
        self.yaw = np.array([r.uniform(0, 2 * np.pi) for r in rs], np.float32)
        self.berries = np.stack([[[r.uniform(4, ARENA - 4), r.uniform(4, ARENA - 4), 0.8] for _ in range(K)] for r in rs]).astype(np.float32)
        self.got = np.zeros((P, K), bool)
        self.sol = np.zeros((P, NS, 3), np.float32); self.vel = np.zeros((P, NS, 2), np.float32)
        for i, r in enumerate(rs):
            for j, k in enumerate(KINDS):
                self.sol[i, j] = [r.uniform(2, ARENA - 2), r.uniform(2, ARENA - 2), k[2]]
            self.vel[i, 3:5] = 0
        self.approach = np.array([r.random() < 0.5 for r in rs])
        for i in np.where(self.approach)[0]:                      # in half of the episodes the human walks straight at the fly
            ang = self.rng.uniform(0, 2 * np.pi)
            self.sol[i, 0, :2] = np.clip(self.pos[i, :2] + 22 * np.array([np.cos(ang), np.sin(ang)]), 2, ARENA - 2)
        self.esc = np.zeros(P, int); self.cool = np.zeros(P, int); self.esc_turn = np.zeros(P, np.float32)
        self.t = 0
        self.prev = np.minimum(self.nearest()[0], 60)
        self.collisions = np.zeros(P); self.escapes = np.zeros(P); self.false_esc = np.zeros(P)
        return self.observe()

    def set_berries(self, pts):
        """Use the colony's real berry positions for every fly (live mode)."""
        n = min(K, len(pts)); b = np.zeros((K, 3), np.float32); b[:, :2] = 1e4; b[:, 2] = 0.8
        for i in range(n): b[i, :2] = pts[i]
        self.berries[:] = b; self.got[:] = False; self.got[:, n:] = True
        self.prev = np.minimum(self.nearest()[0], 60)

    # ---- sensing ------------------------------------------------------------------------------------
    def nearest(self):
        d = np.linalg.norm(self.berries - self.pos[:, None], axis=2)
        d = np.where(self.got, 1e9, d)
        return d.min(1), d.argmin(1)

    def smell(self):
        s, c = np.sin(self.yaw), np.cos(self.yaw)
        out = np.zeros((self.P, 2), np.float32)
        for side, sg in enumerate((-1.0, 1.0)):                 # left, right antenna 0.35 to each side
            ax = self.pos[:, 0] + sg * 0.35 * (-s); ay = self.pos[:, 1] + sg * 0.35 * c
            d2 = (self.berries[..., 0] - ax[:, None]) ** 2 + (self.berries[..., 1] - ay[:, None]) ** 2
            out[:, side] = (np.exp(-d2 / (2 * 9.0 ** 2)) * ~self.got).sum(1)
        return out

    def mech(self):
        d = np.linalg.norm(self.sol - self.pos[:, None], axis=2)
        human = np.linalg.norm(self.vel[:, 0], axis=1) / (1 + (d[:, 0] / 6.0) ** 2)   # air buffeting: only when the human moves nearby
        flies = (0.1 / (1 + d[:, 1:3] ** 2)).sum(1)
        return (human + flies).astype(np.float32)

    def observe(self):
        P = self.P
        img = np.zeros((P, H, W, 3), np.float32)
        agl = self.pos[:, 2] - ground(self.pos[:, 0], self.pos[:, 1])
        img[:, H // 2:, :, 1] = (0.5 * np.clip(1.0 - agl / 8.0, 0.1, 1.0))[:, None, None]
        xs = (np.arange(W) + 0.5) / W * 2 - 1; ys = (np.arange(H) + 0.5) / H * 2 - 1
        c, s = np.cos(self.yaw), np.sin(self.yaw)

        def paint(target, radius, ok, ch):
            dv = target - self.pos
            f = dv[:, 0] * c + dv[:, 1] * s; r = -dv[:, 0] * s + dv[:, 1] * c
            dist = np.linalg.norm(dv, axis=1); ok = ok & (f > 0.05)
            X = np.where(ok, FC * r / np.maximum(f, 0.05), 9.0); row = -FC * dv[:, 2] / np.maximum(f, 0.05) * W / H
            sig = np.maximum(0.9, FC * radius / np.maximum(f, 0.3) * W / 2)
            gx = np.exp(-((xs[None] - X[:, None]) * W / 2) ** 2 / (2 * sig[:, None] ** 2))
            gy = np.exp(-((ys[None] - row[:, None]) * H / 2) ** 2 / (2 * (sig[:, None] * 0.8) ** 2))
            blob = gy[:, :, None] * gx[:, None, :] * ok[:, None, None]
            img[..., ch] = np.maximum(img[..., ch], blob)
            img[..., 2] = np.maximum(img[..., 2], blob * np.clip(1 - dist / 25.0, 0, 1)[:, None, None])
        for k in range(K):
            paint(self.berries[:, k], 0.6, ~self.got[:, k], 0)
        for j in range(NS):
            paint(self.sol[:, j], float(SOL_R[j]), np.ones(P, bool), 1)
        return img

    # ---- dynamics -----------------------------------------------------------------------------------
    def step(self, act, gf=None, away=None):
        """act (P,4) bool [forward, left, right, climb]; gf (P,) bool: giant fibre fired; away (P,) +1 turn right / -1 turn left."""
        act = act.copy()
        if gf is not None:
            fire = gf & (self.esc == 0) & (self.cool == 0)
            near_human = np.linalg.norm(self.sol[:, 0] - self.pos, axis=1) < 12
            self.escapes += fire & near_human; self.false_esc += fire & ~near_human
            self.esc[fire] = ESC_STEPS; self.cool[fire] = ESC_COOLDOWN
            self.esc_turn[fire] = away[fire]
        escaping = self.esc > 0
        act[escaping, 0] = True; act[escaping, 3] = True
        act[escaping, 1] = self.esc_turn[escaping] < 0; act[escaping, 2] = self.esc_turn[escaping] > 0
        self.esc = np.maximum(0, self.esc - 1); self.cool = np.maximum(0, self.cool - 1)
        fwd, tl, tr, up = [act[:, i] for i in range(4)]
        g0 = ground(self.pos[:, 0], self.pos[:, 1]); walking = (self.pos[:, 2] - g0) < 0.35
        speed = np.where(walking, SPEED_WALK, np.where(escaping, 2.2 * SPEED_AIR, SPEED_AIR))
        self.yaw += TURN * (tr.astype(np.float32) - tl.astype(np.float32))
        mv = speed * fwd
        nx, ny = self.pos[:, 0] + mv * np.cos(self.yaw), self.pos[:, 1] + mv * np.sin(self.yaw)
        hit_edge = (nx < 0) | (nx > ARENA) | (ny < 0) | (ny > ARENA)          # the edge of the world is physical: a fly that meets it is turned aside
        self.pos[:, 0], self.pos[:, 1] = np.clip(nx, 0, ARENA), np.clip(ny, 0, ARENA)
        self.yaw[hit_edge] += np.pi / 2 + self.rng.normal(0, 0.3, int(hit_edge.sum()))
        g1 = ground(self.pos[:, 0], self.pos[:, 1])
        self.pos[:, 2] = np.clip(self.pos[:, 2] + np.where(up, CLIMB * (1.6 if True else 1) * np.where(escaping, 1.6, 1.0), -SINK), g1, 8.0)
        if not self.external: self._move_solids()
        d, idx = self.nearest()
        r = 0.1 * (self.prev - np.minimum(d, 60)) - 0.002
        self.prev = np.minimum(d, 60)
        landed = d < LAND_R
        for i in np.where(landed)[0]:
            self.got[i, idx[i]] = True
        r = r + 2.0 * landed
        dd = np.linalg.norm(self.sol - self.pos[:, None], axis=2) - SOL_R[None]
        hit = (dd < 0.6).any(1); self.collisions += hit; r = r - 0.5 * hit
        if landed.any():
            if self.respawn:
                for i in np.where(self.got.all(1))[0]:
                    self.berries[i] = np.array([[self.rng.uniform(4, ARENA - 4), self.rng.uniform(4, ARENA - 4), 0.8] for _ in range(K)], np.float32); self.got[i] = False
            d2 = self.nearest()[0]; self.prev = np.where(d2 > 1e8, 0, np.minimum(d2, 60))
        self.t += 1
        return self.observe(), r.astype(np.float32), landed

    def _move_solids(self):
        P = self.P
        for i in range(P):
            if self.approach[i]:
                v = self.pos[i, :2] - self.sol[i, 0, :2]; n = np.linalg.norm(v)
                self.vel[i, 0] = v / max(n, 1e-6) * (0.45 if n > 3.5 else 0.0)
            else:
                self.vel[i, 0] = self.vel[i, 0] * 0.97 + self.rng.normal(0, 0.05, 2)
        for j in (1, 2, 3, 4):
            self.vel[:, j] = self.vel[:, j] * 0.95 + self.rng.normal(0, 0.04 if j > 2 else 0.1, (P, 2))
        self.sol[:, :5, :2] = np.clip(self.sol[:, :5, :2] + self.vel[:, :5], 1, ARENA - 1)
