"""Flight world for the real FlyWire connectome brain (see C:\\dev\\fly-brain\\brain.py).

P flies fly in parallel in the 50x50 colony arena. They SEE berries through the connectome's own
photoreceptor layout (24x16 image: left eye = left half, right eye = right half) and the connectome
brain's descending neurons, read out by a small trained layer, choose 4 actions:
    [forward, turn_left, turn_right, climb]   (when not climbing the fly sinks slowly)
Reward: get closer to the nearest berry, and land on it.
"""
import numpy as np

ARENA, K, W, H = 50.0, 3, 24, 16
FOV = np.radians(100.0)
FC = 1.0 / np.tan(FOV / 2)
SPEED, TURN, CLIMB, SINK = 0.25, 0.12, 0.15, 0.05
LAND_R = 0.9


class FlyWorld:
    def __init__(self, P, seed=0):
        self.P, self.rng, self.respawn = P, np.random.default_rng(seed), True

    def reset(self, seeds=None):
        P = self.P
        rs = [np.random.default_rng(None if seeds is None else seeds[i]) for i in range(P)]
        self.pos = np.stack([[r.uniform(5, 45), r.uniform(5, 45), r.uniform(1.5, 5)] for r in rs]).astype(np.float32)
        self.yaw = np.array([r.uniform(0, 2 * np.pi) for r in rs], np.float32)
        self.berries = np.stack([[[r.uniform(4, 46), r.uniform(4, 46), 0.8] for _ in range(K)] for r in rs]).astype(np.float32)
        self.got = np.zeros((P, K), bool)
        self.t = 0
        self.prev = self.nearest()[0]
        return self.observe()

    def set_berries(self, pts):
        """Use real berry positions (list of (x, y)) for all flies (live mode)."""
        n = min(K, len(pts))
        b = np.zeros((K, 3), np.float32); b[:, :2] = 1e4; b[:, 2] = 0.8
        for i in range(n): b[i, :2] = pts[i]
        self.berries[:] = b; self.got[:] = False
        self.got[:, n:] = True

    def nearest(self):
        d = np.linalg.norm(self.berries - self.pos[:, None], axis=2)
        d = np.where(self.got, 1e9, d)
        return d.min(1), d.argmin(1)

    def step(self, act):
        fwd, tl, tr, up = [act[:, i].astype(bool) for i in range(4)]
        self.yaw += TURN * (tr.astype(np.float32) - tl.astype(np.float32))
        mv = SPEED * fwd
        self.pos[:, 0] = np.clip(self.pos[:, 0] + mv * np.cos(self.yaw), 0, ARENA)
        self.pos[:, 1] = np.clip(self.pos[:, 1] + mv * np.sin(self.yaw), 0, ARENA)
        self.pos[:, 2] = np.clip(self.pos[:, 2] + np.where(up, CLIMB, -SINK), 0.4, 8.0)
        d, idx = self.nearest()
        r = 0.1 * (self.prev - np.minimum(d, 60)) - 0.002
        self.prev = np.minimum(d, 60)
        landed = d < LAND_R
        for i in np.where(landed)[0]:
            self.got[i, idx[i]] = True
        r = r + 2.0 * landed
        if landed.any():
            if self.respawn:   # training: a fresh berry appears once all are eaten (live mode keeps the colony's real berries)
                for i in np.where(self.got.all(1))[0]:
                    self.berries[i] = np.array([[self.rng.uniform(4, 46), self.rng.uniform(4, 46), 0.8] for _ in range(K)], np.float32)
                    self.got[i] = False
            d2 = self.nearest()[0]
            self.prev = np.where(d2 > 1e8, 0, np.minimum(d2, 60))
        self.t += 1
        return self.observe(), r.astype(np.float32), landed

    def observe(self):
        P = self.P
        img = np.zeros((P, H, W, 3), np.float32)
        ys = (np.arange(H) + 0.5) / H * 2 - 1
        xs = (np.arange(W) + 0.5) / W * 2 - 1
        # ground: lower half of the view, brighter when low (a cheap altitude cue)
        img[:, H // 2:, :, 1] = np.clip(1.0 - self.pos[:, 2] / 8.0, 0.1, 1.0)[:, None, None]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        for k in range(K):
            dv = self.berries[:, k] - self.pos
            f = dv[:, 0] * c + dv[:, 1] * s
            r = -dv[:, 0] * s + dv[:, 1] * c
            dist = np.linalg.norm(dv, axis=1)
            ok = (f > 0.05) & ~self.got[:, k]
            X = np.where(ok, FC * r / np.maximum(f, 0.05), 9.0)
            Y = FC * dv[:, 2] / np.maximum(f, 0.05)
            row = -Y * W / H
            sig = np.maximum(0.9, FC * 0.6 / np.maximum(f, 0.3) * W / 2)
            gx = np.exp(-((xs[None] - X[:, None]) * W / 2) ** 2 / (2 * sig[:, None] ** 2))        # (P, W)
            gy = np.exp(-((ys[None] - row[:, None]) * H / 2) ** 2 / (2 * (sig[:, None] * 0.8) ** 2))  # (P, H)
            blob = gy[:, :, None] * gx[:, None, :] * ok[:, None, None]
            img[..., 0] = np.maximum(img[..., 0], blob)
            img[..., 2] = np.maximum(img[..., 2], blob * np.clip(1 - dist / 25.0, 0, 1)[:, None, None])
        return img
