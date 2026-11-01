# brain.py from the author's fly-brain project, included here with permission so the flies can run on a fresh checkout.
# The only change: the FlyWire data folder comes from FLYWIRE_DIR (see ../../config.py) instead of a fixed path.
# On first use it builds cache/conn.npz (about 70 MB, ignored by git) from the FlyWire files.
"""FlyWire (FAFB v783) connectome as a fixed recurrent rate network on GPU.

Inputs : photoreceptors (sensory/visual), retinotopically mapped to image pixels by position.
Hidden : all other neurons, fixed signed connectome weights.
Outputs: descending neurons -> learned linear readout -> 4 button logits.
Learned: per-channel input gains + readout (small), trained with evolution strategies.
"""
import os
import numpy as np
import pandas as pd
import torch

D = os.environ.get("FLYWIRE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "flywire")   # the FlyWire FAFB v783 tables
CACHE = os.path.join(os.path.dirname(__file__), "cache")
SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "DA": 1.0, "SER": 1.0, "OCT": 1.0}


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    cls = pd.read_csv(f"{D}/classification.csv.gz")
    ids = cls.root_id.values
    idx = {r: i for i, r in enumerate(ids)}
    con = pd.read_csv(f"{D}/connections_princeton.csv.gz")
    pre = con.pre_root_id.map(idx).values
    post = con.post_root_id.map(idx).values
    ok = ~(np.isnan(pre.astype(float)) | np.isnan(post.astype(float)))
    pre, post = pre[ok].astype(np.int64), post[ok].astype(np.int64)
    w = con.syn_count.values[ok].astype(np.float32) * con.nt_type.map(SIGN).fillna(1).values[ok].astype(np.float32)
    # aggregate duplicate (pre,post) pairs across neuropils
    key = pre * len(ids) + post
    uk, inv = np.unique(key, return_inverse=True)
    wsum = np.zeros(len(uk), np.float32)
    np.add.at(wsum, inv, w)
    pre, post = uk // len(ids), uk % len(ids)

    # photoreceptor retinotopy: PCA of positions per eye -> grid coordinates in [0,1]
    coords = pd.read_csv(f"{D}/coordinates.csv.gz")
    pos = np.array([[float(v) for v in s.strip("[]").split()] for s in coords.position])
    pos = dict(zip(coords.root_id.values, pos))
    pr = cls[(cls.super_class == "sensory") & (cls["class"] == "visual") & cls.root_id.isin(pos)]
    pr_idx, uv, side = [], [], []
    for s in ("left", "right"):
        sub = pr[pr.side == s]
        P = np.array([pos[r] for r in sub.root_id])
        P = P - P.mean(0)
        _, _, vt = np.linalg.svd(P, full_matrices=False)
        q = P @ vt[:2].T
        q = (q - q.min(0)) / (q.max(0) - q.min(0) + 1e-9)
        pr_idx += [idx[r] for r in sub.root_id]
        uv.append(q)
        side += [0 if s == "left" else 1] * len(sub)
    out_idx = [idx[r] for r in cls[cls.super_class == "descending"].root_id]
    np.savez(f"{CACHE}/conn.npz", n=len(ids), pre=pre, post=post, w=wsum,
             pr_idx=np.array(pr_idx), pr_uv=np.concatenate(uv), pr_side=np.array(side),
             out_idx=np.array(out_idx))


class FlyBrain:
    def __init__(self, img_w=24, img_h=16, n_actions=4, device="cuda", gain=0.02, tau=0.5, seed=0):
        if not os.path.exists(f"{CACHE}/conn.npz"):
            build_cache()
        z = np.load(f"{CACHE}/conn.npz")
        self.dev = torch.device(device)
        self.n = int(z["n"])
        w = z["w"].astype(np.float32)
        # normalise each neuron's total input so the fixed recurrent dynamics stay stable
        post = z["post"]
        tot = np.bincount(post, weights=np.abs(w), minlength=self.n).astype(np.float32)
        w = w / np.maximum(tot[post], 1.0) * 1.0  # spectral-radius-ish control
        self.W = torch.sparse_coo_tensor(
            torch.tensor(np.stack([post, z["pre"]])), torch.tensor(w), (self.n, self.n)
        ).coalesce().to_sparse_csr().to(self.dev)
        self.pr_idx = torch.tensor(z["pr_idx"], device=self.dev)
        self.out_idx = torch.tensor(z["out_idx"], device=self.dev)
        self.n_out, self.n_act = len(z["out_idx"]), n_actions
        # photoreceptor -> pixel (column offset by eye)
        uv, side = z["pr_uv"], z["pr_side"]
        col = np.clip((uv[:, 0] * (img_w // 2)).astype(int), 0, img_w // 2 - 1) + side * (img_w // 2)
        row = np.clip((uv[:, 1] * img_h).astype(int), 0, img_h - 1)
        self.pix = torch.tensor(row * img_w + col, device=self.dev)
        self.n_pix, self.n_ch = img_w * img_h, 3
        self.gain, self.tau = gain, tau
        g = torch.Generator(device="cpu").manual_seed(seed)
        # learned params: per-channel PR gains (3) + per-PR channel mixing bias, readout (n_out x n_act)
        self.theta_dim = 3 + self.n_out * n_actions + n_actions
        self.theta = torch.zeros(self.theta_dim, device=self.dev)
        self.theta[3:3 + self.n_out * n_actions] = 0.0
        self.theta[:3] = 1.0
        self.reset()

    def reset(self):
        self.x = torch.zeros(self.n, device=self.dev)

    def set_theta(self, th):
        self.theta = th.to(self.dev)

    @torch.no_grad()
    def act(self, obs, theta=None, steps=4, sample=False):
        """obs (H,W,3) numpy -> action logits (n_act,) after `steps` of connectome dynamics."""
        th = self.theta if theta is None else theta
        o = torch.as_tensor(obs, device=self.dev, dtype=torch.float32).reshape(-1, 3)
        pixval = (o * th[:3]).sum(1)  # (n_pix)
        inp = torch.zeros(self.n, device=self.dev)
        inp[self.pr_idx] = pixval[self.pix]
        for _ in range(steps):
            rec = self.W @ self.x
            self.x = (1 - self.tau) * self.x + self.tau * torch.clamp(rec + self.gain * inp, 0.0, 1.0)
        out = self.x[self.out_idx]
        out = out / (out.mean() + 1e-3)
        R = th[3:3 + self.n_out * self.n_act].view(self.n_out, self.n_act)
        logits = out @ R / np.sqrt(self.n_out) + th[-self.n_act:]
        return logits

    def activity(self):
        return self.x


class BatchedBrain(FlyBrain):
    """P independent copies of the same connectome with different learned params, run in lock-step."""

    def reset_batch(self, P):
        self.X = torch.zeros(self.n, P, device=self.dev)

    @torch.no_grad()
    def act_batch(self, obs, thetas, steps=2):
        """obs (P,H,W,3) numpy, thetas (P,D) tensor -> logits (P,n_act)."""
        P = obs.shape[0]
        o = torch.as_tensor(obs, device=self.dev, dtype=torch.float32).reshape(P, -1, 3)
        pixval = (o * thetas[:, None, :3]).sum(2)                  # (P, n_pix)
        inp = torch.zeros(self.n, P, device=self.dev)
        inp[self.pr_idx] = pixval[:, self.pix].T
        for _ in range(steps):
            rec = self.W @ self.X
            self.X = (1 - self.tau) * self.X + self.tau * torch.clamp(rec + self.gain * inp, 0.0, 1.0)
        out = self.X[self.out_idx].T                                # (P, n_out)
        out = out / (out.mean(1, keepdim=True) + 1e-3)
        R = thetas[:, 3:3 + self.n_out * self.n_act].view(P, self.n_out, self.n_act)
        return torch.einsum("po,poa->pa", out, R) / np.sqrt(self.n_out) + thetas[:, -self.n_act:]
