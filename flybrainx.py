"""The real FlyWire network with extra senses: smell (olfactory neurons), air movement (mechanosensory neurons),
and read-outs of the giant-fibre escape neurons and the left/right looming detectors."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config; config.add_flybrain_to_path()
os.environ.setdefault("FLYWIRE_DIR", config.FLYWIRE_DIR)    # brain.py reads the FlyWire location from here
import numpy as np, torch
from brain import BatchedBrain
import flycells
import pandas as pd


class FlyBrainX(BatchedBrain):
    def __init__(self, gain=0.3, g_smell=1.0, g_mech=1.5, n_vpn=0, **kw):
        super().__init__(gain=gain, **kw)
        c = flycells.load(); dev = self.dev
        T = lambda a: torch.tensor(a, device=dev)
        self.gf, self.lplc2 = T(c["gf"]), T(c["lplc2"])
        orn, side = c["orn"], c["orn_side"]
        self.orn_l, self.orn_r = T(orn[(side == 0) | (side == 2)]), T(orn[(side == 1) | (side == 2)])
        self.mech = T(c["mech"]); self.g_smell, self.g_mech = g_smell, g_mech
        # which looming detectors look left / right
        D = config.FLYWIRE_DIR
        vt = pd.read_csv(f"{D}/visual_neuron_types.csv.gz"); cls = pd.read_csv(f"{D}/classification.csv.gz")
        idx = {r: i for i, r in enumerate(cls.root_id.values)}
        sub = vt[vt.type == "LPLC2"]
        self.lp_l = T(np.array([idx[r] for r, s in zip(sub.root_id, sub.side) if s == "left" and r in idx]))
        self.lp_r = T(np.array([idx[r] for r, s in zip(sub.root_id, sub.side) if s == "right" and r in idx]))
        if n_vpn:   # also read out the eye's own output lines (visual projection neurons), not only descending neurons
            vp = np.array([idx[r] for r in cls[cls.super_class == "visual_projection"].root_id])
            vp = np.random.default_rng(0).choice(vp, n_vpn, replace=False)
            self.out_idx = torch.cat([self.out_idx, T(vp)]); self.n_out = len(self.out_idx)
            self.theta_dim = 3 + self.n_out * self.n_act + self.n_act
            self.theta = torch.zeros(self.theta_dim, device=dev); self.theta[:3] = 1.0

    @torch.no_grad()
    def act_x(self, obs, thetas, smell, mech, steps=2):
        P = obs.shape[0]
        o = torch.as_tensor(obs, device=self.dev, dtype=torch.float32).reshape(P, -1, 3)
        pixval = (o * thetas[:, None, :3]).sum(2)
        inp = torch.zeros(self.n, P, device=self.dev)
        inp[self.pr_idx] = pixval[:, self.pix].T
        sm = torch.as_tensor(smell, device=self.dev, dtype=torch.float32); me = torch.as_tensor(mech, device=self.dev, dtype=torch.float32)
        inp[self.orn_l] += self.g_smell * sm[:, 0][None]
        inp[self.orn_r] += self.g_smell * sm[:, 1][None]
        inp[self.mech] += self.g_mech * me[None]
        for _ in range(steps):
            rec = self.W @ self.X
            self.X = (1 - self.tau) * self.X + self.tau * torch.clamp(rec + self.gain * inp, 0.0, 1.0)
        out = self.X[self.out_idx].T
        out = out / (out.mean(1, keepdim=True) + 1e-3)
        R = thetas[:, 3:3 + self.n_out * self.n_act].view(P, self.n_out, self.n_act)
        logits = torch.einsum("po,poa->pa", out, R) / np.sqrt(self.n_out) + thetas[:, -self.n_act:]
        gf = self.X[self.gf].mean(0)                                     # (P,) giant-fibre activity
        left = self.X[self.lp_l].mean(0); right = self.X[self.lp_r].mean(0)
        return logits, gf, left, right
