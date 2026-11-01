"""The male fly's whole central nervous system (MaleCNS v0.9, 165,114 neurons) as a fixed recurrent rate network on the GPU.

Same interface as flybrainx.FlyBrainX so the world, training and server code work for both brains.
Senses: photoreceptors (retina map from the lamina columns they feed), olfactory neurons (left/right), and
mechanosensory neurons (chordotonal, bristle, hair plate). Escape: the two DNp01 giant fibres and the LPLC2 looming detectors.
Readout: descending neurons + motor neurons + looming detectors, with a trained linear layer. Weights = synapse counts
with the sign of the predicted transmitter; every neuron's total input is normalised (as brain.py does for FlyWire).
"""
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


class MaleBrainX:
    def __init__(self, gain=0.3, g_smell=0.1, g_mech=4.0, device="cuda", img_w=24, img_h=16, n_act=4):
        z = np.load(os.path.join(HERE, "malecns_net.npz"), allow_pickle=True)
        r = np.load(os.path.join(HERE, "malecns_retina.npz"))
        self.dev = torch.device(device); self.n = int(z["n"]); n = self.n
        pre, post, w = z["pre"].astype(np.int64), z["post"].astype(np.int64), z["w"].astype(np.float32)
        tot = np.bincount(post, weights=np.abs(w), minlength=n).astype(np.float32)
        w = w / np.maximum(tot[post], 1.0)
        self.W = torch.sparse_coo_tensor(torch.tensor(np.stack([post, pre])), torch.tensor(w), (n, n)).coalesce().to_sparse_csr().to(self.dev)
        T = lambda a: torch.tensor(np.asarray(a), device=self.dev)
        # retina: each eye's photoreceptors laid out by their column coordinates, left eye = left half of the picture
        col = np.zeros(len(r["pr"]), int); row = np.zeros(len(r["pr"]), int)
        for s in (0, 1):
            m = r["side"] == s
            if not m.any(): continue
            u = (r["x"][m] - np.percentile(r["x"][m], 1)) / (np.ptp(np.percentile(r["x"][m], [1, 99])) + 1e-6)
            v = (r["y"][m] - np.percentile(r["y"][m], 1)) / (np.ptp(np.percentile(r["y"][m], [1, 99])) + 1e-6)
            col[m] = np.clip((u * (img_w // 2)).astype(int), 0, img_w // 2 - 1) + s * (img_w // 2)
            row[m] = np.clip((v * img_h).astype(int), 0, img_h - 1)
        self.pr_idx, self.pix = T(r["pr"]), T(row * img_w + col)
        self.gf, self.lplc2 = T(z["gf"]), T(z["lplc2"])
        side = z["side"]
        self.lp_l, self.lp_r = T(z["lplc2"][side[z["lplc2"]] == 0]), T(z["lplc2"][side[z["lplc2"]] == 1])
        orn = z["orn"]; self.orn_l, self.orn_r = T(orn[(side[orn] == 0) | (side[orn] == 2)]), T(orn[(side[orn] == 1) | (side[orn] == 2)])
        self.mech = T(z["mech"])
        self.out_idx = T(np.concatenate([z["desc"], z["motor"], z["lplc2"], z["lc4"]]))
        self.n_out, self.n_act, self.gain, self.tau, self.g_smell, self.g_mech = len(self.out_idx), n_act, gain, 0.5, g_smell, g_mech
        self.theta_dim = 3 + self.n_out * n_act + n_act
        self.theta = torch.zeros(self.theta_dim, device=self.dev); self.theta[:3] = 1.0
        self.n_motor = len(z["motor"]); self.motor_kind = z["motor_kind"]

    def reset_batch(self, P):
        self.X = torch.zeros(self.n, P, device=self.dev)

    @torch.no_grad()
    def act_x(self, obs, thetas, smell, mech, steps=2):
        P = obs.shape[0]
        o = torch.as_tensor(obs, device=self.dev, dtype=torch.float32).reshape(P, -1, 3)
        pixval = (o * thetas[:, None, :3]).sum(2)
        inp = torch.zeros(self.n, P, device=self.dev)
        inp[self.pr_idx] = pixval[:, self.pix].T
        sm = torch.as_tensor(smell, device=self.dev, dtype=torch.float32); me = torch.as_tensor(mech, device=self.dev, dtype=torch.float32)
        inp[self.orn_l] += self.g_smell * sm[:, 0][None]; inp[self.orn_r] += self.g_smell * sm[:, 1][None]
        inp[self.mech] += self.g_mech * me[None]
        for _ in range(steps):
            self.X = (1 - self.tau) * self.X + self.tau * torch.clamp(self.W @ self.X + self.gain * inp, 0.0, 1.0)
        out = self.X[self.out_idx].T
        out = out / (out.mean(1, keepdim=True) + 1e-3)
        R = thetas[:, 3:3 + self.n_out * self.n_act].view(P, self.n_out, self.n_act)
        logits = torch.einsum("po,poa->pa", out, R) / np.sqrt(self.n_out) + thetas[:, -self.n_act:]
        return logits, self.X[self.gf].mean(0), self.X[self.lp_l].mean(0), self.X[self.lp_r].mean(0)
