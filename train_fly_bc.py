"""Teach the real FlyWire connectome to fly to berries by imitation (DAgger).

A scripted pilot knows where the berries are. The connectome only sees the image. We record the brain's
descending-neuron activity and fit a linear readout to predict the pilot's action; then let the brain fly
(sometimes with the pilot taking over) and fit again on the states it actually visits.
Run: C:\\dev\\fly-brain\\.venv\\Scripts\\python.exe train_fly_bc.py
"""
import os, sys, time
sys.path.insert(0, r"C:\dev\fly-brain")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from brain import BatchedBrain
from flyworld import FlyWorld

P, EP, ROUNDS = 32, 300, int(os.environ.get("ROUNDS", 8))
GAIN = float(os.environ.get("GAIN", 0.02))
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly_ckpt"); os.makedirs(CK, exist_ok=True)


def pilot(w):
    d, idx = w.nearest(); b = w.berries[np.arange(w.P), idx]
    want = np.arctan2(b[:, 1] - w.pos[:, 1], b[:, 0] - w.pos[:, 0])
    err = (want - w.yaw + np.pi) % (2 * np.pi) - np.pi
    return np.stack([np.abs(err) < 0.5, err < -0.1, err > 0.1, w.pos[:, 2] < b[:, 2]], 1)


def features(brain):
    out = brain.X[brain.out_idx].T
    return out / (out.mean(1, keepdim=True) + 1e-3)


def rollout(brain, theta, beta, seed):
    """beta = probability the pilot controls a given fly's step. Returns features, pilot labels, mean landings of policy-only flies."""
    w = FlyWorld(P); obs = w.reset(seeds=[seed + i for i in range(P)]); brain.reset_batch(P)
    th = theta[None].repeat(P, 1); F, Y = [], []; land = np.zeros(P)
    for t in range(EP):
        logits = brain.act_batch(obs, th, steps=2)
        lab = pilot(w); F.append(features(brain).clone()); Y.append(torch.tensor(lab, dtype=torch.float32, device=brain.dev))
        pol = (logits > 0).cpu().numpy()
        use_pilot = np.random.rand(P) < beta
        obs, r, landed = w.step(np.where(use_pilot[:, None], lab, pol)); land += landed
    return torch.cat(F), torch.cat(Y), land


def fit(F, Y, theta, brain, steps=400, lam=1e-3):
    n_out = brain.n_out
    R = theta[3:3 + n_out * 4].view(n_out, 4).clone().requires_grad_(True); b = theta[-4:].clone().requires_grad_(True)
    opt = torch.optim.Adam([R, b], lr=0.05); pos = Y.mean(0).clamp(0.02, 0.98); wgt = ((1 - pos) / pos).sqrt()
    for _ in range(steps):
        logits = F @ R / np.sqrt(n_out) + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y, pos_weight=wgt) + lam * (R ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    acc = ((logits > 0).float() == Y).float().mean(0).detach().cpu().numpy()
    th = theta.clone(); th[3:3 + n_out * 4] = R.detach().flatten(); th[-4:] = b.detach()
    return th, float(loss), acc


def main():
    brain = BatchedBrain(gain=GAIN); theta = brain.theta.clone(); FS, YS, best = [], [], -1
    for r in range(ROUNDS):
        t0 = time.time(); beta = [1.0, 0.6, 0.3, 0.15, 0.05, 0.0, 0.0, 0.0][min(r, 7)]
        F, Y, _ = rollout(brain, theta, beta, seed=1000 + r * 50); FS.append(F); YS.append(Y)
        theta, loss, acc = fit(torch.cat(FS), torch.cat(YS), theta, brain)
        _, _, land = rollout(brain, theta, 0.0, seed=9000 + r * 50)      # the brain flies by itself
        print(f"round {r} beta {beta:.2f} samples {sum(len(f) for f in FS)} loss {loss:.3f} acc {np.round(acc, 2)} | SOLO landings per fly {land.mean():.2f} max {land.max():.0f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(theta.cpu(), f"{CK}/latest_bc.pt")
        if land.mean() > best: best = land.mean(); torch.save(theta.cpu(), f"{CK}/best_bc.pt")


if __name__ == "__main__":
    main()
