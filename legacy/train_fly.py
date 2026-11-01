"""Train the readout of the real fruit-fly connectome to fly to berries (evolution strategies).

Run with the fly-brain environment:  C:\\dev\\fly-brain\\.venv\\Scripts\\python.exe train_fly.py [generations]
Only the 3 photoreceptor gains and a linear readout from 1,305 descending neurons are learned;
all 3.7 million connectome synapses stay fixed.
"""
import os, sys, time
sys.path.insert(0, r"C:\dev\fly-brain")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from brain import BatchedBrain
from flyworld import FlyWorld

P = int(os.environ.get("POP", 32))
EP = int(os.environ.get("EP_LEN", 300))
SIGMA, LR = float(os.environ.get("SIGMA", 0.3)), float(os.environ.get("LR", 0.2))
GAIN = float(os.environ.get("GAIN", 0.02))
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly_ckpt")
os.makedirs(CK, exist_ok=True)


def evaluate(brain, thetas, seed, steps=2):
    w = FlyWorld(P)
    obs = w.reset(seeds=[seed + i // 2 for i in range(P)])      # antithetic pairs share a start
    brain.reset_batch(P)
    fit = np.zeros(P); landings = np.zeros(P)
    for t in range(EP):
        logits = brain.act_batch(obs, thetas, steps=steps)
        obs, r, landed = w.step((logits > 0).cpu().numpy())
        fit += r; landings += landed
    return fit, landings


def main(gens):
    brain = BatchedBrain(gain=GAIN)
    D = brain.theta_dim
    theta = brain.theta.clone()
    if os.path.exists(f"{CK}/latest.pt"):
        theta = torch.load(f"{CK}/latest.pt").to(brain.dev); print("resumed", flush=True)
    best = -1e9
    for g in range(gens):
        t0 = time.time()
        eps = torch.randn(P // 2, D, device=brain.dev); eps = torch.cat([eps, -eps])
        fit, land = evaluate(brain, theta[None] + SIGMA * eps, seed=g * 100)
        ranks = np.argsort(np.argsort(fit)) / (P - 1) - 0.5
        wts = torch.tensor(ranks, device=brain.dev, dtype=torch.float32)
        theta = theta + LR / (P * SIGMA) * (wts[:, None] * eps).sum(0)
        torch.save(theta.cpu(), f"{CK}/latest.pt")
        if fit.mean() > best:
            best = fit.mean(); torch.save(theta.cpu(), f"{CK}/best.pt")
        print(f"gen {g} fit mean {fit.mean():.2f} max {fit.max():.2f} | landings per fly mean {land.mean():.2f} max {land.max():.0f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
