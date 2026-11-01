"""Imitation (DAgger) training of the FlyWire readout in the richer world: berries + obstacles + human + smell + escape reflex.
Run: C:\dev\fly-brain\.venv\Scripts\python.exe train_fly2.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from flybrainx import FlyBrainX
from maleflybrain import MaleBrainX
from flyworld2 import FlyWorld2, SOL_R

P, EP, ROUNDS, GF_THR = 32, 300, int(os.environ.get("ROUNDS", 14)), float(os.environ.get("GF_THR", 0.0035))
TAG = os.environ.get("TAG", "")
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly_ckpt"); os.makedirs(CK, exist_ok=True)


def pilot(w):
    d, idx = w.nearest(); b = w.berries[np.arange(w.P), idx]
    goal = b[:, :2] - w.pos[:, :2]; goal = goal / np.maximum(np.linalg.norm(goal, axis=1, keepdims=True), 1e-6)
    push = np.zeros_like(goal)
    dv = w.pos[:, None, :2] - w.sol[..., :2]; dist = np.linalg.norm(dv, axis=2)
    gap = dist - SOL_R[None]
    for j in range(dv.shape[1]):
        wgt = np.clip((6.0 - gap[:, j]) / 6.0, 0, 1) ** 2 * (gap[:, j] < 6.0)
        push += dv[:, j] / np.maximum(dist[:, j:j + 1], 1e-6) * wgt[:, None]
    want = np.arctan2(*(goal + 2.5 * push).T[::-1])
    err = (want - w.yaw + np.pi) % (2 * np.pi) - np.pi
    return np.stack([np.abs(err) < 0.6, err < -0.1, err > 0.1, w.pos[:, 2] < b[:, 2] + 0.2], 1)


def feats(brain):
    out = brain.X[brain.out_idx].T
    return out / (out.mean(1, keepdim=True) + 1e-3)


def rollout(brain, theta, beta, seed):
    w = FlyWorld2(P); obs = w.reset(seeds=[seed + i for i in range(P)]); brain.reset_batch(P)
    th = theta[None].repeat(P, 1); F, Y = [], []; land = np.zeros(P)
    for t in range(EP):
        logits, gf, l, r = brain.act_x(obs, th, w.smell(), w.mech())
        lab = pilot(w); F.append(feats(brain).clone()); Y.append(torch.tensor(lab, dtype=torch.float32, device=brain.dev))
        pol = (logits > 0).cpu().numpy(); use = np.random.rand(P) < beta
        away = np.where((l > r).cpu().numpy(), 1.0, -1.0)
        obs, rew, landed = w.step(np.where(use[:, None], lab, pol), (gf > GF_THR).cpu().numpy(), away); land += landed
    return torch.cat(F), torch.cat(Y), land, w


def fit(F, Y, theta, brain, steps=400, lam=1e-3):
    n = brain.n_out
    R = theta[3:3 + n * 4].view(n, 4).clone().requires_grad_(True); b = theta[-4:].clone().requires_grad_(True)
    opt = torch.optim.Adam([R, b], lr=0.05); pos = Y.mean(0).clamp(0.02, 0.98); wgt = ((1 - pos) / pos).sqrt()
    for _ in range(steps):
        logits = F @ R / np.sqrt(n) + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y, pos_weight=wgt) + lam * (R ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    th = theta.clone(); th[3:3 + n * 4] = R.detach().flatten(); th[-4:] = b.detach(); return th, float(loss)


def main():
    if os.environ.get("BRAIN") == "male":
        brain = MaleBrainX(gain=0.3, g_smell=float(os.environ.get("G_SMELL", 0.1)), g_mech=float(os.environ.get("G_MECH", 4.0)))
    else:
        brain = FlyBrainX(gain=0.3, g_smell=float(os.environ.get("G_SMELL", 1.0)), g_mech=float(os.environ.get("G_MECH", 1.5)), n_vpn=int(os.environ.get("N_VPN", 0)))
    theta = brain.theta.clone(); FS, YS, best = [], [], -1e9
    for r in range(ROUNDS):
        t0 = time.time(); beta = [1.0, 0.6, 0.3, 0.15, 0.05][min(r, 4)] if r < 5 else 0.0
        F, Y, _, _ = rollout(brain, theta, beta, 1000 + r * 50); FS.append(F); YS.append(Y)
        theta, loss = fit(torch.cat(FS), torch.cat(YS), theta, brain)
        _, _, land, w = rollout(brain, theta, 0.0, 9000 + r * 50)
        appr = w.approach
        print(f"round {r} beta {beta:.2f} loss {loss:.3f} | SOLO landings/fly calm {land[~appr].mean():.2f} charged {land[appr].mean():.2f} | collisions/fly {w.collisions.mean():.1f} "
              f"| escapes: human-near {w.escapes.mean():.2f} (approach eps {w.escapes[appr].mean():.2f}), false {w.false_esc.mean():.2f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(theta.cpu(), f"{CK}/latest_v2{TAG}.pt")
        score = land[~appr].mean() - 0.05 * w.collisions.mean()
        if score > best: best = score; torch.save(theta.cpu(), f"{CK}/best_v2{TAG}.pt")


if __name__ == "__main__":
    main()
