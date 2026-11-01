import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from flybrainx import FlyBrainX
import train_fly2 as T
from flyworld2 import FlyWorld2
b = FlyBrainX(gain=0.3); th = torch.load(sys.argv[1] if len(sys.argv) > 1 else "fly_ckpt/best_v2.pt").to(b.dev)
tot = {True: [], False: []}
for rep in range(3):
    w = FlyWorld2(32); obs = w.reset(seeds=[5000 + rep * 40 + i for i in range(32)]); b.reset_batch(32); land = np.zeros(32); T_ = th[None].repeat(32, 1)
    for t in range(300):
        lg, gf, l, r = b.act_x(obs, T_, w.smell(), w.mech())
        away = np.where((l > r).cpu().numpy(), 1.0, -1.0)
        obs, rew, landed = w.step((lg > 0).cpu().numpy(), (gf > T.GF_THR).cpu().numpy(), away); land += landed
    for i in range(32): tot[bool(w.approach[i])].append((land[i], w.collisions[i], w.escapes[i], w.false_esc[i]))
for k, name in ((False, "human wandering (calm)"), (True, "human charging the fly")):
    a = np.array(tot[k]); print(f"{name:26s} n={len(a)} landings/fly {a[:,0].mean():.2f} | collisions {a[:,1].mean():.1f} | escapes {a[:,2].mean():.1f} | false alarms {a[:,3].mean():.2f}")
