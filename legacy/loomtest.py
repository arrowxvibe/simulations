import sys; sys.path.insert(0, r"C:\dev\fly-brain")
import numpy as np, torch
from brain import BatchedBrain
import flycells
c = flycells.load()
def blob(cx, cy, s):
    xs=(np.arange(24)+.5)/24*2-1; ys=(np.arange(16)+.5)/16*2-1
    gx=np.exp(-((xs-cx)*12)**2/(2*s*s)); gy=np.exp(-((ys-cy)*8)**2/(2*(0.8*s)**2))
    return (gy[:,None]*gx[None,:])
def run(gain, kind, T=40):
    b = BatchedBrain(gain=gain); b.reset_batch(1); th = b.theta[None]
    rec = {k: [] for k in ("gf", "lplc2", "lc4")}
    for t in range(T):
        if kind == "loom": s = 0.8 + 5.5 * (t / T) ** 2; im = blob(0.0, 0.0, s)
        elif kind == "slide": im = blob(-0.8 + 1.6 * t / T, 0.0, 2.0)
        elif kind == "static": im = blob(0.0, 0.0, 2.0)
        else: im = np.zeros((16, 24))
        obs = np.repeat(im[None, :, :, None], 3, axis=3).astype(np.float32)[None] if False else np.repeat(im[None, :, :, None], 3, 3).astype(np.float32)
        b.act_batch(obs, th, steps=2)
        x = b.X[:, 0]
        for k in rec: rec[k].append(float(x[torch.tensor(c[k], device=b.dev)].mean()))
    return {k: (np.mean(v[:5]), np.mean(v[-8:])) for k, v in rec.items()}
for gain in (0.3, 2.0):
    print("gain", gain)
    for kind in ("none", "static", "slide", "loom"):
        r = run(gain, kind); print(f"  {kind:7s} " + "  ".join(f"{k}: {a:.3f}->{z:.3f}" for k, (a, z) in r.items()), flush=True)
