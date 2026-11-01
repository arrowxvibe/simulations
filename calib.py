import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from flybrainx import FlyBrainX
from maleflybrain import MaleBrainX
from flyworld2 import FlyWorld2, KINDS
P = 32
b = (MaleBrainX(gain=0.3, g_smell=float(os.environ.get('G_SMELL', 0.1)), g_mech=float(os.environ.get('G_MECH', 4.0))) if os.environ.get('BRAIN') == 'male' else FlyBrainX(gain=0.3, g_smell=float(os.environ.get('G_SMELL', 1.0)), g_mech=float(os.environ.get('G_MECH', 1.5)))); th = b.theta[None].repeat(P, 1)
w = FlyWorld2(P); obs = w.reset(seeds=list(range(P))); b.reset_batch(P)
gf_log = {True: [], False: []}; dist_log = []
for t in range(160):
    act = np.zeros((P, 4), bool); act[:, 0] = True; act[:, 3] = True   # fly straight, no steering
    sm, me = w.smell(), w.mech()
    lg, gf, l, r = b.act_x(obs, th, sm, me)
    d = np.linalg.norm(w.sol[:, 0] - w.pos, axis=1)
    for i in range(P): gf_log[bool(w.approach[i])].append((float(gf[i]), d[i]))
    obs, _, _ = w.step(act)
for k in (True, False):
    a = np.array(gf_log[k]); near = a[a[:, 1] < 8][:, 0]; far = a[a[:, 1] > 15][:, 0]
    print("human approaching" if k else "human wandering", "| GF when human <8 away: mean %.4f p90 %.4f max %.4f | when >15 away: mean %.4f p90 %.4f max %.4f" % (near.mean() if len(near) else -1, np.percentile(near,90) if len(near) else -1, near.max() if len(near) else -1, far.mean(), np.percentile(far,90), far.max()))
