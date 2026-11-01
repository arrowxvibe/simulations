"""Tune vision learning: for each variant, sighted vs blind ants with a patrolling human. 4 seeds, 40 episodes."""
import numpy as np
from multiprocessing import Pool
import ant_colony as ac
import grown_brain as gb

VARIANTS = {   # fear now changes only the visual memory cells
    "V0 baseline":                          dict(),
    "V1 + footstep warning (vision only)":  dict(fear_r=6.0, fear_lr=0.02),
    "V2 + laden ants watch, x2 stomp":      dict(fear_r=6.0, fear_lr=0.02, carry_vision=True, stomp_mult=2.0),
}
_OLD = {
    "V0 baseline":                          dict(),
    "V1 + footstep-vibration warning":      dict(fear_r=6.0, fear_lr=0.02),
    "V2 + laden ants watch too, x5 lesson": dict(fear_r=6.0, fear_lr=0.02, carry_vision=True, stomp_mult=5.0),
    "V3 + stronger avoidance steering":     dict(fear_r=6.0, fear_lr=0.02, carry_vision=True, stomp_mult=5.0, avoid_gain=2.0),
}
BRAINS = {"sighted": gb.GENOME, "blind": {**gb.GENOME, "eye_ap_max": 0.0}}


def job(v, b, seed):
    ac.TUNE.update(dict(stomp_mult=1.0, fear_r=0.0, fear_lr=0.0, carry_vision=False, avoid_gain=1.0)); ac.TUNE.update(VARIANTS[v])
    scores, _, picks = ac.run(True, True, episodes=40, seed=seed, make_brain=gb.brain_maker(BRAINS[b]), human=True)
    return scores, [p[2] for p in picks]


if __name__ == "__main__":
    jobs = [(v, b, s) for v in VARIANTS for b in BRAINS for s in range(4)]
    with Pool(4) as pool:
        out = pool.starmap(job, jobs)
    print(f"{'variant':38s} {'brain':8s} {'stomps/ep first 8':>18s} {'last 10':>8s} {'food last 10':>13s}", flush=True)
    for v in VARIANTS:
        res = {}
        for b in BRAINS:
            sel = [o for (vv, bb, s), o in zip(jobs, out) if vv == v and bb == b]
            h = np.mean([o[1] for o in sel], axis=0); f = np.mean([o[0] for o in sel], axis=0)
            res[b] = (h[:8].mean(), h[-10:].mean(), f[-10:].mean())
            print(f"{v:38s} {b:8s} {h[:8].mean():18.1f} {h[-10:].mean():8.1f} {f[-10:].mean():13.1f}", flush=True)
        s, bl = res["sighted"], res["blind"]
        print(f"{'':38s} -> sighted has {100 * (1 - s[1] / bl[1]):.0f}% fewer stomps than blind at the end; food {100 * (s[2] / bl[2] - 1):+.0f}%", flush=True)
