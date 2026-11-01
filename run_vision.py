"""Do ants with a grown eye avoid the dangerous human better than blind ants?  (fly-scale smell, learning + trails on)"""
import numpy as np
from multiprocessing import Pool
import ant_colony as ac
import grown_brain as gb

CONDS = {
    "sighted (grown eye)": gb.GENOME,
    "blind mutant (no eye field)": {**gb.GENOME, "eye_ap_max": 0.0},
    "short egg gradient (small eye)": {**gb.GENOME, "bcd_lambda": 0.12},
}


def job(name, seed):
    scores, _, picks = ac.run(True, True, episodes=40, seed=seed, make_brain=gb.brain_maker(CONDS[name]), human=True)
    return scores, picks


if __name__ == "__main__":
    jobs = [(n, s) for n in CONDS for s in range(3)]
    with Pool() as pool:
        out = pool.starmap(job, jobs)
    print(f"{'brain':32s} {'stomps/episode: first 8':>24s} {'last 10':>9s} {'food delivered (last 10)':>26s}", flush=True)
    for n in CONDS:
        sel = [o for (m, s), o in zip(jobs, out) if m == n]
        h = np.mean([[p[2] for p in r[1]] for r in sel], axis=0)
        f = np.mean([r[0] for r in sel], axis=0)
        print(f"{n:32s} {h[:8].mean():24.1f} {h[-10:].mean():9.1f} {f[-10:].mean():26.1f}", flush=True)
