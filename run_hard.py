"""Harder smell task: poison smells increasingly like food. Does a bigger (ant-sized) brain discriminate better?"""
import numpy as np
from multiprocessing import Pool
import ant_colony as ac
import grown_brain as gb

ANT = {**gb.GENOME, **gb.ANT_SCALE}
BRAINS = {"fly scale (40 glom)": (gb.GENOME, 40), "ant scale (400 glom)": (ANT, 400)}
OVERLAPS = [0.6, 0.9]
NOISE = 0.05


def job(name, overlap, seed):
    genome, n = BRAINS[name]
    scores, _, picks = ac.run(True, True, seed=seed, make_brain=gb.brain_maker(genome), n_glom=n, overlap=overlap, noise=NOISE)
    return scores, picks


if __name__ == "__main__":
    jobs = [(b, o, s) for b in BRAINS for o in OVERLAPS for s in range(3)]
    with Pool() as pool:
        out = pool.starmap(job, jobs)
    print(f"{'brain':22s} {'overlap':>7s} {'food delivered':>15s} {'food pickups':>13s} {'poison hits':>12s} {'poison share':>13s}")
    for b in BRAINS:
        for o in OVERLAPS:
            sel = [r for (bb, oo, s), r in zip(jobs, out) if bb == b and oo == o]
            deliv = np.mean([r[0][-10:] for r in sel])
            f = np.mean([[p[0] for p in r[1][-10:]] for r in sel]); p = np.mean([[q[1] for q in r[1][-10:]] for r in sel])
            print(f"{b:22s} {o:7.1f} {deliv:15.1f} {f:13.1f} {p:12.1f} {p / (f + p):13.2f}", flush=True)
