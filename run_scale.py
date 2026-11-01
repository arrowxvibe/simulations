"""Fly-scale vs ant-scale grown brains (same colony test, learning + pheromone on)."""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

import ant_colony as ac
import grown_brain as gb

ANT = {**gb.GENOME, **gb.ANT_SCALE}
CONDS = {
    "hand-built, fly scale (40 glom, 2k KC)": (None, 40),
    "grown, fly scale (40 glom, ~2k KC)": (gb.GENOME, 40),
    "grown, ant scale (400 glom, ~17k KC)": (ANT, 400),
}


def job(name, seed):
    t = time.time()
    genome, n = CONDS[name]
    make = None if genome is None else gb.brain_maker(genome)
    scores = ac.run(True, True, seed=seed, make_brain=make, n_glom=n)[0]
    print(f"done {name} seed {seed} in {time.time() - t:.0f}s", flush=True)
    return scores


def main():
    jobs = [(n, s) for n in CONDS for s in range(3)]
    with Pool() as pool:
        out = pool.starmap(job, jobs)
    res = {n: np.mean([o for (m, s), o in zip(jobs, out) if m == n], axis=0) for n in CONDS}
    for n, c in res.items():
        print(f"{n:42s} first 5 episodes: {c[:5].mean():.1f}   last 10: {c[-10:].mean():.1f}", flush=True)
    plt.figure(figsize=(8, 4.6))
    for n, c in res.items():
        plt.plot(c, label=n, lw=2 if "ant scale" in n else 1)
    plt.xlabel("episode"); plt.ylabel("food delivered to nest"); plt.legend(fontsize=8)
    plt.title("Fly-sized vs ant-sized grown brains"); plt.tight_layout(); plt.savefig("scale_result.png", dpi=120)


if __name__ == "__main__":
    main()
