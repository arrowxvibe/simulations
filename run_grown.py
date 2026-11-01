"""Grown brain vs hand-built brain vs mutants (same colony test as before: learning + pheromone on)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

import ant_colony as ac
import grown_brain as gb

CONDS = {
    "hand-built (v0.2)": None,
    "grown from genome": {},
    "mutant: short egg gradient": dict(bcd_lambda=0.12),
    "mutant: neuroblasts stop early": dict(mb_divisions=100),
    "mutant: few antennal-lobe types": dict(pn_types_per_lineage=1),
}


def job(name, seed):
    mods = CONDS[name]
    make = None if mods is None else gb.brain_maker({**gb.GENOME, **mods})
    return ac.run(True, True, seed=seed, make_brain=make)[0]


def main():
    for name, mods in CONDS.items():
        if mods is not None:
            print(f"{name:32s}", gb.summary(gb.develop({**gb.GENOME, **mods}, np.random.default_rng(0))), flush=True)
    jobs = [(n, s) for n in CONDS for s in range(3)]
    with Pool() as pool:
        out = pool.starmap(job, jobs)
    res = {n: np.mean([o for (m, s), o in zip(jobs, out) if m == n], axis=0) for n in CONDS}
    for n, c in res.items():
        print(f"{n:32s} food delivered per episode, last 10: {c[-10:].mean():.1f}", flush=True)
    plt.figure(figsize=(8, 4.6))
    for n, c in res.items():
        plt.plot(c, label=n, lw=2 if n.startswith("grown") else 1)
    plt.xlabel("episode"); plt.ylabel("food delivered to nest"); plt.legend(fontsize=8)
    plt.title("Hand-built vs gene-grown brains"); plt.tight_layout(); plt.savefig("grown_result.png", dpi=120)


if __name__ == "__main__":
    main()
