"""Why do workers die young? Same colony, different ant instincts and energy costs. 10000 steps, 4 seeds each."""
import numpy as np
from multiprocessing import Pool
import colony_life as cl
import grown_brain as gb

# innate drives (rows: forage, build, eat, rest; columns: hunger, food store, nest needs work, carrying food, fatigue)
INSTINCTS = [0.5, -1.0, 0.0, 1.5, -1.0,    -1.0, 0.8, 1.5, -2.0, -0.8,    4.0, 0.3, 0.0, -3.0, 0.0,    -1.5, 0.0, 0.0, -2.0, 3.0]
VARIANTS = {
    "new defaults (instincts, burn 0.05)": (dict(), 0.05),
    "old behaviour (no instincts, burn 0.08)": (dict(task_prior=[0.0] * 20), 0.08),
}


def run(args):
    name, seed = args; mods, burn = VARIANTS[name]
    cl.BURN = burn; c = cl.Colony(seed=seed, genome=mods)
    for _ in range(10000): c.step()
    workers_ever = 2 + c.born
    return dict(deaths=c.died, born=c.born, alive=len(c.workers), delivered=c.total_food, lifetime=c.worker_steps / max(1, workers_ever), store=c.store)


if __name__ == "__main__":
    jobs = [(n, s) for n in VARIANTS for s in range(4)]
    with Pool(4) as p: out = p.map(run, jobs)
    print(f"{'variant':46s} {'deaths':>7s} {'born':>6s} {'alive':>6s} {'food in':>8s} {'avg worker lifetime (steps)':>28s}")
    for n in VARIANTS:
        r = [o for (m, s), o in zip(jobs, out) if m == n]; f = lambda k: np.mean([x[k] for x in r])
        print(f"{n:46s} {f('deaths'):7.1f} {f('born'):6.1f} {f('alive'):6.1f} {f('delivered'):8.0f} {f('lifetime'):28.0f}")
