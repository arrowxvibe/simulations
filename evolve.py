"""Colonies compete; the genomes of the colonies that do well become the next generation.

Run: python evolve.py [generations]      Result: best_genome.json (the live colony loads it automatically)
Each colony starts from an egg-grown naive brain and has to survive 4000 steps (workers eat, forage, build...).
Learning happens within a life; only the genome is inherited, never what an ant learned.
Fitness (no hand-written strategy): mean workers alive + brood born + food delivered + nest standing + queen alive.
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
import colony_life as cl
import grown_brain as gb

STEPS, POP, KEEP, SEEDS = 4000, 12, 4, 2
HERE = os.path.dirname(os.path.abspath(__file__))
EVOLVED = ["task_prior", "task_bias", "task_T", "task_lr", "state_kc", "state_claws", "claw_reach", "visual_kc_fraction", "mb_divisions"]


def fitness(args):
    genome, seed = args
    cl.POP_LIMIT = 8                      # let colonies grow a little while they are being compared
    c = cl.Colony(seed=seed, genome=genome)
    for _ in range(STEPS):
        c.step()
    avg_workers = c.worker_steps / STEPS
    return (10 * avg_workers + 3 * c.born + 0.05 * c.total_food + 5 * len(c.pellets) / cl.MAX_PELLETS
            + (20 if c.queen["energy"] > 0 else 0))


def random_genome(rng):
    g = {**gb.GENOME}
    g["task_prior"] = list(rng.normal(0, 1.0, 20)); g["task_bias"] = list(rng.normal(0, 0.5, 4))
    return g


def mutate(g, rng):
    c = {**g}
    c["task_prior"] = list(np.array(g["task_prior"]) + rng.normal(0, 0.35, 20) * (rng.random(20) < 0.5))
    c["task_bias"] = list(np.array(g["task_bias"]) + rng.normal(0, 0.2, 4))
    c["task_T"] = float(np.clip(g["task_T"] * np.exp(rng.normal(0, 0.15)), 0.25, 3.0))
    c["task_lr"] = float(np.clip(g["task_lr"] * np.exp(rng.normal(0, 0.2)), 0.02, 0.6))
    c["state_kc"] = int(np.clip(g["state_kc"] + rng.integers(-40, 41), 60, 600))
    c["state_claws"] = int(np.clip(g["state_claws"] + rng.integers(-1, 2), 2, 5))
    c["claw_reach"] = float(np.clip(g["claw_reach"] + rng.normal(0, 0.02), 0.1, 0.6))
    c["visual_kc_fraction"] = float(np.clip(g["visual_kc_fraction"] + rng.normal(0, 0.06), 0.0, 1.0))
    c["mb_divisions"] = int(np.clip(g["mb_divisions"] + rng.integers(-60, 61), 150, 900))
    return c


def main(gens):
    rng = np.random.default_rng(0)
    pop = [random_genome(rng) for _ in range(POP)]
    try:   # start from what evolved in the small world: the new world is bigger, so these ants have to adapt again
        old = json.load(open(os.path.join(HERE, "best_genome_big_run1.json")))
        seed = {**gb.GENOME, **{k: v for k, v in old.items() if k in EVOLVED}}   # the best genome of the previous run
        pop = [seed] + [mutate(seed, rng) for _ in range(5)] + pop[: POP - 6]
        print("seeded from the small-world winner", flush=True)
    except Exception:
        pass
    hist = []; best_ever = -1e9
    with Pool(4) as pool:
        for gen in range(gens):
            t0 = time.time()
            jobs = [(g, 1000 * gen + s) for g in pop for s in range(SEEDS)]
            fit = np.array(pool.map(fitness, jobs)).reshape(POP, SEEDS).mean(1)
            order = np.argsort(-fit)
            hist.append((float(fit.mean()), float(fit.max())))
            print(f"gen {gen}: fitness mean {fit.mean():.1f}  best {fit.max():.1f}  ({time.time() - t0:.0f}s)", flush=True)
            best = pop[order[0]]
            if fit.max() > best_ever:
                best_ever = float(fit.max())
                json.dump({k: best[k] for k in EVOLVED} | {"fitness": best_ever, "generation": gen, "history": hist},
                          open(os.path.join(HERE, "best_genome.json"), "w"), indent=1)
            parents = [pop[i] for i in order[:KEEP]]
            pop = [parents[0], parents[1]] + [mutate(parents[int(rng.integers(0, KEEP))], rng) for _ in range(POP - 2)]


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
