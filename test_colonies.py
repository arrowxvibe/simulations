"""Run several colonies headless and report whether they survive and grow."""
from multiprocessing import Pool
import colony_life as cl


def run(seed, steps=4000):
    c = cl.Colony(seed=seed)
    for _ in range(steps):
        c.step()
    return (f"seed {seed}: workers {len(c.workers)} brood {len(c.brood)} store {c.store:.0f} "
            f"chambers {c.chambers} food {c.total_food} born {c.born} died {c.died} queen {c.queen['energy']:.0f}")


if __name__ == "__main__":
    with Pool(6) as p:
        for r in p.map(run, range(1, 7)):
            print(r, flush=True)
