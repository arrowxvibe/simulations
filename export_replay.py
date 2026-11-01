"""Train the colony (learning + pheromone), then record one full episode as JSON for the 3D viewer."""
import json
import numpy as np
import ant_colony as ac


def main(train_eps=20, every=2):
    rng = np.random.default_rng(0)
    ants = [ac.Ant(rng) for _ in range(ac.N_ANTS)]
    for e in range(train_eps):
        world = ac.World(np.random.default_rng(1000 + e))
        for a in ants: a.reset(rng)
        for _ in range(ac.STEPS):
            for a in ants: ac.ant_step(a, world, rng, True, True)
            world.step_pheromone()
        print("trained episode", e + 1)

    world = ac.World(np.random.default_rng(777))
    for a in ants: a.reset(rng)
    frames, delivered = [], 0
    for t in range(ac.STEPS):
        for a in ants:
            delivered += ac.ant_step(a, world, rng, False, True)
        world.step_pheromone()
        if t % every == 0:
            f = {
                "a": [[round(float(a.pos[0]), 2), round(float(a.pos[1]), 2),
                       round(float(a.heading), 2), int(a.carrying)] for a in ants],
                "s": [[k, round(float(p[0]), 1), round(float(p[1]), 1), int(am)] for k, p, am in world.src],
                "d": delivered,
            }
            if t % (every * 5) == 0:
                f["p"] = np.round(world.pher, 2).T.flatten().tolist()
            frames.append(f)
    json.dump({"arena": ac.ARENA, "nest": ac.NEST.tolist(), "frames": frames}, open("replay.json", "w"))
    print("frames", len(frames), "delivered", delivered)


if __name__ == "__main__":
    main()
