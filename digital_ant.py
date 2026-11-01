"""Digital ant v0.1: an ant-style brain (antennal lobe -> mushroom body -> steering)
living in a 2D world with food and poison odors. It learns by reward and punishment.

Brain layout follows the circuit shared by insects (flies, bees, ants):
  antennae -> antennal lobe glomeruli -> Kenyon cells (sparse, random wiring)
           -> mushroom body output neurons (approach / avoid) -> steering
Learning: dopamine-like reward/punishment strengthens the Kenyon->output synapses
that were recently active (eligibility trace).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARENA = 50.0
N_GLOM = 40          # glomeruli per antenna (real ants have hundreds)
N_KC = 2000          # Kenyon cells
KC_INPUTS = 6        # each Kenyon cell samples ~6 glomeruli, as in insects
KC_SPARSITY = 0.05   # only 5% of Kenyon cells fire
REACH = 1.8


class AntBrain:
    def __init__(self, rng):
        self.rng = rng
        self.W_kc = np.zeros((N_KC, N_GLOM))
        for k in range(N_KC):
            self.W_kc[k, rng.choice(N_GLOM, KC_INPUTS, replace=False)] = 1.0
        self.w_app = np.zeros(N_KC)   # plastic synapses -> approach output neuron
        self.w_avd = np.zeros(N_KC)   # plastic synapses -> avoid output neuron
        self.trace = np.zeros(N_KC)

    def kenyon(self, glom):
        x = self.W_kc @ glom
        thr = np.sort(x)[-int(N_KC * KC_SPARSITY)]
        return np.maximum(x - thr, 0.0)   # winner-take-all, keeps intensity

    def value(self, glom):
        kc = self.kenyon(glom)
        return (self.w_app - self.w_avd) @ kc, kc

    def learn(self, reward, lr=0.005):
        if reward > 0:
            self.w_app = np.clip(self.w_app + lr * self.trace, 0, 1)
        elif reward < 0:
            self.w_avd = np.clip(self.w_avd + lr * self.trace, 0, 1)


class World:
    def __init__(self, rng, n_food=3, n_poison=3):
        self.rng = rng
        self.profile = {  # each odor excites a different, overlapping set of glomeruli
            "food": (rng.random(N_GLOM) < 0.3) * rng.random(N_GLOM),
            "poison": (rng.random(N_GLOM) < 0.3) * rng.random(N_GLOM),
        }
        self.src = [("food", self.spawn()) for _ in range(n_food)] + \
                   [("poison", self.spawn()) for _ in range(n_poison)]

    def spawn(self):
        return self.rng.uniform(5, ARENA - 5, 2)

    def smell(self, pos):
        g = self.rng.normal(0, 0.02, N_GLOM)
        for kind, p in self.src:
            d2 = np.sum((pos - p) ** 2)
            g += np.exp(-d2 / (2 * 6.0 ** 2)) * self.profile[kind]
        return np.maximum(g, 0)


def run_episode(brain, world, steps=600, learn=True, record=False):
    rng = brain.rng
    pos = np.array([ARENA / 2, ARENA / 2])
    heading = rng.uniform(0, 2 * np.pi)
    food = poison = 0
    path = []
    for _ in range(steps):
        vals, kcs = [], []
        for side in (+1, -1):
            a = pos + 1.5 * np.array([np.cos(heading + side * 0.6), np.sin(heading + side * 0.6)])
            v, kc = brain.value(world.smell(a))
            vals.append(v); kcs.append(kc)
        brain.trace = 0.9 * brain.trace + 0.5 * (kcs[0] + kcs[1])
        heading += 0.7 * np.tanh(0.1 * (vals[0] - vals[1])) + rng.normal(0, 0.25)  # steer toward the better-smelling side
        pos = pos + 0.8 * np.array([np.cos(heading), np.sin(heading)])
        if np.any(pos < 0) or np.any(pos > ARENA):  # wall: clip and turn away
            pos = np.clip(pos, 0, ARENA)
            heading += np.pi / 2 + rng.normal(0, 0.3)
        for j, (kind, p) in enumerate(world.src):
            if np.linalg.norm(pos - p) < REACH:
                r = 1 if kind == "food" else -1
                food += r > 0; poison += r < 0
                if learn: brain.learn(r)
                world.src[j] = (kind, world.spawn())
        if record: path.append(pos.copy())
    return food, poison, np.array(path)


def main():
    rng = np.random.default_rng(1)
    brain = AntBrain(rng)
    world = World(rng)
    eps = 60
    f, p = [], []
    for e in range(eps):
        a, b, _ = run_episode(brain, world)
        f.append(a); p.append(b)
        print(f"episode {e+1:2d}: food {a:2d}  poison {b:2d}")
    # control: same world, brand-new untrained brain
    ctrl = [run_episode(AntBrain(np.random.default_rng(100 + i)), world, learn=False)[:2] for i in range(10)]
    cf, cp = np.mean([c[0] for c in ctrl]), np.mean([c[1] for c in ctrl])
    print(f"\nuntrained ant: food {cf:.1f}, poison {cp:.1f} per episode")
    print(f"trained ant (last 10 episodes): food {np.mean(f[-10:]):.1f}, poison {np.mean(p[-10:]):.1f}")

    _, _, path = run_episode(brain, world, learn=False, record=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(f, label="food reached", color="green"); ax[0].plot(p, label="poison hit", color="red")
    ax[0].axhline(cf, ls=":", color="green"); ax[0].axhline(cp, ls=":", color="red")
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("per episode (600 steps)"); ax[0].legend()
    ax[0].set_title("Learning curve (dotted = untrained ant)")
    ax[1].plot(path[:, 0], path[:, 1], lw=0.8, color="gray")
    for kind, s in world.src:
        ax[1].scatter(*s, c="green" if kind == "food" else "red", s=90)
    ax[1].set_xlim(0, ARENA); ax[1].set_ylim(0, ARENA); ax[1].set_aspect("equal")
    ax[1].set_title("Trained ant path (green food, red poison)")
    plt.tight_layout(); plt.savefig("ant_result.png", dpi=120)


if __name__ == "__main__":
    main()
