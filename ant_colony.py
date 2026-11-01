"""Digital ant v0.2: a colony with a nest, pheromone trails and a carry-food-home loop.

Each ant has its own mushroom-body brain (see digital_ant.py for the circuit).
New here:
  - nest at the arena centre; ants search, pick up food, walk home, deliver it
  - returning ants lay pheromone; searching ants are innately drawn to it
  - food piles are finite and run out; poison punishes with a stronger lesson
  - ablation: learning on/off x pheromone on/off, to see which part helps
"""
import numpy as np

ARENA = 120
NEST = np.array([ARENA / 2.0, ARENA / 2.0])
N_GLOM, N_KC, KC_INPUTS, KC_SPARSITY = 40, 2000, 6, 0.05
REACH, N_ANTS, STEPS = 1.8, 8, 800


# vision-learning knobs (changed by the tuning experiments)
TUNE = dict(stomp_mult=1.0, fear_r=0.0, fear_lr=0.0, carry_vision=False, avoid_gain=1.0)


class AntBrain:
    """Kenyon cells are stored as claw lists (idx, msk): a cell has at most ~7 inputs, so a dense matrix is wasteful."""
    def __init__(self, rng, W=None, idx=None, msk=None, n_vis=0, n_bins=0):
        self.n_vis, self.n_bins = n_vis, n_bins   # visual input channels (0 = blind brain)
        if idx is None:
            if W is None:  # hand-built brain: random wiring (the v0.1 baseline)
                W = np.zeros((N_KC, N_GLOM))
                for k in range(N_KC):
                    W[k, rng.choice(N_GLOM, KC_INPUTS, replace=False)] = 1.0
            m = max(1, int(W.sum(1).max()))
            idx = np.argsort(-W, axis=1)[:, :m]
            msk = np.take_along_axis(W, idx, axis=1)
        self.idx, self.msk = idx, msk
        n = idx.shape[0]
        self.scale = n / 2000.0   # keeps steering strength comparable across brain sizes
        self.w_app = np.zeros(n)
        self.w_avd = np.zeros(n)
        self.trace = np.zeros(n)
        self.n_top = max(1, int(n * KC_SPARSITY))

    def kenyon(self, glom):
        x = (glom[self.idx] * self.msk).sum(1)
        thr = np.partition(x, -self.n_top)[-self.n_top]
        return np.maximum(x - thr, 0.0)

    def learn(self, reward, mult=1.0):
        if reward > 0:
            self.w_app = np.clip(self.w_app + 0.005 * self.trace, 0, 1)
        else:
            self.w_avd = np.clip(self.w_avd + 0.02 * mult * self.trace, 0, 1)
        self.trace[:] = 0

    def fear(self, lr, n_olf=0):
        """Ground vibration from the human's footsteps: a mild warning that strengthens avoidance without clearing the trace.
        Only the cells that carry VISION take the lesson, so fear of the human does not spread onto the smell of food."""
        if getattr(self, "_vm", None) is None:
            self._vm = ((self.idx >= n_olf) & (self.msk > 0)).any(1).astype(np.float64) if self.n_vis else np.zeros(len(self.trace))
        self.w_avd = np.clip(self.w_avd + lr * self.trace * self._vm, 0, 1)


class Ant:
    def __init__(self, rng, make_brain=None):
        self.brain = make_brain(rng) if make_brain else AntBrain(rng)
        self.reset(rng)

    def reset(self, rng):
        self.pos = NEST.copy()
        self.heading = rng.uniform(0, 2 * np.pi)
        self.carrying = False
        self.brain.trace[:] = 0


class World:
    def __init__(self, rng, n_glom=N_GLOM, overlap=0.0, noise=0.02):
        self.rng, self.n_glom, self.noise = rng, n_glom, noise
        self.food_picks = self.poison_hits = 0
        prng = np.random.default_rng(42)  # same food/poison smells in every episode, so learning can carry over
        self.profile = {k: (prng.random(n_glom) < 0.3) * prng.random(n_glom) for k in ("food", "poison")}
        # overlap: how much the poison smell is a copy of the food smell (1 = identical, impossible to tell apart)
        self.profile["poison"] = overlap * self.profile["food"] + (1 - overlap) * self.profile["poison"]
        self.src = [[("food", self.spawn()), 15] for _ in range(20)] + \
                   [[("poison", self.spawn()), 0] for _ in range(8)]
        self.src = [[k, p, a] for (k, p), a in self.src]
        self.pher = np.zeros((ARENA, ARENA))
        self.human, self.human_on, self.human_hits, self.hwp = None, False, 0, None

    def spawn(self):
        while True:     # ants forage within a limited range of their nest, not across the whole map
            r, a = self.rng.uniform(12, 48), self.rng.uniform(0, 2 * np.pi)
            p = NEST + r * np.array([np.cos(a), np.sin(a)])
            if 4 < p[0] < ARENA - 4 and 4 < p[1] < ARENA - 4:
                return p

    def smell(self, pos):
        g = self.rng.normal(0, self.noise, self.n_glom)
        for kind, p, _ in self.src:
            g += np.exp(-np.sum((pos - p) ** 2) / (2 * 6.0 ** 2)) * self.profile[kind]
        return np.maximum(g, 0)

    def see(self, pos, heading, n_bins, side):
        """What one eye sees: for each of 4 channels (food, poison, big animal, nest) the closeness/size seen in each
        direction bin across ~300 degrees. Each side's eye sees its own half (side +1 = left)."""
        out = np.zeros(4 * n_bins, np.float32)
        objs = [(0 if k == "food" else 1, p, 1.0) for k, p, _ in self.src] + [(3, NEST, 3.0)]
        if self.human is not None: objs.append((2, self.human, 2.5))
        for cls, p, rad in objs:
            d = p - pos; dist = float(np.hypot(d[0], d[1]))
            if dist > 25 or dist < 1e-6: continue
            b = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
            if abs(b) > 2.6 or side * b < 0: continue
            k = min(n_bins - 1, int((b + 2.6) / 5.2 * n_bins)); i = cls * n_bins + k
            out[i] = max(out[i], min(1.0, 3.0 * rad / (dist + 1.0)))
        return out

    def step_human(self):
        """A patrolling human (headless tests): walks between waypoints, often near the nest."""
        if self.human is None:
            self.human = self.rng.uniform(5, 45, 2); self.hwp = self.rng.uniform(5, 45, 2)
        d = self.hwp - self.human; n = float(np.hypot(d[0], d[1]))
        if n < 2:
            self.hwp = np.clip(NEST + self.rng.normal(0, 7, 2), 3, 47) if self.rng.random() < 0.4 else self.rng.uniform(5, 45, 2)
        self.human = self.human + d / max(n, 1e-6) * 0.6

    def pher_at(self, pos):
        i, j = np.clip(pos.astype(int), 0, ARENA - 1)
        return self.pher[i, j]

    def step_pheromone(self):
        p = self.pher
        blur = (p + np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1)) / 5
        self.pher = 0.985 * blur


def ant_step(ant, world, rng, learning, use_pher):
    b = ant.brain
    d = np.array([np.cos(ant.heading), np.sin(ant.heading)])
    if ant.carrying:
        # path integration: real ants always know the direction home
        to_home = NEST - ant.pos
        want = np.arctan2(to_home[1], to_home[0])
        err = (want - ant.heading + np.pi) % (2 * np.pi) - np.pi
        ant.heading += 0.5 * np.clip(err, -1, 1) + rng.normal(0, 0.1)
        world.pher[tuple(np.clip(ant.pos.astype(int), 0, ARENA - 1))] += 1.0
        if TUNE['carry_vision'] and b.n_vis and world.human is not None:      # a laden ant still watches out for the human
            vv = []
            for side in (+1, -1):
                g = np.concatenate([np.zeros(world.n_glom), world.see(ant.pos, ant.heading, b.n_bins, side)]); kc = b.kenyon(g)
                vv.append(-(b.w_avd @ kc) / b.scale); b.trace = 0.9 * b.trace + 0.25 * kc
            ant.heading += TUNE['avoid_gain'] * 0.7 * np.tanh(0.1 * (vv[0] - vv[1]))
    else:
        turn = rng.normal(0, 0.25)
        vals, phs = [], []
        for side in (+1, -1):
            a = ant.pos + 1.5 * np.array([np.cos(ant.heading + side * 0.6), np.sin(ant.heading + side * 0.6)])
            g = world.smell(a)
            if b.n_vis: g = np.concatenate([g, world.see(ant.pos, ant.heading, b.n_bins, side)])
            kc = b.kenyon(g)
            vals.append((b.w_app - b.w_avd) @ kc / b.scale)
            b.trace = 0.9 * b.trace + 0.25 * kc
            phs.append(world.pher_at(a))
        turn += (TUNE['avoid_gain'] if world.human is not None else 1.0) * 0.7 * np.tanh(0.1 * (vals[0] - vals[1]))
        if use_pher:
            turn += 0.8 * np.tanh(2.0 * (phs[0] - phs[1]))
        ant.heading += turn
    ant.pos = ant.pos + 0.8 * np.array([np.cos(ant.heading), np.sin(ant.heading)])
    if np.any(ant.pos < 0) or np.any(ant.pos > ARENA):
        ant.pos = np.clip(ant.pos, 0, ARENA)
        ant.heading += np.pi / 2 + rng.normal(0, 0.3)

    if world.human is not None:                      # the human is dangerous: get under his feet and you are hurt
        dh = ant.pos - world.human; nh = float(np.hypot(dh[0], dh[1]))
        if nh < 2.2:
            world.human_hits += 1; ant.stomped = True
            ant.pos = ant.pos + dh / max(nh, 1e-6) * 2.0; ant.heading += np.pi + rng.normal(0, 0.3)
            if learning: b.learn(-1, TUNE['stomp_mult'])
        elif learning and nh < TUNE['fear_r']:
            b.fear(TUNE['fear_lr'], world.n_glom)
    delivered = 0
    if ant.carrying:
        if np.linalg.norm(ant.pos - NEST) < 3:
            ant.carrying = False
            delivered = 1
            ant.heading += np.pi
    else:
        for s in world.src:
            kind, p, amt = s
            if np.linalg.norm(ant.pos - p) < REACH:
                if kind == "food":
                    world.food_picks += 1
                    ant.carrying = True
                    s[2] -= 1
                    if learning: b.learn(+1)
                    if s[2] <= 0:
                        s[1] = world.spawn(); s[2] = 15
                else:
                    world.poison_hits += 1
                    if learning: b.learn(-1)
                    s[1] = world.spawn()
                    ant.heading += np.pi
                break
    return delivered


def run(learning, use_pher, episodes=30, seed=0, record_last=False, make_brain=None, n_glom=N_GLOM, overlap=0.0, noise=0.02, human=False):
    rng = np.random.default_rng(seed)
    ants = [Ant(rng, make_brain) for _ in range(N_ANTS)]
    scores, snap, picks = [], None, []
    for e in range(episodes):
        world = World(np.random.default_rng(1000 + e), n_glom, overlap, noise)  # new layout each episode
        world.human_on = human
        for a in ants: a.reset(rng)
        total = 0
        for _ in range(STEPS):
            if human: world.step_human()
            for a in ants:
                total += ant_step(a, world, rng, learning, use_pher)
            world.step_pheromone()
        scores.append(total); picks.append((world.food_picks, world.poison_hits, world.human_hits))
        if record_last and e == episodes - 1:
            snap = (world.pher.copy(), [s[:2] for s in world.src])
    return scores, snap, picks


def _job(name, seed, lrn, ph):
    return run(lrn, ph, seed=seed, record_last=(name == "both" and seed == 0))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt          # only the experiment plots need it
    conds = {"neither": (False, False), "pheromone only": (False, True),
             "learning only": (True, False), "both": (True, True)}
    from multiprocessing import Pool
    jobs = [(n, s, l, p) for n, (l, p) in conds.items() for s in range(3)]
    with Pool() as pool:
        out = pool.starmap(_job, jobs)
    res, snap = {}, None
    for name in conds:
        runs = [o for (n, s, l, p), o in zip(jobs, out) if n == name]
        res[name] = np.mean([r[0] for r in runs], axis=0)
        if name == "both": snap = runs[0][1]
        print(f"{name:15s} food delivered per episode, last 10: {res[name][-10:].mean():.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, curve in res.items():
        ax[0].plot(curve, label=name, lw=2 if name == "both" else 1)
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("food delivered to nest"); ax[0].legend()
    ax[0].set_title("Colony of 8 ants (mean of 3 runs)")
    pher, srcs = snap
    ax[1].imshow(pher.T, origin="lower", cmap="magma", extent=(0, ARENA, 0, ARENA))
    for (kind, p) in srcs:
        ax[1].scatter(*p, c="lime" if kind == "food" else "cyan", s=70, edgecolors="w")
    ax[1].scatter(*NEST, marker="s", c="white", s=90)
    ax[1].set_title("Pheromone trails at end (white square = nest)")
    plt.tight_layout(); plt.savefig("colony_result.png", dpi=120)


if __name__ == "__main__":
    main()
