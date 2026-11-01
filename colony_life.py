"""Colony life: a queen and workers that forage, eat, build, rest and reproduce.

WHO DECIDES WHAT
  Ants decide for themselves. Every worker has a task circuit grown from its genome (decide.py): it reads its own
  drives (hunger, the food store it can feel, how much the nest needs work, whether it carries food, fatigue) and chooses
  forage / build / eat / rest. Consequences teach it: food delivered, a meal when hungry, a repaired nest, rest when tired
  are rewarded; wasted trips, eating when full or starving are not. No rule assigns jobs.
  World rules (physics, not decisions): energy burns, fatigue builds up and slows ants, stored food spoils,
  wind and rain wear the nest away, larvae need food, brood takes time to develop.
  Still rule-driven: the queen (she walks about her mound and lays eggs when there is food and room) and the 3-ant cap.
"""
import numpy as np
import ant_colony as ac
import grown_brain as gb
from decide import TaskCircuit, TASKS

NEST = ac.NEST
PELLETS_PER_CHAMBER, MAX_CHAMBERS, ROOM_PER_CHAMBER = 10, 40, 3     # every chamber the ants dig makes room for 3 more
MAX_PELLETS = PELLETS_PER_CHAMBER * MAX_CHAMBERS
EGG_T, LARVA_T, PUPA_T = 80, 280, 400
POP_LIMIT = 100   # no other limit: the colony starts with a queen and 2 workers and can grow to 100 ants, as food and room allow
START_WORKERS = 2
FORAGE, BUILD, EAT, REST = range(4)


class Worker(ac.Ant):
    def __init__(self, rng, make_brain, uid, genome):
        super().__init__(rng, make_brain)
        self.id, self.energy, self.fed = uid, 100.0, 0
        self.fatigue, self.task, self.mode, self.target, self.task_t, self.pained = 0.0, None, "idle", None, 0, False
        self.bites, self.bite_cd = 0, 0
        self.tc = TaskCircuit(rng, genome)

    def state_code(self):
        if self.mode == "forage": return 1 if self.carrying else 0
        return {"eat": 2, "fetch": 3, "place": 4, "rest": 6, "rest_go": 6, "idle": 0}[self.mode]


class Colony:
    def __init__(self, seed=None, genome=None, make_brain=None):
        self.rng = np.random.default_rng(seed)
        self.genome = {**gb.GENOME, **(genome or {})}
        self.make_brain = make_brain or gb.brain_maker(self.genome)
        self.world = ac.World(np.random.default_rng(5), self.genome["n_or"])
        self.uid = 0
        self.workers = [self.new_worker() for _ in range(START_WORKERS)]
        self.queen = dict(pos=NEST.copy(), heading=0.0, energy=200.0, since_egg=0, wp=NEST.copy(), wp_t=0, idle=0)
        self.store, self.pellets, self.brood = 60.0, [], []      # a starting supply of food for the queen and the first workers
        self.total_food, self.step_n, self.born, self.died, self.worker_steps = 0, 0, 0, 0, 0
        self.events = []

    def new_worker(self):
        self.uid += 1
        return Worker(self.rng, self.make_brain, self.uid, self.genome)

    @property
    def chambers(self):
        return min(MAX_CHAMBERS, 1 + len(self.pellets) // PELLETS_PER_CHAMBER)

    @property
    def capacity(self):
        cap = 8 + ROOM_PER_CHAMBER * (self.chambers - 1)
        return cap if POP_LIMIT is None else min(cap, POP_LIMIT)

    def chamber_centre(self, k):
        ring, pos = divmod(k // PELLETS_PER_CHAMBER, 8)      # chambers are laid out in rings around the entrance
        a = 2 * np.pi * pos / 8 + 0.3 + 0.4 * ring
        return NEST + (6.5 + 3.4 * ring) * np.array([np.cos(a), np.sin(a)])

    def goto(self, w, target, turn_noise=0.1):
        d = target - w.pos
        err = (np.arctan2(d[1], d[0]) - w.heading + np.pi) % (2 * np.pi) - np.pi
        w.heading += 0.5 * np.clip(err, -1, 1) + self.rng.normal(0, turn_noise)
        w.pos = np.clip(w.pos + 0.8 * np.array([np.cos(w.heading), np.sin(w.heading)]), 0, ac.ARENA)
        return np.linalg.norm(d)

    def log(self, msg):
        self.events.append(msg)

    # ---- the ant's decision loop --------------------------------------------------------------------
    def drives(self, w):
        return np.array([1 - w.energy / 100.0, min(1.0, self.store / 60.0), 1 - len(self.pellets) / MAX_PELLETS,
                         1.0 if w.carrying else 0.0, w.fatigue])

    def assign_task(self, w):
        w.pained = False
        w.task = w.tc.decide(self.drives(w), self.rng)
        w.start = (w.energy, w.fatigue)
        if w.task == FORAGE:
            w.mode, w.task_t = "forage", 500
        elif w.task == BUILD:
            if len(self.pellets) >= MAX_PELLETS:
                self.finish(w, -0.1); return                       # nothing to build: a wasted decision
            a, r = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(9, 13)
            w.target, w.mode, w.task_t = np.clip(NEST + r * np.array([np.cos(a), np.sin(a)]), 2, ac.ARENA - 2), "fetch", 350
        elif w.task == EAT:
            w.mode, w.task_t = "eat", 220
        else:
            w.mode, w.task_t = "rest_go", int(self.rng.integers(120, 240))

    def finish(self, w, reward):
        w.tc.learn(w.task, reward)
        w.task, w.mode = None, "idle"

    def step_worker(self, w):
        w.energy -= 0.03 if w.mode == "rest" else 0.08
        w.fatigue = float(np.clip(w.fatigue + (-0.004 if w.mode == "rest" else 0.0004), 0, 1))
        w.bite_cd = max(0, w.bite_cd - 1)
        if self.world.human is not None and w.bite_cd == 0 and np.linalg.norm(w.pos - self.world.human) < 3.6 and self.rng.random() < 0.04:
            w.bites += 1; w.bite_cd = 160; self.log(f"Ant {w.id} bit the human")          # ants defend themselves
        if w.energy <= 0:
            self.log(f"Ant {w.id} starved"); self.died += 1; return False
        if w.energy < 15 and not w.pained and w.task is not None and w.task != EAT:     # starving hurts whatever it was doing instead of eating
            w.pained = True; w.tc.learn(w.task, -0.4)
        if w.task is None:
            self.assign_task(w)
            if w.task is None: return True
        w.task_t -= 1
        slow = self.rng.random() < 0.5 * w.fatigue                                     # tired ants move less
        if w.task == FORAGE:
            if not slow:
                was, avd = w.carrying, w.brain.w_avd.sum()
                if ac.ant_step(w, self.world, self.rng, True, True):
                    self.store += 1; self.total_food += 1; w.fed += 1
                    self.log(f"Ant {w.id} delivered food (store {self.store:.0f})")
                    self.finish(w, 1.0 if self.store < 40 else 0.3); return True
                elif w.carrying and not was:
                    self.log(f"Ant {w.id} found food and learned its smell")
                elif w.brain.w_avd.sum() > avd and not getattr(w, "stomped", False):
                    self.log(f"Ant {w.id} hit poison and learned to avoid it")
                if getattr(w, "stomped", False):
                    w.stomped = False; w.energy -= 12; w.tc.learn(FORAGE, -0.5)
                    self.log(f"Ant {w.id} was stepped on by the human and learned to avoid him")
            if w.task_t <= 0 and not w.carrying:
                self.finish(w, -0.1)
        elif w.task == BUILD:
            if w.mode == "fetch":
                if not slow and self.goto(w, w.target) < 1.5:
                    w.mode, w.target = "place", self.chamber_centre(len(self.pellets))
            else:
                if len(self.pellets) >= MAX_PELLETS:
                    self.finish(w, 0.0); return True
                if not slow and self.goto(w, w.target) < 1.5:
                    self.pellets.append(self.pellet_pos(len(self.pellets))); w.energy -= 3
                    if len(self.pellets) % PELLETS_PER_CHAMBER == 0:
                        self.log(f"Ant {w.id} finished chamber {self.chambers}: room for {self.capacity} ants")
                    self.finish(w, 0.8); return True
            if w.task_t <= 0: self.finish(w, -0.1)
        elif w.task == EAT:
            if not slow and self.goto(w, NEST) < 2.5:
                if self.store >= 1:
                    gain = min(100.0 - w.energy, 60.0); self.store -= 1; w.energy += gain
                    self.log(f"Ant {w.id} ate from the store")
                    self.finish(w, 1.2 * gain / 60.0 - 0.1)         # a meal is only worth something when hungry
                else:
                    self.finish(w, -0.2)                            # the store was empty
            elif w.task_t <= 0:
                self.finish(w, -0.1)
        else:                                                       # rest, in the nest
            if w.mode == "rest_go":
                if np.linalg.norm(w.pos - NEST) > 3.0:
                    self.goto(w, NEST)
                else:
                    w.mode = "rest"
            else:
                w.heading += self.rng.normal(0, 0.1)
            if w.task_t <= 0:
                self.finish(w, 0.5 * w.start[1] - 0.05)             # worth it if it was tired
        return True

    def pellet_pos(self, k):
        j, c = k % PELLETS_PER_CHAMBER, self.chamber_centre(k)
        if j < 6:
            a = j * np.pi / 3; return [round(float(c[0] + 1.0 * np.cos(a)), 2), round(float(c[1] + 1.0 * np.sin(a)), 2), 0]
        if j < 9:
            a = j * 2.1; return [round(float(c[0] + 0.5 * np.cos(a)), 2), round(float(c[1] + 0.5 * np.sin(a)), 2), 1]
        return [round(float(c[0]), 2), round(float(c[1]), 2), 2]

    def step_queen(self):
        q = self.queen
        q["energy"] -= 0.03
        if q["energy"] < 120 and self.store >= 1:
            self.store -= 1; q["energy"] = min(200.0, q["energy"] + 60); self.log("Queen ate from the store")
        if q["energy"] <= 0:
            if not q.get("dead"): self.log("The queen starved: the colony is doomed")
            q["dead"] = True; q["energy"] = 0
        q["wp_t"] -= 1
        if q["idle"] > 0:
            q["idle"] -= 1
        elif q["wp_t"] <= 0 or np.linalg.norm(q["wp"] - q["pos"]) < 0.4:
            a, r = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(1.6, 4.0)
            q["wp"] = NEST + r * np.array([np.cos(a), np.sin(a)]); q["wp_t"] = int(self.rng.integers(150, 380))
            if self.rng.random() < 0.3: q["idle"] = int(self.rng.integers(60, 160))
        else:
            d = q["wp"] - q["pos"]; q["heading"] = np.arctan2(d[1], d[0]) + self.rng.normal(0, 0.08)
            q["pos"] = q["pos"] + 0.3 * np.array([np.cos(q["heading"]), np.sin(q["heading"])])
        q["since_egg"] += 1
        room = self.capacity - (1 + len(self.workers) + len(self.brood))
        if q["energy"] > 0 and q["since_egg"] > 50 and self.store >= 8 + 2 * (len(self.workers) + len(self.brood)) and room > 0:    # she only lays when there is a surplus to feed the new ants
            self.store -= 2; q["since_egg"] = 0
            ang = self.rng.uniform(0, 2 * np.pi)
            self.brood.append(dict(age=0, off=[float(1.8 * np.sqrt(self.rng.random()) * np.cos(ang)),
                                               float(1.8 * np.sqrt(self.rng.random()) * np.sin(ang))]))
            self.log("The queen laid an egg")

    def step_brood(self):
        keep = []
        for b in self.brood:
            if EGG_T <= b["age"] < EGG_T + LARVA_T:       # larvae eat
                if self.store >= 0.02:
                    self.store -= 0.015; b["age"] += 1
            else:
                b["age"] += 1
            if b["age"] == EGG_T + LARVA_T:
                self.log("A larva became a pupa: its brain is growing")
            if b["age"] >= EGG_T + LARVA_T + PUPA_T:
                w = self.new_worker(); self.workers.append(w); self.born += 1
                self.log(f"Ant {w.id} was born (new worker, own grown brain)")
            else:
                keep.append(b)
        self.brood = keep

    def step(self):
        self.step_n += 1
        self.store *= 1 - 0.0003                                    # world rule: stored food spoils
        if self.step_n % 500 == 0 and self.pellets and self.rng.random() < 0.6:
            self.pellets.pop(); self.log("Wind and rain wore away some of the nest")   # world rule: weather
        self.workers = [w for w in self.workers if self.step_worker(w)]
        self.worker_steps += len(self.workers)
        self.step_queen(); self.step_brood(); self.world.step_pheromone()

    def frame(self, with_pher):
        q = self.queen
        a = [[0, 0, round(float(q["pos"][0]), 2), round(float(q["pos"][1]), 2), round(float(q["heading"]), 2), 5, 0, 0, 0, round(q["energy"] / 2), [0, 0, 0, 0], 0]]
        for w in self.workers:
            a.append([w.id, 1, round(float(w.pos[0]), 2), round(float(w.pos[1]), 2), round(float(w.heading), 2),
                      w.state_code(), w.fed, round(float(w.brain.w_app.sum()), 1), round(float(w.brain.w_avd.sum()), 1),
                      round(w.energy), [round(float(p), 2) for p in w.tc.probs], w.bites])
        f = {"t": self.step_n, "d": self.total_food, "store": round(self.store, 1), "ch": self.chambers,
             "cap": self.capacity, "born": self.born, "died": self.died, "a": a,
             "b": [[round(float(NEST[0] + b["off"][0]), 2), round(float(NEST[1] + b["off"][1]), 2),
                    0 if b["age"] < EGG_T else (1 if b["age"] < EGG_T + LARVA_T else 2)] for b in self.brood],
             "pl": self.pellets,
             "s": [[k, round(float(p[0]), 1), round(float(p[1]), 1), int(am)] for k, p, am in self.world.src],
             "ev": self.events[-6:]}
        self.events = []
        if with_pher:
            f["p"] = np.round(np.minimum(self.world.pher, 9.9) * 10).astype(int).T.flatten().tolist()
        return f


if __name__ == "__main__":
    c = Colony(seed=1)
    for i in range(1, 4001):
        c.step()
        if i % 500 == 0:
            tasks = [TASKS[w.task] if w.task is not None else "-" for w in c.workers]
            print(f"step {i}: workers {len(c.workers)} store {c.store:.0f} pellets {len(c.pellets)} food {c.total_food} "
                  f"died {c.died} | tasks {tasks} | queen {c.queen['energy']:.0f}", flush=True)
