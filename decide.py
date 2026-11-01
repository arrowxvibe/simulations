"""How an ant decides what to do: a small circuit grown from the genome.

Inner state (the ant's drives) is read by a group of interoceptive Kenyon-like cells: each cell samples a few
drive units at random (wiring grown by the genome), only the most active ~10% fire. Four output neurons
(forage, build, eat, rest) read them with plastic weights. The choice is sampled from the outputs.
Learning: when a task ends, its consequence is a reward or punishment that changes the weights of the
output neuron that was chosen, for the cells that were active at the decision (a dopamine-like prediction error).
There is also an innate pathway (genome: prior and bias) that evolution tunes. Nothing here says what a good job is.
"""
import numpy as np

TASKS = ["forage", "build", "eat", "rest"]
N_DRIVE, BINS = 5, 3          # hunger, food store (felt), nest needs work, carrying food, fatigue


def pop_code(d):
    d = np.clip(d, 0, 1)
    return np.concatenate([np.maximum(0, 1 - 2 * d), 1 - np.abs(2 * d - 1), np.maximum(0, 2 * d - 1)])   # low, mid, high units


class TaskCircuit:
    def __init__(self, rng, g):
        n, k = int(g["state_kc"]), int(g["state_claws"])
        self.idx = rng.integers(0, N_DRIVE * BINS, (n, k))
        self.n_top = max(1, int(n * 0.1))
        self.w = np.zeros((4, n)); self.base = np.zeros(4)
        self.prior = np.array(g["task_prior"], float).reshape(4, N_DRIVE); self.bias = np.array(g["task_bias"], float)
        self.T, self.lr = max(0.2, float(g["task_T"])), float(g["task_lr"])
        self.kcn = np.zeros(n); self.probs = np.full(4, 0.25)

    def decide(self, drives, rng):
        d = np.clip(np.asarray(drives, float), 0, 1)
        a = pop_code(d)[self.idx].sum(1)
        thr = np.partition(a, -self.n_top)[-self.n_top]
        kc = np.maximum(a - thr, 0.0)
        self.kcn = kc / (np.sqrt((kc ** 2).sum()) + 1e-6)
        logits = self.prior @ d + self.bias + self.w @ self.kcn
        z = (logits - logits.max()) / self.T
        self.probs = np.exp(z) / np.exp(z).sum()
        return int(rng.choice(4, p=self.probs))

    def learn(self, task, reward):
        delta = reward - self.base[task]
        self.base[task] += 0.1 * delta
        self.w[task] = np.clip(self.w[task] + self.lr * delta * self.kcn, -6, 6)
