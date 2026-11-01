"""Egg -> brain. The ant brain is grown from a small genome instead of being typed in.

Stages (all rules are simplified stand-ins for fly/insect developmental biology):
  1. EGG       a maternal "bicoid-like" gradient from the head end switches on a head gene (otd-like)
               through a positive feedback loop; where it stays on is the brain domain.
  2. SHEET     the brain domain is a sheet of cells. Lateral inhibition (Notch-like) picks well-spaced
               neuroblasts (neural stem cells). Position on the sheet decides their lineage type:
               posterior-dorsal -> mushroom body (memory), anterior-ventral -> antennal lobe (smell).
  3. BIRTH     each mushroom-body neuroblast divides on a timer; every division adds a Kenyon cell.
               Birth order sets the Kenyon cell subtype (early / middle / late), which sets its claw count.
               Antennal-lobe lineages produce projection neurons, each type targeting one odor receptor.
  4. WIRING    each Kenyon cell grows claws at random angles from its cell body and each claw
               attaches to the nearest projection-neuron bouton in the calyx (the input zone).
Everything here is driven by GENOME numbers. Individual ants get different brains from the same
genome because development is noisy.
"""
import numpy as np

# Innate drives of a worker (rows: forage, build, eat, rest; columns: hunger, food store felt, nest needs work, carrying food, fatigue).
# Every animal is born with some: hungry -> eat, tired -> rest, nest worn -> build when there is food, carrying food -> keep going home.
# Learning and evolution change them; without any instincts the first workers starve before they can learn to eat.
INSTINCTS = [0.5, -1.0, 0.0, 1.5, -1.0,    -1.0, 0.8, 1.5, -2.0, -0.8,    4.0, 0.3, 0.0, -3.0, 0.0,    -1.5, 0.0, 0.0, -2.0, 3.0]

GENOME = dict(
    seed=7,                 # fixes the species' receptor "address" code (same for every individual)
    n_or=40,                # odor receptor genes (real ants have ~350)
    bcd_lambda=0.20,        # how far the maternal gradient reaches from the head end
    otd_theta=0.30,         # threshold for switching on the head gene
    otd_feedback=0.20,      # head gene reinforces itself
    otd_gain=12.0,
    cells_per_len=50,       # cell density along the body axis
    dv_rows=20,
    nb_spacing=2.5,         # lateral inhibition range (cells)
    mb_ap_min=0.60, mb_dv_min=0.75,    # mushroom-body territory (fraction of the sheet)
    al_ap_max=0.40, al_dv_max=0.50,    # antennal-lobe territory
    mb_divisions=500,       # divisions before the neuroblast stops (timer gene)
    pn_types_per_lineage=6,
    claws=(4, 5, 7),        # claws per Kenyon cell for early / middle / late born
    claw_reach=0.30,
    lineage_spread=0.20,
    eye_ap_max=0.40, eye_dv_min=0.50,   # eye territory on the sheet (anterior, dorsal)
    eye_cells_per_bin=6,    # eye cells needed per visual direction bin
    max_bins=16,
    visual_kc_fraction=0.5, # share of late-born Kenyon cells that take visual input instead of smell
    # task-choice circuit (see decide.py). The innate prior/bias start at zero: evolution and learning fill them in.
    state_kc=300, state_claws=3, task_prior=INSTINCTS, task_bias=[0.0] * 4, task_T=1.0, task_lr=0.15,
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def develop(genome, rng):
    g = genome
    info = {}

    # 1. egg: maternal gradient -> head gene domain
    x = np.linspace(0, 1, 400)
    bcd = np.exp(-x / g["bcd_lambda"])
    otd = np.zeros_like(x)
    for _ in range(80):
        otd = sigmoid(g["otd_gain"] * (bcd + g["otd_feedback"] * otd - g["otd_theta"]))
    on = np.where(otd > 0.5)[0]
    brain_len = x[on.max()] if len(on) else 0.0
    info.update(x=x, bcd=bcd, otd=otd, brain_len=brain_len)

    # 2. sheet + lateral inhibition
    cols = max(1, int(round(brain_len * g["cells_per_len"])))
    rows = g["dv_rows"]
    cells = [(i, j) for i in range(cols) for j in range(rows)]
    rng.shuffle(cells)
    nbs = []
    for (i, j) in cells:
        if all((i - a) ** 2 + (j - b) ** 2 >= g["nb_spacing"] ** 2 for a, b in nbs):
            nbs.append((i, j))
    mb_nbs, al_nbs = [], []
    for (i, j) in nbs:
        fx, fy = (i + .5) / cols, (j + .5) / rows
        if fx > g["mb_ap_min"] and fy > g["mb_dv_min"]:
            mb_nbs.append((i, j))
        elif fx < g["al_ap_max"] and fy < g["al_dv_max"]:
            al_nbs.append((i, j))
    # the eye field: more eye cells -> finer vision (more direction bins). A mutant with no eye field is blind.
    eye_cells = sum(1 for (i, j) in cells if (i + .5) / cols < g["eye_ap_max"] and (j + .5) / rows > g["eye_dv_min"])
    n_bins = int(min(g["max_bins"], round(eye_cells / g["eye_cells_per_bin"])))
    n_bins = n_bins if n_bins >= 2 else 0
    n_vis = 4 * n_bins            # 4 visual channels per direction: food, poison, big moving animal, nest landmark
    info.update(cols=cols, rows=rows, nbs=nbs, mb_nbs=mb_nbs, al_nbs=al_nbs, eye_cells=eye_cells, n_bins=n_bins, n_vis=n_vis)

    # 3a. antennal lobe: lineages -> projection neurons -> glomeruli
    n_pn_types = len(al_nbs) * g["pn_types_per_lineage"]
    covered = np.zeros(g["n_or"], bool)
    covered[rng.permutation(g["n_or"])[:min(g["n_or"], n_pn_types)]] = True
    cue_rng = np.random.default_rng(g["seed"])
    bouton = cue_rng.random((g["n_or"], 2)) + rng.normal(0, 0.02, (g["n_or"], 2))

    # visual boutons sit in the calyx in a retinotopic map: direction left to right, channel top to bottom
    vb = np.array([[((b + .5) / n_bins) * 0.9 + 0.05, ((c + .5) / 4) * 0.9 + 0.05] for c in range(4) for b in range(n_bins)]).reshape(-1, 2) + rng.normal(0, 0.015, (n_vis, 2))

    # 3b. mushroom body: neuroblast timer -> Kenyon cells with a birth order
    soma, subtype = [], []
    for _ in mb_nbs:
        centre = rng.random(2)
        for k in range(g["mb_divisions"]):
            soma.append(centre + rng.normal(0, g["lineage_spread"], 2))
            subtype.append(min(2, 3 * k // g["mb_divisions"]))
    soma, subtype = np.array(soma).reshape(-1, 2), np.array(subtype, int)
    n_kc = len(soma)

    # late-born Kenyon cells: a share of them are "visual" (they sample the retinotopic map instead of smell)
    is_vis = np.zeros(n_kc, bool)
    if n_vis and n_kc:
        is_vis = (subtype == 2) & (rng.random(n_kc) < g["visual_kc_fraction"])
        soma[is_vis] = rng.random((int(is_vis.sum()), 2))
    # 4. wiring (vectorised): every claw tip attaches to the nearest bouton of its kind
    maxc = max(g["claws"])
    idx, msk = np.zeros((max(n_kc, 1), maxc), int), np.zeros((max(n_kc, 1), maxc))
    allb = np.concatenate([bouton, vb]) if n_vis else bouton

    def wire(sel, cov):
        kcs = np.where(sel)[0]
        if not len(kcs) or not len(cov): return
        per = np.array(g["claws"])[subtype[kcs]]
        owner = np.repeat(np.arange(len(kcs)), per)
        slot = np.arange(len(owner)) - np.repeat(np.cumsum(per) - per, per)
        ang = rng.uniform(0, 2 * np.pi, len(owner)); r = g["claw_reach"] * np.sqrt(rng.random(len(owner)))
        tip = soma[kcs][owner] + np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        nearest = np.empty(len(owner), int)
        for c in range(0, len(owner), 4000):
            d2 = ((tip[c:c + 4000, None, :] - allb[cov][None]) ** 2).sum(-1)
            nearest[c:c + 4000] = cov[d2.argmin(1)]
        idx[kcs[owner], slot], msk[kcs[owner], slot] = nearest, 1.0
    if n_kc:
        wire(~is_vis, np.where(covered)[0])
        wire(is_vis, g["n_or"] + np.arange(n_vis))
    info.update(n_vis_kc=int(is_vis.sum()), n_kc=n_kc, covered=covered, idx=idx, msk=msk, soma=soma, subtype=subtype, bouton=bouton)
    return info


def summary(info):
    return (f"brain domain {info['brain_len']:.2f} of body | {len(info['nbs'])} neuroblasts "
            f"({len(info['mb_nbs'])} mushroom body, {len(info['al_nbs'])} antennal lobe) | "
            f"{info['n_kc']} Kenyon cells ({info['n_vis_kc']} visual) | {int(info['covered'].sum())}/{len(info['covered'])} glomeruli | "
            f"eye field {info['eye_cells']} cells -> {info['n_bins']} view directions")


def brain_maker(genome):
    """Returns a function rng -> AntBrain grown from this genome (for ant_colony)."""
    import ant_colony as ac

    def make(rng):
        info = develop(genome, rng)
        return ac.AntBrain(rng, idx=info["idx"], msk=info["msk"], n_vis=info["n_vis"], n_bins=info["n_bins"])
    return make


ANT_SCALE = dict(n_or=400, mb_divisions=5000, pn_types_per_lineage=100)   # ~400 glomeruli, ~20k Kenyon cells


if __name__ == "__main__":
    import time
    for name, gen in (("fly-scale", GENOME), ("ant-scale", {**GENOME, **ANT_SCALE})):
        for s in range(3):
            t = time.time()
            print(name, summary(develop(gen, np.random.default_rng(s))), f"({time.time() - t:.1f}s)")
