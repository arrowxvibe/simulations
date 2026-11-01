"""Index lookups into the FlyWire network: giant fibers, looming detectors, olfactory and mechanosensory neurons."""
import os
import numpy as np
import pandas as pd

import config
D = config.FLYWIRE_DIR
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells.npz")


def build():
    cls = pd.read_csv(f"{D}/classification.csv.gz")            # row order == network neuron index (see brain.py)
    idx = {r: i for i, r in enumerate(cls.root_id.values)}
    ct = pd.read_csv(f"{D}/consolidated_cell_types.csv.gz")
    vt = pd.read_csv(f"{D}/visual_neuron_types.csv.gz")

    def ids(df, col, names):
        m = df[df[col].astype(str).isin(names)].root_id.values
        return np.array([idx[r] for r in m if r in idx], np.int64)
    out = dict(gf=ids(ct, "primary_type", ["DNp01"]), lplc2=ids(vt, "type", ["LPLC2"]), lc4=ids(vt, "type", ["LC4"]),
               lc6=ids(vt, "type", ["LC6"]), lplc1=ids(vt, "type", ["LPLC1"]))
    for name, c in (("orn", "olfactory"), ("mech", "mechanosensory"), ("gust", "gustatory")):
        sub = cls[(cls.super_class == "sensory") & (cls["class"] == c)]
        out[name] = np.array([idx[r] for r in sub.root_id], np.int64)
        out[name + "_side"] = np.array([0 if s == "left" else 1 if s == "right" else 2 for s in sub.side], np.int64)
    np.savez(CACHE, **out)


def load():
    if not os.path.exists(CACHE):
        build()
    return dict(np.load(CACHE))


if __name__ == "__main__":
    c = load()
    print({k: (len(v)) for k, v in c.items()})
