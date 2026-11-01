"""Shrink the MaleCNS (v0.9) connectome to a GPU-sized sparse network.

Run (needs pandas and pyarrow):  python convert_malecns.py
Reads the two .feather files in MALECNS_DIR (see config.py; read only) and writes malecns_net.npz next to this script.
Keeps connections with >= MIN_COUNT synapses; weight = synapse count with the sign of the predicted transmitter.
"""
import os, sys, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
D = config.MALECNS_DIR
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "malecns_net.npz")
MIN_COUNT = 5
SIGN = {"acetylcholine": 1.0, "glutamate": -1.0, "gaba": -1.0, "histamine": -1.0, "dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0}

t0 = time.time()
m = feather.read_table(f"{D}/malecns_09_meta.feather").to_pandas()
ids = pd.Index(m.malecns_09_id.astype(str).values)
n = len(ids)
sign = m.neurotransmitter_predicted.map(SIGN).fillna(1.0).values.astype(np.float32)

pres, posts, ws = [], [], []
f = pa.ipc.open_file(pa.memory_map(f"{D}/malecns_09_simple_edgelist.feather"))
kept = total = 0
for b in range(f.num_record_batches):
    t = f.get_batch(b)
    cnt = t.column("count").to_numpy(zero_copy_only=False)
    total += len(cnt)
    sel = cnt >= MIN_COUNT
    if not sel.any():
        continue
    pre = ids.get_indexer(t.column("pre").to_pandas().astype(str).values[sel])
    post = ids.get_indexer(t.column("post").to_pandas().astype(str).values[sel])
    ok = (pre >= 0) & (post >= 0)
    pres.append(pre[ok].astype(np.int32)); posts.append(post[ok].astype(np.int32))
    ws.append((cnt[sel][ok] * sign[pre[ok]]).astype(np.float32)); kept += int(ok.sum())
    if b % 200 == 0:
        print(f"batch {b}/{f.num_record_batches}  kept {kept:,} of {total:,}  ({time.time()-t0:.0f}s)", flush=True)
pre, post, w = np.concatenate(pres), np.concatenate(posts), np.concatenate(ws)


def where(mask):
    return np.where(mask.values)[0].astype(np.int64)


ct = m.cell_type.astype(str)
cc = m.cell_class.astype(str)
sc = m.super_class.astype(str)
side = m.side.astype(str).map({"left": 0, "right": 1}).fillna(2).values.astype(np.int8)
hex1 = pd.to_numeric(m.optic_lobe_hex_1, errors="coerce").fillna(-999).values.astype(np.float32)
hex2 = pd.to_numeric(m.optic_lobe_hex_2, errors="coerce").fillna(-999).values.astype(np.float32)
motor_kind = m.cell_class.astype(str).where(sc == "motor", "")
pr = where(cc == "photoreceptor_neuron")
np.savez_compressed(OUT, n=n, pre=pre, post=post, w=w, side=side, hex1=hex1, hex2=hex2,
                    pr=pr, desc=where(sc == "descending"), motor=where(sc == "motor"),
                    motor_kind=motor_kind.values[where(sc == "motor")].astype(str),
                    gf=where(ct.str.contains("DNp01")), lplc2=where(ct.str.contains("LPLC2")), lc4=where(ct == "LC4"),
                    orn=where(cc == "olfactory_receptor_neuron"), mech=where(cc.isin(["chordotonal_organ_neuron", "bristle_neuron", "hair_plate_neuron"])),
                    kenyon=where(cc == "kenyon_cell"))
print(f"DONE: {n:,} neurons, {len(w):,} connections kept of {total:,} (>= {MIN_COUNT} synapses), photoreceptors {len(pr)}, "
      f"with eye-column coordinates: {(hex1[pr] > -900).sum()}  ({time.time()-t0:.0f}s)", flush=True)
