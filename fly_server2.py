"""Live fruit flies driven by the real FlyWire connectome (139,255 neurons on your GPU).

Run: C:\\dev\\fly-brain\\.venv\\Scripts\\python.exe fly_server2.py
In:  colony stream (berries, ants) from :8771, and POSTs from the viewer: /human (position) and /scene (trees, rocks)
Out: SSE stream at http://localhost:8772/stream
Each fly sees the human, other flies, ants, trees, rocks and berries through the photoreceptors, smells berries,
feels air movement, and escapes through its real giant-fibre neurons (DNp01). Walking happens when it is on the ground.
"""
import json, os, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
from flybrainx import FlyBrainX
from flyworld2 import FlyWorld2, NS, ground, ARENA
HALF = ARENA / 2
from maleflybrain import MaleBrainX

NF = 2                                   # FlyWire (female) flies; one more fly runs on the MaleCNS (male) brain
N, FPS, PORT = NF + 1, 25, 8772
THR = np.array([0.0045] * NF + [0.003])  # giant-fibre firing thresholds, calibrated for each brain
CK = os.path.join(HERE, "fly_ckpt", os.environ.get("CKPT", "best_v2_D.pt"))
CKM = os.path.join(HERE, "fly_ckpt", "best_v2_male.pt")
cond, state = threading.Condition(), {"seq": 0, "frame": None}
world = {"berries": [], "ants": [], "trees": [], "rocks": [], "human": None, "hvel": (0.0, 0.0), "hseen": 0.0}


def watch_colony():
    while True:
        try:
            with urllib.request.urlopen("http://localhost:8771/stream", timeout=10) as r:
                for line in r:
                    if line.startswith(b"data:"):
                        f = json.loads(line[5:])
                        world["ants"] = [(a[2], a[3]) for a in f["a"]]
        except Exception:
            time.sleep(2)


def nearest(pts, p, k):
    if not len(pts): return [(300.0, 300.0)] * k
    a = np.array(pts, np.float32); d = np.linalg.norm(a - p, axis=1)
    out = [tuple(a[i]) for i in np.argsort(d)[:k]]
    return out + [(300.0, 300.0)] * (k - len(out))


def simulate():
    brain = FlyBrainX(gain=0.3, g_smell=0.1, g_mech=4.0, n_vpn=3000)
    theta = torch.load(CK).to(brain.dev) if os.path.exists(CK) else brain.theta.clone()
    th = theta[None].repeat(NF, 1); brain.reset_batch(NF)
    male = MaleBrainX(gain=0.3, g_smell=0.1, g_mech=10.0)
    mtheta = torch.load(CKM).to(male.dev) if os.path.exists(CKM) else male.theta.clone(); mth = mtheta[None]; male.reset_batch(1)
    w = FlyWorld2(N); w.external = True; w.respawn = False; obs = w.reset(); w.pos[:, 2] = 3.0
    seq, last_b, rest_until, total_esc = 0, None, np.zeros(N), 0
    frng = np.random.default_rng()

    def new_fruit(kind=0):    # the flies' own food: rotting fruit (kind 0) or a patch of nectar blossoms (kind 1), away from the ants' nest
        while True:
            p = frng.uniform(12, ARENA - 12, 2)
            if np.hypot(p[0] - ARENA / 2, p[1] - ARENA / 2) > 22: return [float(p[0]), float(p[1]), 12.0, kind]
    fruits = [new_fruit(0), new_fruit(1)]               # two pieces of fruit, and the water well (the third place flies go to)
    WELL = (ARENA / 2 + 20.0, ARENA / 2 - 16.0)
    mode, tm = ['fly'] * N, np.random.randint(100, 250, N)      # a real fly alternates flying, landing, walking and taking off
    MODES = {'fly': 0, 'descend': 1, 'walk': 2, 'takeoff': 3}
    while True:
        t0 = time.time()
        world["berries"] = [(q[0], q[1]) for q in fruits if q[2] > 0] + [WELL]
        if world["berries"] and world["berries"] != last_b:
            w.set_berries(world["berries"][:3]); last_b = list(world["berries"])
        if not world["berries"]: w.got[:] = True
        hum = world["human"]; hv = world["hvel"] if time.time() - world["hseen"] < 1.0 else (0.0, 0.0)
        for i in range(N):
            p = w.pos[i, :2]
            w.sol[i, 0] = [hum[0], hum[1], 3.0] if hum else [300, 300, 3.0]
            w.vel[i, 0] = [hv[0] / FPS, hv[1] / FPS]
            others = [tuple(w.pos[j, :2]) for j in range(N) if j != i]
            for k, q in enumerate(nearest(others, p, 2)): w.sol[i, 1 + k] = [q[0], q[1], w.pos[(i + 1 + k) % N, 2]]
            for k, q in enumerate(nearest(world["ants"], p, 2)): w.sol[i, 3 + k] = [q[0], q[1], 0.5]
            for k, q in enumerate(nearest(world["trees"], p, 3)): w.sol[i, 5 + k] = [q[0], q[1], 6.0]
            for k, q in enumerate(nearest(world["rocks"], p, 3)): w.sol[i, 8 + k] = [q[0], q[1], 0.5]
        sm, me = w.smell(), w.mech()
        lg1, gf1, l1, r1 = brain.act_x(obs[:NF], th, sm[:NF], me[:NF])
        lg2, gf2, l2, r2 = male.act_x(obs[NF:], mth, sm[NF:], me[NF:])
        logits, gf, l, r = torch.cat([lg1, lg2]), torch.cat([gf1, gf2]), torch.cat([l1, l2]), torch.cat([r1, r2])
        act = (logits > 0).cpu().numpy(); away = np.where((l > r).cpu().numpy(), 1.0, -1.0)
        resting = time.time() < rest_until; act[resting] = False
        grounded = (w.pos[:, 2] - ground(w.pos[:, 0], w.pos[:, 1])) < 0.35
        for i in range(N):
            if w.esc[i] > 0: mode[i] = 'fly'; tm[i] = np.random.randint(150, 300); continue     # escape overrides everything
            if mode[i] == 'fly':
                tm[i] -= 1
                if tm[i] <= 0: mode[i] = 'descend'
            elif mode[i] == 'descend':
                act[i, 3] = False                                   # stop climbing: sink to the ground
                if grounded[i]: mode[i] = 'walk'; tm[i] = np.random.randint(100, 230)
            elif mode[i] == 'walk':
                act[i, 3] = False; act[i, 0] = np.random.rand() < 0.88 or act[i, 0]       # walk on six legs, steering by the brain
                if not (act[i, 1] or act[i, 2]) and np.random.rand() < 0.08: act[i, 1 + np.random.randint(2)] = True
                tm[i] -= 1
                if tm[i] <= 0: mode[i] = 'takeoff'; tm[i] = 22
            elif mode[i] == 'takeoff':
                act[i, 3] = True; act[i, 0] = True; tm[i] -= 1
                if tm[i] <= 0: mode[i] = 'fly'; tm[i] = np.random.randint(150, 300)
        fire = gf.cpu().numpy() > THR
        before = w.escapes.sum(); obs, rew, landed = w.step(act, fire, away); total_esc += int(w.escapes.sum() - before)
        for i in np.where(landed)[0]:
            rest_until[i] = time.time() + 3 + np.random.rand() * 3
            dists = [np.hypot(q[0] - w.pos[i, 0], q[1] - w.pos[i, 1]) for q in fruits]; j = int(np.argmin(dists))
            if dists[j] < 3.0:                                  # it feeds; a piece of fruit runs out and a new one appears elsewhere
                fruits[j][2] -= 1.0
                if fruits[j][2] <= 0: fruits[j] = new_fruit(fruits[j][3])
        for i in range(N):                                  # berries are still in the colony: a rested fly may feed again
            if not resting[i] and time.time() > rest_until[i] + 4: w.got[i, :len(world["berries"][:3])] = False
        x = brain.X; seq += 1
        gfv = gf.cpu().numpy()
        fr = {"t": seq,
              "f": [[round(float(w.pos[i, 0]), 2), round(float(w.pos[i, 1]), 2), round(float(w.pos[i, 2]), 2), round(float(w.yaw[i]), 3),
                     int(resting[i]), int(w.esc[i] > 0), [int(a) for a in act[i]], round(float(gfv[i]), 4), MODES[mode[i]]] for i in range(N)],
              "brain": {"neurons": brain.n, "active": round(float((x > 0.05).float().mean()) * 100, 1), "gpu": torch.cuda.get_device_name(0), "escapes": total_esc}, "fr": [[round(q[0], 1), round(q[1], 1), round(q[2], 1), q[3]] for q in fruits], "well": [WELL[0], WELL[1]],
              "male": {"neurons": male.n, "active": round(float((male.X > 0.05).float().mean()) * 100, 1)}}
        with cond:
            state["seq"], state["frame"] = seq, json.dumps(fr); cond.notify_all()
        time.sleep(max(0, 1 / FPS - (time.time() - t0)))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _cors(self, code=204):
        self.send_response(code); self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Headers", "*"); self.end_headers()

    def do_OPTIONS(self): self._cors()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        try:
            d = json.loads(body)
            if self.path == "/human":
                x, y, now = d["x"] + HALF, d["z"] + HALF, time.time()
                old = world["human"]
                if old is not None and now - world["hseen"] > 0.01:
                    dt = now - world["hseen"]; world["hvel"] = ((x - old[0]) / dt, (y - old[1]) / dt)
                world["human"], world["hseen"] = (x, y), now
            elif self.path == "/scene":
                world["trees"] = [(a + HALF, b + HALF) for a, b in d["trees"]]; world["rocks"] = [(a + HALF, b + HALF) for a, b in d["rocks"]]
        except Exception:
            pass
        self._cors()

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        for k, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"), ("Access-Control-Allow-Origin", "*")): self.send_header(k, v)
        self.end_headers(); last = 0
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: state["seq"] != last, timeout=5); last, data = state["seq"], state["frame"]
                if data: self.wfile.write(f"data: {data}\n\n".encode()); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


if __name__ == "__main__":
    threading.Thread(target=watch_colony, daemon=True).start(); threading.Thread(target=simulate, daemon=True).start()
    print(f"Fly brain stream on http://localhost:{PORT}/stream  (checkpoint {os.path.basename(CK)}: {'found' if os.path.exists(CK) else 'MISSING'})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
