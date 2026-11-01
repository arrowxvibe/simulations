"""Streams flies steered by the real FlyWire connectome (GPU) to the viewer.

Run with the fly-brain environment:  C:\\dev\\fly-brain\\.venv\\Scripts\\python.exe fly_server.py
Reads the colony's berries from http://localhost:8771/stream, publishes flies at http://localhost:8772/stream.
"""
import json, os, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, r"C:\dev\fly-brain")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from brain import BatchedBrain
from flyworld import FlyWorld

N, FPS, PORT = 5, 25, 8772
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly_ckpt", "best_bc.pt")
cond, state, berries = threading.Condition(), {"seq": 0, "frame": None}, {"pts": []}


def watch_colony():
    """Follow the colony stream to learn where the real berries are."""
    while True:
        try:
            with urllib.request.urlopen("http://localhost:8771/stream", timeout=10) as r:
                for line in r:
                    if line.startswith(b"data:"):
                        f = json.loads(line[5:])
                        berries["pts"] = [(s[1], s[2]) for s in f["s"] if s[0] == "food" and s[3] > 0]
        except Exception:
            time.sleep(2)


def simulate():
    brain = BatchedBrain(gain=float(os.environ.get('GAIN', 0.3)))
    theta = torch.load(CK).to(brain.dev) if os.path.exists(CK) else brain.theta.clone()
    thetas = theta[None].repeat(N, 1)
    brain.reset_batch(N)
    w = FlyWorld(N); w.respawn = False; obs = w.reset()
    seq, last_set, landed_until = 0, None, np.zeros(N)
    while True:
        t0 = time.time()
        pts = berries["pts"][:3]
        if pts and pts != last_set:
            w.set_berries(pts); last_set = list(pts)
        if not pts:
            w.got[:] = True
        logits = brain.act_batch(obs, thetas, steps=2)
        act = (logits > 0).cpu().numpy()
        near_before, _ = w.nearest()
        landed_now = near_before < 0.9
        # a fly that has landed rests a few seconds, then lifts off and looks for the next berry
        resting = time.time() < landed_until
        act[resting] = False
        obs, r, landed = w.step(act)
        for i in np.where(landed)[0]:
            landed_until[i] = time.time() + 3 + np.random.rand() * 3
        for i in np.where(~resting & (time.time() > landed_until + 3) & w.got.all(1))[0]:
            w.got[i] = False        # berries regrow in the colony; keep foraging
        x = brain.activity()
        seq += 1
        fr = {"t": seq,
              "f": [[round(float(w.pos[i, 0]), 2), round(float(w.pos[i, 1]), 2), round(float(w.pos[i, 2]), 2),
                     round(float(w.yaw[i]), 3), int(resting[i]), [int(a) for a in act[i]]] for i in range(N)],
              "brain": {"neurons": brain.n, "active": round(float((x > 0.05).float().mean()) * 100, 1), "on_gpu": torch.cuda.get_device_name(0)}}
        with cond:
            state["seq"], state["frame"] = seq, json.dumps(fr)
            cond.notify_all()
        time.sleep(max(0, 1 / FPS - (time.time() - t0)))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        for k, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"), ("Access-Control-Allow-Origin", "*")):
            self.send_header(k, v)
        self.end_headers(); last = 0
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: state["seq"] != last, timeout=5)
                    last, data = state["seq"], state["frame"]
                if data:
                    self.wfile.write(f"data: {data}\n\n".encode()); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


if __name__ == "__main__":
    threading.Thread(target=watch_colony, daemon=True).start()
    threading.Thread(target=simulate, daemon=True).start()
    print(f"Fly brain stream at http://localhost:{PORT}/stream (checkpoint: {'yes' if os.path.exists(CK) else 'none yet'})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
