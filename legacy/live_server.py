"""Live ant colony: runs the simulation continuously and streams it to a browser.

Start:  python live_server.py      then open http://localhost:8770
The ants start with untrained brains, so you can watch them learn in real time.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import ant_colony as ac

PORT = 8770
FPS = 25

cond = threading.Condition()
state = {"seq": 0, "frame": None, "steps_per_frame": 2}


def simulate():
    rng = np.random.default_rng()
    ants = [ac.Ant(rng) for _ in range(ac.N_ANTS)]
    world = ac.World(np.random.default_rng(5))
    fed = [0] * len(ants)
    total, step, seq = 0, 0, 0
    events = []
    while True:
        t0 = time.time()
        for _ in range(state["steps_per_frame"]):
            for i, a in enumerate(ants):
                was, avd = a.carrying, a.brain.w_avd.sum()
                d = ac.ant_step(a, world, rng, True, True)
                if d:
                    total += 1; fed[i] += 1; events.append(f"Ant {i+1} delivered food to the nest")
                elif a.carrying and not was:
                    events.append(f"Ant {i+1} found food and learned its smell")
                elif a.brain.w_avd.sum() > avd:
                    events.append(f"Ant {i+1} hit poison and learned to avoid it")
            world.step_pheromone()
            step += 1
        seq += 1
        frame = {
            "t": step, "d": total,
            "a": [[round(float(a.pos[0]), 2), round(float(a.pos[1]), 2), round(float(a.heading), 2),
                   int(a.carrying), fed[i], round(float(a.brain.w_app.sum()), 1), round(float(a.brain.w_avd.sum()), 1)]
                  for i, a in enumerate(ants)],
            "s": [[k, round(float(p[0]), 1), round(float(p[1]), 1), int(am)] for k, p, am in world.src],
            "ev": events[-6:],
        }
        events = []
        if seq % 4 == 1:
            frame["p"] = np.round(np.minimum(world.pher, 9.9) * 10).astype(int).T.flatten().tolist()
        with cond:
            state["seq"], state["frame"] = seq, json.dumps(frame)
            cond.notify_all()
        time.sleep(max(0, 1 / FPS - (time.time() - t0)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = open("live.html", "rb").read()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif u.path == "/speed":
            v = int(parse_qs(u.query).get("v", ["2"])[0])
            state["steps_per_frame"] = max(1, min(8, v))
            self.send_response(204); self.end_headers()
        elif u.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            last = 0
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: state["seq"] != last, timeout=5)
                        last, data = state["seq"], state["frame"]
                    if data:
                        self.wfile.write(f"data: {data}\n\n".encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    threading.Thread(target=simulate, daemon=True).start()
    print(f"Live ant colony at http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
