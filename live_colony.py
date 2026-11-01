"""Live ant colony with queen, workers, building and reproduction: runs the simulation continuously and streams it to a browser.

Start:  python live_server.py      then open http://localhost:8771
The ants start with untrained brains, so you can watch them learn in real time.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from colony_life import Colony
from ant_colony import ARENA

PORT = 8771
FPS = 25

cond = threading.Condition()
state = {"seq": 0, "frame": None, "steps_per_frame": 1}


def load_genome():
    """Use the genome that evolution found, if there is one."""
    try:
        g = json.load(open("best_genome.json"))
        return {k: v for k, v in g.items() if k not in ("fitness", "generation", "history")}
    except Exception:
        return None


def simulate():
    colony = Colony(genome=load_genome()); state["colony"] = colony
    seq = 0
    while True:
        t0 = time.time()
        if time.time() - state.get("hseen", 0) > 2: colony.world.human = None     # the viewer stopped reporting the human
        for _ in range(state["steps_per_frame"]):
            colony.step()
        if colony.queen["energy"] <= 0 and not colony.workers and not colony.brood:       # the colony died out: a new queen founds another
            colony = Colony(genome=load_genome()); state["colony"] = colony; colony.log("The old colony died out. A new queen founded a colony.")
        seq += 1
        frame = json.dumps(colony.frame(seq % 4 == 1))
        with cond:
            state["seq"], state["frame"] = seq, frame
            cond.notify_all()
        time.sleep(max(0, 1 / FPS - (time.time() - t0)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Headers", "*"); self.end_headers()

    def do_POST(self):
        u = urlparse(self.path); body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        try:
            d = json.loads(body); c = state.get("colony")
            if u.path == "/human" and c is not None:       # ants only meet him on the ground: a hovering human cannot step on them
                c.world.human = None if d.get("y", 0) > 3.5 else np.array([d["x"] + ARENA / 2, d["z"] + ARENA / 2]); state["hseen"] = time.time()
        except Exception:
            pass
        self.send_response(204); self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "live4.html"), "rb").read()
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
