"""One command: python start.py   then open http://localhost:8771

Starts the colony and the 3D viewer. The fruit flies need PyTorch with CUDA plus the FlyWire and MaleCNS data (see README);
if anything is missing they are skipped and the viewer shows scripted flies instead. Use --no-flies to skip them on purpose.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config


def flies_ready():
    try:
        import torch
        if not torch.cuda.is_available(): return "PyTorch has no CUDA GPU"
    except ImportError:
        return "PyTorch is not installed"
    for f in ("classification.csv.gz", "connections_princeton.csv.gz", "coordinates.csv.gz"):
        if not os.path.exists(os.path.join(config.FLYWIRE_DIR, f)): return f"FlyWire data not found in {config.FLYWIRE_DIR}"
    if not os.path.exists(os.path.join(config.FLY_BRAIN_DIR, "brain.py")): return f"brain.py not found in {config.FLY_BRAIN_DIR}"
    if not os.path.exists(os.path.join(HERE, "malecns_net.npz")): return "MaleCNS not converted yet (python download_data.py, then python convert_malecns.py)"
    return None


def main():
    procs = [subprocess.Popen([sys.executable, "live_colony.py"], cwd=HERE)]
    why = "skipped by --no-flies" if "--no-flies" in sys.argv else flies_ready()
    if why is None:
        time.sleep(6); procs.append(subprocess.Popen([sys.executable, "fly_server2.py"], cwd=HERE))
    else:
        print(f"Flies are scripted: {why}")
    print("Open http://localhost:8771   (Ctrl+C to stop)")
    try:
        for p in procs: p.wait()
    except KeyboardInterrupt:
        for p in procs: p.terminate()


if __name__ == "__main__":
    main()
