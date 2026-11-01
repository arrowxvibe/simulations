"""Fetch the public MaleCNS files and check the FlyWire files.

  python download_data.py            download MaleCNS (about 3.3 GB, resumes if interrupted) and report what is missing
  python download_data.py --check    only report

MaleCNS v0.9 (Janelia / Google, CC-BY) comes from the public Lee-lab bucket. FlyWire (FAFB v783) is NOT downloaded here:
get it from FlyWire and follow its terms of use, then put the five files listed below in FLYWIRE_DIR (see config.py).
"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

BASE = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/malecns_09"
MALECNS = [("malecns_09_meta.feather", 16593866), ("malecns_09_simple_edgelist.feather", 3404193362)]
FLYWIRE = ["classification.csv.gz", "connections_princeton.csv.gz", "coordinates.csv.gz", "consolidated_cell_types.csv.gz", "visual_neuron_types.csv.gz"]   # about 73 MB of the 17 GB release


def download(name, size):
    os.makedirs(config.MALECNS_DIR, exist_ok=True)
    dest = os.path.join(config.MALECNS_DIR, name)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have == size:
        print(f"  {name}: already complete"); return
    req = urllib.request.Request(f"{BASE}/{name}", headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r, open(dest, "ab" if have else "wb") as f:
        done = have
        while True:
            chunk = r.read(1 << 20)
            if not chunk: break
            f.write(chunk); done += len(chunk)
            print(f"\r  {name}: {done / 1e6:8.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    if os.path.getsize(dest) != size:
        raise SystemExit(f"{name}: size is {os.path.getsize(dest)}, expected {size}. Run again to resume.")


def main():
    if "--check" not in sys.argv:
        print(f"Downloading MaleCNS into {config.MALECNS_DIR}")
        for name, size in MALECNS: download(name, size)
    print("\nMaleCNS:", "OK" if all(os.path.exists(os.path.join(config.MALECNS_DIR, n)) for n, _ in MALECNS) else "files missing, run without --check")
    missing = [n for n in FLYWIRE if not os.path.exists(os.path.join(config.FLYWIRE_DIR, n))]
    print("FlyWire:", "OK" if not missing else f"missing in {config.FLYWIRE_DIR}: {', '.join(missing)}")
    print("brain.py:", "OK" if os.path.exists(os.path.join(config.FLY_BRAIN_DIR, "brain.py")) else f"missing in {config.FLY_BRAIN_DIR} (see README)")
    print("\nThe ants and the viewer need none of this. The flies need all three.")


if __name__ == "__main__":
    main()
