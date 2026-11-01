"""Where the data and the outside code live. Every location can be overridden with an environment variable,
so nothing in this project is tied to one machine.

  FLYWIRE_DIR    folder with the FlyWire FAFB v783 tables (classification.csv.gz, connections_princeton.csv.gz, ...)
  MALECNS_DIR    folder with malecns_09_meta.feather and malecns_09_simple_edgelist.feather (see convert_malecns.py)
  FLY_BRAIN_DIR  folder containing brain.py (the FlyWire connectome as a GPU network; BatchedBrain)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _first_existing(*candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return candidates[0]            # nothing found: the first (repo-relative) choice is what error messages will name


FLYWIRE_DIR = os.environ.get("FLYWIRE_DIR") or _first_existing(os.path.join(HERE, "data", "flywire"), r"C:\Fruitfly\FlyWire Brain Dataset (FAFB v783)")
MALECNS_DIR = os.environ.get("MALECNS_DIR") or _first_existing(os.path.join(HERE, "data", "malecns"), r"C:\Fruitfly\MaleCNS")
FLY_BRAIN_DIR = os.environ.get("FLY_BRAIN_DIR") or _first_existing(os.path.join(HERE, "vendor", "fly-brain"), r"C:\dev\fly-brain")


def add_flybrain_to_path():
    """Make `from brain import BatchedBrain` work."""
    if FLY_BRAIN_DIR not in sys.path:
        sys.path.insert(0, FLY_BRAIN_DIR)
    if not os.path.exists(os.path.join(FLY_BRAIN_DIR, "brain.py")):
        raise FileNotFoundError(f"brain.py not found in {FLY_BRAIN_DIR}. Set FLY_BRAIN_DIR to the folder that contains it (see README).")
