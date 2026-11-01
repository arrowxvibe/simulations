# Digital ant colony: ants grown from an egg, real fruit-fly brains

An island in an ocean with an ant colony (a queen and workers), fruit flies steered by real connectomes, a small stone well,
and a boy with a jetpack you can play. Everything runs live and streams to a 3D viewer in your browser.

## What is real and what is a rule
- **Ant brains are grown, not typed in** (`grown_brain.py`): an egg gradient sets the brain domain, neural stem cells are picked by
  lateral inhibition, they divide into mushroom-body Kenyon cells, antennal-lobe inputs and an eye field, and the claws wire to the
  nearest inputs. Smell, vision and a task circuit all come out of one genome. Sizes and gene rules are simplified stand-ins
  borrowed from insect biology, **not derived from real ant DNA**.
- **Ants decide their own job** (`decide.py`, `colony_life.py`): forage, build, eat or rest, from their own drives (hunger, food store,
  nest condition, carrying food, fatigue), taught by consequences. Genomes evolve across colonies (`evolve.py`).
  Still rule-driven: the queen's movement and egg laying, and world physics (energy, spoilage, weather wearing the nest).
- **Flies run on real connectomes**: two females on FlyWire (139,255 neurons) and one male on MaleCNS (165,114 neurons, with the nerve
  cord). Vision, smell and air movement go into the real sensory neurons, and the escape from the human is driven by the real
  giant-fibre neurons (DNp01). A linear steering readout is trained by imitation (`train_fly2.py`); it steers poorly in clutter,
  so the flies flee well but rarely find food.
- Known weak spots: the ant colony's workers still die young in this world, ant vision has a safety/food trade-off, and the fly brains
  forage badly. The `run_*.py` scripts reproduce the experiments behind these statements.

## Quick start (the ants and the viewer need no data and no GPU)
```
python -m venv .venv            # then activate it
pip install -r requirements.txt # for torch choose the CUDA build from pytorch.org if you want the flies
python smoke_test.py            # 4 quick checks, about 2 seconds
python start.py                 # colony + viewer; open http://localhost:8771
```
Controls: **T** play the boy, **WASD** move, **Shift** sprint, **Space** jump (hold in the air for the jetpack), **V** first person,
**[** and **]** cycle creature feeds, **B** back. Click a portal card at the bottom to see through that creature's eyes; the small
arrow on a card opens a large live window. Sound switches on at your first key press.

## The flies need data (about 3.4 GB) and a GPU
| What | Needed for | Size | How to get it |
|---|---|---|---|
| MaleCNS v0.9 (`malecns_09_meta.feather`, `malecns_09_simple_edgelist.feather`) | the male fly | 3.3 GB | `python download_data.py`, then `python convert_malecns.py` (makes a 33 MB `malecns_net.npz`) |
| FlyWire FAFB v783: `classification`, `connections_princeton`, `coordinates`, `consolidated_cell_types`, `visual_neuron_types` (`.csv.gz`) | the two female flies | 73 MB | download from FlyWire and follow its terms of use. The full release is 17 GB; only these five files are read |
| `brain.py` (the FlyWire connectome as a GPU network, class `BatchedBrain`) | the two female flies | small | not part of this repository yet: put it in `vendor/fly-brain/` or point `FLY_BRAIN_DIR` at it |

`python download_data.py --check` tells you what is missing. Locations can be changed with the environment variables
`FLYWIRE_DIR`, `MALECNS_DIR` and `FLY_BRAIN_DIR` (see `config.py`); by default the code looks in `data/` and `vendor/` next to it.
Without the flies' data, `start.py` runs everything else and shows scripted flies.

## Train and evolve
```
python evolve.py 12                   # colonies compete; writes best_genome.json, which the colony loads at start
python train_fly2.py                  # fit the FlyWire flies' steering readout (BRAIN=male for the MaleCNS fly); needs the flies' data and a GPU
```

## Files
`ant_colony.py` ant brain operation, learning, world and vision | `grown_brain.py` development from the genome | `decide.py` task circuit |
`colony_life.py` colony | `evolve.py` colony evolution | `flyworld2.py`, `flybrainx.py`, `maleflybrain.py`, `flycells.py`, `train_fly2.py`,
`fly_server2.py` the flies | `live_colony.py` server | `live4.html` the 3D viewer | `config.py` where data lives | `run_*.py` experiments |
`legacy/` early versions, kept for the record and not maintained.

## Credits and data terms
FlyWire (FAFB v783) and MaleCNS (Janelia and Google Research, v0.9, released under CC-BY) are the work of their teams; cite them when you use
their data, and check each dataset's own terms. This repository does not contain either dataset.
