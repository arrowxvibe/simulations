# Digital ant colony: ants grown from an egg, real fruit-fly brains

An island in an ocean with an ant colony (queen and workers), fruit flies steered by real connectomes, a small stone well,
and a boy with a jetpack you can play. Everything runs live and streams to a 3D viewer in the browser.

## What is real and what is a rule
- **Ant brains are grown, not typed in** (`grown_brain.py`): an egg gradient sets the brain domain, neural stem cells are picked by
  lateral inhibition, they divide into mushroom-body Kenyon cells, antennal-lobe inputs and an eye field, and the claws wire to the
  nearest inputs. Smell, vision and an interoceptive task circuit all come out of one genome. Sizes and the gene rules are simplified
  stand-ins borrowed from insect biology, not derived from real ant DNA.
- **Ants decide their own job** (`decide.py`, `colony_life.py`): forage, build, eat or rest, from their own drives (hunger, food store,
  nest condition, carrying food, fatigue), taught by consequences. Genomes evolve across colonies (`evolve.py`).
  Still rule-driven: the queen's movement and egg laying, and world physics (energy, spoilage, weather wearing the nest).
- **Flies run on real connectomes**: two females on FlyWire (139,255 neurons) and one male on MaleCNS (165,114 neurons, with the
  nerve cord). Vision, smell and air movement go into the real sensory neurons; the escape from the human is driven by the real
  giant-fibre neurons (DNp01). A linear steering readout is trained by imitation (`train_fly2.py`); it steers poorly in clutter.

## Run it
Needs Python 3 with numpy, a GPU build of PyTorch for the flies (the existing `C:\dev\fly-brain\.venv` works), and the data below.

```
python live_colony.py                                     # colony + viewer on http://localhost:8771
C:\dev\fly-brain\.venv\Scripts\python.exe fly_server2.py  # fly brains on port 8772 (optional: flies are scripted without it)
python evolve.py 12                                       # evolve the colony genome (writes best_genome.json, loaded by the colony)
```
Open http://localhost:8771. T = play the boy, WASD, Shift sprint, Space jump / hold for jetpack, V first person, `[` `]` cycle creature
feeds, B back. Click a portal card at the bottom to see through that creature.

## Data (not in this repo)
- FlyWire FAFB v783: `C:\Fruitfly\FlyWire Brain Dataset (FAFB v783)` and the brain code in `C:\dev\fly-brain` (`brain.py`).
- MaleCNS v0.9 (public Lee-lab bucket, files `malecns_09_meta.feather`, `malecns_09_simple_edgelist.feather`) in `C:\Fruitfly\MaleCNS`.
  `convert_malecns.py` (run in an environment with pyarrow) shrinks it to `malecns_net.npz`; the retina map comes from the lamina columns.

## Files
`ant_colony.py` ant brain operation, learning, world and vision | `grown_brain.py` development from the genome | `decide.py` task circuit |
`colony_life.py` colony | `evolve.py` colony evolution | `flyworld2.py`, `flybrainx.py`, `maleflybrain.py`, `flycells.py`, `train_fly2.py`,
`fly_server2.py` the flies | `live_colony.py` server | `live4.html` the viewer | `run_*.py` experiments.
