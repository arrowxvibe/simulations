"""Quick check that a fresh checkout works (no GPU or datasets needed): python smoke_test.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Smoke(unittest.TestCase):
    def test_brain_grows_from_the_genome(self):
        import numpy as np, grown_brain as gb
        info = gb.develop(gb.GENOME, np.random.default_rng(0))
        self.assertGreater(info["n_kc"], 0); self.assertGreater(info["n_vis"], 0)
        blind = gb.develop({**gb.GENOME, "eye_ap_max": 0.0}, np.random.default_rng(0))
        self.assertEqual(blind["n_vis"], 0)                         # no eye field, no vision

    def test_colony_runs_and_ants_choose_jobs(self):
        import colony_life as cl
        c = cl.Colony(seed=1)
        for _ in range(600): c.step()
        self.assertEqual(len(c.queen["pos"]), 2); frame = c.frame(True)
        self.assertIn("a", frame); self.assertEqual(len(frame["p"]), 120 * 120)

    def test_viewer_and_servers_are_present(self):
        here = os.path.dirname(os.path.abspath(__file__))
        for f in ("live4.html", "live_colony.py", "fly_server2.py", "config.py"):
            self.assertTrue(os.path.exists(os.path.join(here, f)), f)

    def test_no_machine_specific_paths(self):
        here = os.path.dirname(os.path.abspath(__file__))
        for f in os.listdir(here):
            if f.endswith(".py") and f not in ("config.py", "smoke_test.py"):
                with open(os.path.join(here, f), encoding="utf8") as fh:
                    self.assertNotIn("Fruitfly", fh.read(), f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
