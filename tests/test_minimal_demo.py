from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples" / "run_minimal_demo.py"
SPEC = importlib.util.spec_from_file_location("sr_mggs_minimal_demo", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MinimalDemoTest(unittest.TestCase):
    def test_expected_source_is_accepted(self) -> None:
        input_path = ROOT / "examples" / "minimal_demo" / "input.json"
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result = MODULE.run_demo(payload)
        self.assertEqual(result["predicted_source"], "source_alpha")
        self.assertEqual(result["decision"], "Accept")
        self.assertEqual(result["templates_per_source"]["source_alpha"], 5)
        self.assertGreater(result["authentication_score"], result["calibrated_threshold"])


if __name__ == "__main__":
    unittest.main()

