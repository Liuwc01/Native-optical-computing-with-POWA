import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

from examples.sost_demo import main, run_demo


class SOSTDemoTests(unittest.TestCase):
    def test_demo_is_deterministic_and_exercises_both_paths(self):
        first = run_demo()
        second = run_demo()
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            {
                "continuous_checksum": 3.78125,
                "continuous_output_shape": [2, 8, 3, 3],
                "device": "cpu",
                "gradient_norm": 0.42765,
                "input_shape": [2, 1, 4, 4],
                "max_abs_rounding_difference": 0.28125,
                "rounded_checksum": 13.75,
                "rounded_indices": [0, 1, 3, 4],
                "rounded_output_shape": [2, 8, 3, 3],
                "status": "ok",
            },
        )

    def test_cli_output_is_machine_readable_json(self):
        output = StringIO()
        with redirect_stdout(output):
            main()
        self.assertEqual(json.loads(output.getvalue()), run_demo())


if __name__ == "__main__":
    unittest.main()
