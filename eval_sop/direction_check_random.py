"""How often does the pipeline's uplift direction check pass for pure noise?

positivity_signflip.py showed ONE random-noise draw passing
src/uplift_model.validate_uplift_direction. One draw is an anecdote; this runs
200 independent uniform-noise scores (seeds 0..199) on the committed Cell2Cell
uplift.parquet and reports the pass rate. A check that cannot tell noise from
signal passes about half the time.

Run:  python eval_sop/direction_check_random.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common import OUT, ROOT  # noqa: E402

import uplift_model  # noqa: E402

N_DRAWS = 200


def main():
    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "uplift.parquet"))
    passed = []
    for s in range(N_DRAWS):
        df["_s"] = np.random.default_rng(s).random(len(df))
        passed.append(bool(uplift_model.validate_uplift_direction(df, uplift_col="_s")["direction_ok"]))
    k = int(sum(passed))
    # Wilson 95% interval for the pass rate
    p, n, z = k / N_DRAWS, N_DRAWS, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    res = {"n_draws": N_DRAWS, "n_passed": k, "pass_rate": p, "wilson95": [c - h, c + h],
           "seeds": "numpy default_rng(0..199), uniform[0,1)"}
    with open(os.path.join(OUT, "direction_check_random_noise.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(res)


if __name__ == "__main__":
    main()
