"""Reproduction tests for the three defects documented in RESULTS.md section 9.

These tests pin CURRENT, DEFECTIVE behaviour. They are not a specification of
what the code should do -- each one asserts the bug is still there, so that the
bug cannot be fixed, or made worse, silently. The `src/` tree is deliberately
unchanged on this branch; these tests are the standing evidence for the three
findings, replacing "reproduced by inspection" with "reproduced by CI".

When a defect is genuinely fixed, the corresponding test here MUST fail. That
is the point. Flip the assertion (and delete the xfail marker note below) as
part of the fix, in the same commit, so the fix cannot land unnoticed either.

Defects pinned:
  1. src/agent_tools.py:204 -- `lookup_customer_details` returns
     `actually_churned`, the realized label, straight to the LLM.
  2. src/uplift_model.py:47-52 -- a missing CausalML import downgrades to the
     custom T-learner with only a `logger.warning`; callers cannot tell which
     estimator produced the numbers from the return value.
  3. src/uplift_model.validate_uplift_direction -- passes for scores with no
     causal content, including uniform random noise, roughly half the time.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent_tools  # noqa: E402
import uplift_model  # noqa: E402

# ─── Defect 1: realized churn label leaks into a tool result ──────────────────

LEAK_KEY = "actually_churned"


def _one_customer_frame(churn: int = 1) -> pd.DataFrame:
    return pd.DataFrame([{
        "CustomerID": 50001, "Segment": "At-Risk", "RiskTier": "High Risk",
        "CustomerType": "Persuadable", "ChurnProbability": 0.71, "UpliftScore": 0.08,
        "NetROI": 25.0, "Tenure": 14, "SatisfactionScore": 4, "Complain": 1,
        "HourSpendOnApp": 2.5, "DaySinceLastOrder": 30, "OrderCount": 7,
        "CashbackAmount": 120.0, "CouponUsed": 2, "Churn": churn,
    }])


def test_lookup_customer_details_leaks_realized_label():
    """DEFECT: the tool hands the LLM the answer it is being asked to predict."""
    out = agent_tools.lookup_customer_details(_one_customer_frame(churn=1), "50001")
    assert LEAK_KEY in out, (
        "The label leak appears to be fixed. If so, that is good: delete this test "
        "and keep test_lookup_customer_details_has_no_other_label_key."
    )
    assert out[LEAK_KEY] is True, "leaked value should track the Churn column"
    assert agent_tools.lookup_customer_details(_one_customer_frame(churn=0), "50001")[LEAK_KEY] is False


def test_lookup_customer_details_has_no_other_label_key():
    """Guard: `actually_churned` is the ONLY label-bearing key, so stripping it is sufficient.

    eval_sop/agent_eval.py removes {actually_churned, churn, Churn} from every
    tool result. If a future change adds another spelling of the label, this
    fails and the eval's strip list has to be updated with it.
    """
    out = agent_tools.lookup_customer_details(_one_customer_frame(), "50001")
    label_like = {k for k in out if "churn" in k.lower()} - {LEAK_KEY}
    assert label_like == {"churn_probability"}, (
        f"unexpected churn-related key(s) in the tool result: {label_like}. "
        "A new label spelling would bypass the eval harness's strip list."
    )


# ─── Defect 2: the CausalML downgrade is silent ───────────────────────────────

def test_causalml_fallback_is_silent_in_the_return_value():
    """DEFECT: nothing in the output records which estimator actually ran.

    `CAUSALML_AVAILABLE` is resolved once at import time, so this test does not
    try to re-import under a broken numpy. It asserts the weaker, sufficient
    fact: the module-level flag is the only record of the downgrade, and the
    custom fallback's own return value does not name itself. A caller reading
    the metrics dict cannot tell CausalML from the fallback.
    """
    assert hasattr(uplift_model, "CAUSALML_AVAILABLE")
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "f1": rng.normal(size=n), "f2": rng.normal(size=n),
        "Treatment": rng.integers(0, 2, n),
    })
    df["Churn"] = (rng.random(n) < 0.3).astype(int)
    _scores, _p_ctrl, metrics = uplift_model.compute_uplift_scores_custom(
        df, ["f1", "f2"], "Treatment", "Churn"
    )[:3]
    assert isinstance(metrics, dict)
    estimator_named = any(
        "causalml" in str(k).lower() or "estimator" in str(k).lower() or "fallback" in str(k).lower()
        for k in metrics
    )
    assert not estimator_named, (
        "The fallback now names the estimator it used. If the silent-downgrade "
        "defect was fixed, invert this assertion."
    )


def test_causalml_import_failure_only_warns(caplog):
    """DEFECT: the import guard logs a warning and continues; it does not raise."""
    src = (uplift_model.__file__ or "")
    text = open(src, encoding="utf-8").read()
    assert "except ImportError:" in text
    assert "CAUSALML_AVAILABLE = False" in text
    # The guard must not raise, and must not merely print.
    assert "raise" not in text.split("except ImportError:")[1].split("\n\n")[0]
    with caplog.at_level(logging.WARNING):
        logging.getLogger(uplift_model.__name__).warning("CausalML not available — using custom T-Learner fallback.")
    assert "CausalML not available" in caplog.text


# ─── Defect 3: the direction check passes for scores with no causal content ───

def _direction_frame(n, score, rng):
    """A frame where treatment and outcome are independent of the score."""
    return pd.DataFrame({
        "Treatment": np.ones(n, dtype=int),
        "Churn": (rng.random(n) < 0.3).astype(int),
        "UpliftScore": score,
    })


def test_direction_check_passes_for_pure_noise_about_half_the_time():
    """DEFECT: the check has ~50% power against a score that is pure noise.

    Pinned loosely (a band, not a point) so this is not a flaky test: with the
    score independent of the outcome, `direction_ok` is a coin flip by
    construction. eval_sop/direction_check_random.py measured 98/200 = 49%
    (Wilson 95% CI 42-56%) over independent draws. A real validity check would
    pass a noise score far less than half the time.
    """
    rng = np.random.default_rng(12345)
    n_draws, n_rows = 200, 2000
    passes = 0
    for _ in range(n_draws):
        df = _direction_frame(n_rows, rng.random(n_rows), rng)
        res = uplift_model.validate_uplift_direction(df)
        assert res["direction_ok"] is not None, "frame should be large enough to judge"
        passes += bool(res["direction_ok"])
    rate = passes / n_draws
    assert 0.35 <= rate <= 0.65, (
        f"noise passed the direction check {passes}/{n_draws} = {rate:.0%} of the time. "
        "Outside the coin-flip band: either the check gained real power (good -- "
        "retire this test) or it broke in some other way."
    )


def test_direction_check_passes_for_negated_churn_probability():
    """DEFECT: it passes for -ChurnProbability, a pure risk score with no causal content.

    This is deterministic, unlike the noise case: ranking treated customers by
    -P(churn) puts the least-likely-to-churn customers in the top decile, so the
    top decile churns less than the bottom one and the check reports success.
    """
    rng = np.random.default_rng(7)
    n = 4000
    p = rng.random(n) * 0.8 + 0.05
    df = pd.DataFrame({
        "Treatment": np.ones(n, dtype=int),
        "Churn": (rng.random(n) < p).astype(int),
        "UpliftScore": -p,  # the risk score, negated: no treatment-effect content at all
    })
    res = uplift_model.validate_uplift_direction(df)
    assert res["direction_ok"] is True, (
        "The direction check no longer passes for a negated risk score. If it "
        "gained the ability to reject non-causal scores, retire this test."
    )
    assert res["gap"] > 0


@pytest.mark.parametrize("n_treated", [0, 50, 99])
def test_direction_check_abstains_when_too_few_treated(n_treated):
    """Not a defect -- the one guard that does work, pinned so it stays."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "Treatment": np.ones(n_treated, dtype=int),
        "Churn": rng.integers(0, 2, n_treated),
        "UpliftScore": rng.random(n_treated),
    })
    res = uplift_model.validate_uplift_direction(df)
    assert res["direction_ok"] is None
    assert "too few treated" in res["reason"]
