"""Two defects in how the project reports on itself.

Both were found while fixing the three feature defects, and both have the same
shape as everything else on this branch: a number was published, CI checked it,
and the thing it was checked against was not the thing it claimed to describe.

  1. `train_segment_model` computed
         int(getattr(base_clf, "best_iteration_", None) or params["iterations"])
     CatBoost sets `best_iteration_` to 0 when the first iteration is the best.
     `0 or 500` is 500, so a ONE-tree model was published as a 500-tree model in
     the README's "Trees" column -- generated automatically and verified by CI,
     because CI compares the README against this number rather than against the
     model. The Lapsed segment hit exactly this.

  2. `eval_sop/churn_eval.py` reproduced the README's per-segment AUCs against a
     hardcoded dict "copied from README.md". The README is generated from the
     artifacts, so the dict went stale the moment any model changed -- and the
     comparison is printed, never asserted, so it went stale silently.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEGMENTS = ("At-Risk", "Price Sensitive", "Loyal Customers", "Champions", "Lapsed")


# ---------------------------------------------------------------------------
# 1. A stump must not be reported as a 500-tree model
# ---------------------------------------------------------------------------


def test_tree_count_is_not_defaulted_when_best_iteration_is_zero():
    """The exact arithmetic of the defect, without training anything: 0 is falsy,
    so `or` replaced it with the iteration cap."""
    for best_iteration_, cap in ((0, 500), (0, 100)):
        defective = int(best_iteration_ or cap)
        correct = int(best_iteration_) if best_iteration_ is not None else cap
        assert defective == cap
        assert correct == 0
        assert defective != correct


def test_churn_model_does_not_use_or_to_default_best_iteration():
    src = (ROOT / "src" / "churn_model.py").read_text(encoding="utf-8")
    offending = [
        ln.strip() for ln in src.splitlines()
        if "best_iteration_" in ln and " or " in ln and not ln.lstrip().startswith("#")
    ]
    assert not offending, (
        f"best_iteration_ is being defaulted with `or` again, which turns a "
        f"legitimate 0 into the iteration cap: {offending}"
    )


def test_trained_models_report_their_real_tree_count():
    """Against the committed artifacts: every reported tree count must match the
    model object, including a stump."""
    import joblib  # noqa: PLC0415

    path = ROOT / "models" / "segment_models.pkl"
    if not path.exists():
        pytest.skip("segment_models.pkl not present")
    models = joblib.load(path)
    mismatched = []
    for seg, md in models.items():
        if not md:
            continue
        actual = getattr(md["base_clf"], "tree_count_", None)
        reported = md["metrics"].get("tree_count")
        if actual is not None and reported is not None and int(actual) != int(reported):
            mismatched.append((seg, int(actual), int(reported)))
    assert not mismatched, (
        f"reported tree counts disagree with the fitted models: {mismatched}"
    )


def test_readme_trees_column_matches_the_models():
    """The README's Trees column is generated. It must come from the model, which
    is the whole failure this test exists for."""
    import joblib  # noqa: PLC0415

    mpath = ROOT / "models" / "segment_models.pkl"
    if not mpath.exists():
        pytest.skip("segment_models.pkl not present")
    models = joblib.load(mpath)
    by_seg = {
        seg: getattr(md["base_clf"], "tree_count_", None)
        for seg, md in models.items() if md
    }
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for seg, trees in by_seg.items():
        if trees is None:
            continue
        m = re.search(rf"^\|\s*{re.escape(seg)}\s*\|.*\|\s*(\d+)\s*\|\s*$", readme, re.M)
        assert m, f"no README row for segment {seg!r}"
        assert int(m.group(1)) == int(trees), (
            f"README says {m.group(1)} trees for {seg}, the model has {trees}"
        )


# ---------------------------------------------------------------------------
# 2. The reproduction check must read the README, not a copy of it
# ---------------------------------------------------------------------------


def test_eval_does_not_hardcode_the_readme_figures():
    src = (ROOT / "eval_sop" / "churn_eval.py").read_text(encoding="utf-8")
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("README_HOLDOUT_AUC = {" in ln for ln in code), (
        "churn_eval.py is hardcoding the README's AUCs again; they are generated "
        "from the artifacts, so a copy of them is stale as soon as a model changes"
    )
    assert any("read_readme_holdout_auc" in ln for ln in code)


def test_readme_parser_finds_every_segment():
    sys.path.insert(0, str(ROOT / "eval_sop"))
    try:
        from churn_eval import read_readme_holdout_auc  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent deps
        pytest.skip(f"churn_eval deps unavailable: {exc}")
    parsed = read_readme_holdout_auc()
    assert set(parsed) == set(SEGMENTS)
    for seg, auc in parsed.items():
        assert 0.4 < auc < 1.0, f"{seg} parsed as {auc}, which is not an AUC"


def test_readme_parser_raises_rather_than_returning_a_partial_table(tmp_path, monkeypatch):
    """If the generated table changes shape the eval must stop, not quietly
    reproduce against four segments out of five."""
    sys.path.insert(0, str(ROOT / "eval_sop"))
    try:
        import churn_eval  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent deps
        pytest.skip(f"churn_eval deps unavailable: {exc}")
    fake = tmp_path / "README.md"
    fake.write_text("| At-Risk | 1 | 2% | **0.689** | 0.7 | a | 61 |\n", encoding="utf-8")
    monkeypatch.setattr(churn_eval, "ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="Could not find a holdout AUC"):
        churn_eval.read_readme_holdout_auc()


def test_a_stump_segment_is_not_silently_scored():
    """A one-tree model's AUC is 'no signal found', not a score. Warning, not an
    exception: the pipeline should still finish and publish the number."""
    src = (ROOT / "src" / "churn_model.py").read_text(encoding="utf-8")
    assert "tree_count <= 1" in src, (
        "nothing flags a stump; the Lapsed segment trained one and the README "
        "presented its AUC like any other"
    )


# ---------------------------------------------------------------------------
# Guard against the arithmetic drifting back
# ---------------------------------------------------------------------------


def test_tree_count_and_best_iteration_differ_by_one(tmp_path):
    """Sanity: CatBoost's tree_count_ is best_iteration_ + 1, which is why the
    README column had to change source rather than just lose the `or`."""
    from catboost import CatBoostClassifier  # noqa: PLC0415

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    y = (X["a"] + rng.normal(scale=0.1, size=200) > 0).astype(int)
    clf = CatBoostClassifier(iterations=20, depth=2, verbose=0, allow_writing_files=False)
    clf.fit(X, y, eval_set=(X, y), early_stopping_rounds=5)
    assert clf.tree_count_ == clf.best_iteration_ + 1
