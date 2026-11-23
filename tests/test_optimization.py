import numpy as np
import optuna
import pytest

from gcvae.optimization import optimize_eta2
from gcvae.selection import select

SEEDS = (1, 2, 3, 4, 5)


def suggest(trial):
    return {"x": trial.suggest_float("x", 0.0, 1.0)}


def fake_train_eval(params, seed):
    """eta2 grows with x; for x > 0.8, seeds 1 and 2 diverge (2 of 5)."""
    if params["x"] > 0.8 and seed in (1, 2):
        return None
    pred = np.arange(100) % 3
    return {"eta2": params["x"], "pi_min": 0.3, "pred": pred, "oracle_ari": 0.5}


def test_two_of_five_diverged_is_discarded():
    _, df = optimize_eta2(fake_train_eval, suggest, n_trials=15, seeds=SEEDS,
                          n_startup_trials=15, progress=False)
    bad = df[df["x"] > 0.8]
    good = df[df["x"] <= 0.8]
    assert len(bad) and bad["diverged"].all() and (bad["score"] == -1.0).all()
    assert bad["eta2"].isna().all()
    assert (~good["diverged"]).all()
    np.testing.assert_allclose(good["score"], good["x"])
    assert (good["agreement"] == 1.0).all() and (good["oracle_ari"] == 0.5).all()
    row, _ = select(df)  # the discarded trials never reach the shortlist
    assert row["x"] <= 0.8


@pytest.mark.parametrize("n_seeds, n_failed, discarded",
                         [(5, 1, False), (5, 2, True), (3, 1, False), (3, 2, True), (2, 1, True)])
def test_diverged_fraction_rule(n_seeds, n_failed, discarded):
    def train_eval(params, seed):
        return None if seed <= n_failed else {"eta2": 0.4, "pi_min": 0.3, "pred": np.zeros(10)}
    _, df = optimize_eta2(train_eval, suggest, n_trials=1, seeds=range(1, n_seeds + 1), progress=False)
    assert bool(df["diverged"].iloc[0]) == discarded


def test_one_of_five_diverged_is_tolerated():
    def train_eval(params, seed):
        return None if seed == 1 else {"eta2": 0.4, "pi_min": 0.3, "pred": np.zeros(10)}
    _, df = optimize_eta2(train_eval, suggest, n_trials=2, seeds=SEEDS, progress=False)
    assert (df["n_ok"] == 4).all() and (~df["diverged"]).all()
    np.testing.assert_allclose(df["score"], 0.4)


def test_collapse_penalty():
    def train_eval(params, seed):
        return {"eta2": 0.6, "pi_min": 0.025, "pred": np.zeros(10)}
    _, df = optimize_eta2(train_eval, suggest, n_trials=1, seeds=(1, 2), progress=False)
    assert df["score"].iloc[0] == pytest.approx(0.6 - 0.5)  # half-way below the 0.05 floor


def test_resume_to_total_budget_and_retry_interrupted(tmp_path):
    storage = str(tmp_path / "study.log")
    calls = []

    def train_eval(params, seed):
        calls.append((params["x"], seed))
        return {"eta2": params["x"], "pi_min": 0.3, "pred": np.zeros(10)}

    optimize_eta2(train_eval, suggest, n_trials=3, seeds=(1, 2), storage=storage, progress=False)
    assert len(calls) == 6

    # Simulate a crash: a trial left RUNNING in the stored study.
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    study = optuna.load_study(study_name="bo", storage=JournalStorage(JournalFileBackend(storage)))
    stale = study.ask({"x": optuna.distributions.FloatDistribution(0.0, 1.0)})
    stale_x = stale.params["x"]

    calls.clear()
    study, df = optimize_eta2(train_eval, suggest, n_trials=5, seeds=(1, 2), storage=storage,
                              progress=False)
    assert len(df) == 5                       # total budget, not 5 more
    assert calls[0][0] == pytest.approx(stale_x)   # interrupted config retried first
    states = [t.state for t in study.trials]
    assert states.count(optuna.trial.TrialState.FAIL) == 1


def test_reports_the_params_returned_by_suggest():
    def rounded(trial):
        return {"x": round(trial.suggest_float("x", 0.0, 1.0), 2)}
    _, df = optimize_eta2(lambda p, s: {"eta2": p["x"], "pi_min": 0.3, "pred": np.zeros(4)},
                          rounded, n_trials=3, seeds=(1,), progress=False)
    np.testing.assert_allclose(df["x"], df["x"].round(2))
    assert "params" not in df.columns
