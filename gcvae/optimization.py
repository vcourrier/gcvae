"""Bayesian optimisation of the label-free relevance signal eta^2 (Optuna TPE).

The thesis selects GCVAE hyperparameters without labels: a Bayesian
optimisation proposes configurations and maximises the guarded mean eta^2 over
a few seeds; the final configuration is then chosen among the trials with the
relevance-then-reproducibility rule of `gcvae.selection.select`.

`optimize_eta2` is model-agnostic: the caller provides `suggest(trial)`, which
draws one configuration, and `train_eval(params, seed)`, which trains it with
one seed and evaluates it without labels.
"""
import os
import traceback
import warnings

import numpy as np
import pandas as pd

from gcvae.selection import agreement


def _call(args):
    """Run `train_eval(params, seed)`; a failure becomes {'failed': True}. Module-level
    so that a process pool can pickle it."""
    train_eval, params, seed = args
    try:
        out = train_eval(params, seed)
    except Exception as exc:
        warnings.warn(f"run failed for params={params} seed={seed}: {exc!r}\n{traceback.format_exc()}")
        return {"failed": True, "err": repr(exc)}
    if out is None:
        return {"failed": True}
    return {"failed": False, **out}


def optimize_eta2(train_eval, suggest, n_trials=40, seeds=(1, 2), diverged_frac=0.4,
                  collapse_floor=0.05, collapse_penalty=1.0, sampler_seed=42,
                  n_startup_trials=10, storage=None, study_name="bo", load_if_exists=True,
                  save_path=None, pool=None, warm_start_trials=None, progress=True):
    """Maximise the guarded mean eta^2 over a search space with Optuna TPE.

    Each trial draws a configuration with `suggest(trial)` and trains it once per
    seed with `train_eval(params, seed)`. The objective is

        mean eta^2 over the seeds that did not fail
        - collapse_penalty * max(0, collapse_floor - pi_min) / collapse_floor,

    where pi_min is the mean over seeds of the smallest mixing weight: the
    penalty grows linearly once a cluster empties below `collapse_floor`. A
    trial in which a share `diverged_frac` or more of the seeds failed scores
    `-collapse_penalty` (the worst value).

    Args:
        train_eval (callable): `(params, seed) -> dict` with at least `eta2`
            (float), `pi_min` (smallest mixing weight at the end of training)
            and `pred` (cluster assignments of the evaluation set, used for the
            across-seed agreement). Optional keys: `min_cluster_frac` and
            `n_clusters_used` (logged as the minimum over seeds) and `oracle_*` metrics computed with
            labels (logged as means, never used by the search). Return None,
            `{'failed': True}` or raise to mark a diverged run. With `pool`, it
            must be a module-level function.
        suggest (callable): `optuna.Trial -> dict` of hyperparameters.
        n_trials (int): Total number of completed trials of the study. When a
            stored study is resumed, only the missing trials are run.
        seeds (sequence[int]): Seeds trained for every configuration.
        diverged_frac (float): A trial is discarded when (failed seeds) / (seeds)
            >= diverged_frac. The default 0.4 discards 2 failures out of 5,
            1 out of 2 and 2 out of 3, and tolerates 1 out of 3 to 5.
        collapse_floor (float): pi_min below which the collapse penalty starts.
        collapse_penalty (float): Penalty at pi_min = 0, and score of a discarded trial.
        sampler_seed (int): Seed of the TPE sampler.
        n_startup_trials (int): Random trials before TPE starts modelling.
        storage (str or None): Path of an Optuna journal file. It makes the
            study resumable and lets several processes optimise it together.
        study_name (str): Name of the study in the storage.
        load_if_exists (bool): Resume the stored study if it exists.
        save_path (str or None): CSV of the trials, rewritten after each trial.
        pool (ResilientPool or None): Runs the seeds of a trial in parallel.
        warm_start_trials (list[optuna.trial.FrozenTrial] or None): Trials added
            to a new, empty study.
        progress (bool): Print one line per trial.

    Returns:
        tuple[optuna.Study, pandas.DataFrame]: The study and one row per completed
            trial (hyperparameters, `score`, `eta2`, `eta2_std`, `pi_min`,
            `agreement`, `n_ok`, `n_failed`, `diverged`, ...), sorted by eta2.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = suggest(trial)
        trial.set_user_attr("params", params)   # the values actually trained (suggest may round)
        jobs = [(train_eval, params, s) for s in seeds]
        res = pool.map(_call, jobs) if pool is not None else [_call(j) for j in jobs]
        res = [r if r is not None else {"failed": True} for r in res]
        ok = [r for r in res if not r["failed"]]
        n_failed = len(res) - len(ok)
        trial.set_user_attr("n_ok", len(ok))
        trial.set_user_attr("n_failed", n_failed)
        discarded = n_failed / len(res) >= diverged_frac
        trial.set_user_attr("diverged", discarded)
        if discarded:
            trial.set_user_attr("eta2", float("nan"))
            return -collapse_penalty

        eta2s = [r["eta2"] for r in ok]
        eta2 = float(np.nanmean(eta2s))
        pi_min = float(np.nanmean([r.get("pi_min", np.nan) for r in ok]))
        penalty = (collapse_penalty * max(0.0, collapse_floor - pi_min) / collapse_floor
                   if np.isfinite(pi_min) else collapse_penalty)
        trial.set_user_attr("eta2", eta2)
        trial.set_user_attr("eta2_std", float(np.nanstd(eta2s, ddof=1)) if len(ok) >= 2 else float("nan"))
        trial.set_user_attr("pi_min", pi_min)
        trial.set_user_attr("agreement", agreement([r["pred"] for r in ok]))
        for key in ("min_cluster_frac", "n_clusters_used"):
            if all(key in r for r in ok):
                trial.set_user_attr(key, float(min(r[key] for r in ok)))
        for key in sorted({k for r in ok for k in r if k.startswith("oracle_")}):
            trial.set_user_attr(key, float(np.nanmean([r.get(key, np.nan) for r in ok])))
        return eta2 - penalty

    sampler = optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=n_startup_trials)
    if storage:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        study = optuna.create_study(
            study_name=study_name, storage=JournalStorage(JournalFileBackend(storage)),
            direction="maximize", sampler=sampler, load_if_exists=load_if_exists)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    if warm_start_trials and len(study.get_trials(deepcopy=False)) == 0:
        study.add_trials(warm_start_trials)
    # A trial cut off by an interruption stays RUNNING: mark it failed and retry
    # its configuration first.
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        study.enqueue_trial(t.params)

    def _build_df(st):
        rows = [{"trial": t.number, **t.params, **t.user_attrs.get("params", {}), "score": t.value,
                 **{k: v for k, v in t.user_attrs.items() if k != "params"}}
                for t in st.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))]
        df = pd.DataFrame(rows)
        if "eta2" in df.columns:
            df = df.sort_values("eta2", ascending=False, ignore_index=True)
        return df

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    def _callback(st, tr):
        if progress:
            print(f"  trial {tr.number:>3}: score={tr.value:+.4f}  eta2={tr.user_attrs.get('eta2', float('nan')):.4f}"
                  f"  pi_min={tr.user_attrs.get('pi_min', float('nan')):.3f}"
                  f"  agreement={tr.user_attrs.get('agreement', float('nan')):.3f}"
                  f"  ok={tr.user_attrs.get('n_ok')}/{len(seeds)}", flush=True)
        if save_path:   # atomic: per-PID tmp file, then replace
            tmp = f"{save_path}.{os.getpid()}.tmp"
            _build_df(st).to_csv(tmp, index=False)
            os.replace(tmp, save_path)

    n_done = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    if n_done < n_trials:
        study.optimize(objective, n_trials=n_trials - n_done, callbacks=[_callback])
    return study, _build_df(study)
