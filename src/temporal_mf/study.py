"""Level-2 data generation, fitting, and evaluation.

Scientific claims are produced only from a bundle whose ``lineage_status``
is ``PDEBENCH_IC`` or ``OFFICIAL_GENERATOR``. A synthetic family is allowed
for solver tests and is never written up as the external benchmark.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from temporal_mf.baselines.pod_ridge import ARRidge, ResidualRidge
from temporal_mf.data.normalization import fit_standardizer
from temporal_mf.data.splits import assign_splits, ic_hash, near_duplicate_violations
from temporal_mf.eval.bootstrap import paired_bootstrap_mean_diff
from temporal_mf.eval.metrics import quadrature_errors
from temporal_mf.eval.pareto import nondominated
from temporal_mf.eval.rollout import rollout, teacher_forced
from temporal_mf.fidelity.allocate import allocate_counts
from temporal_mf.fidelity.interpolate import fourier_resample
from temporal_mf.models.architecture import FNO1d
from temporal_mf.models.trainer import train_fixed_steps
from temporal_mf.numerical.burgers import grid, integrate
from temporal_mf.numerical.cost_meter import time_solver


def observation_times(t_final: float, dt_obs: float, dtype=torch.float64, device="cpu") -> torch.Tensor:
    n = int(round(t_final / dt_obs))
    return torch.linspace(0, t_final, n + 1, dtype=dtype, device=device)


def simulate_fidelities(
    u0: torch.Tensor,
    *,
    nu: float,
    length: float,
    dt_by_level: dict[str, float],
    t_obs: torch.Tensor,
) -> dict:
    out = {}
    for name, dt in dt_by_level.items():
        field, rec = integrate(u0, nu=nu, length=length, dt=dt, t_observe=t_obs)
        out[name] = {"field": field, "record": rec}
    return out


def stack_windows(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """field (n, time, space) -> inputs, targets, trajectory index per window."""
    inputs = []
    targets = []
    owners = []
    for i in range(field.shape[0]):
        inputs.append(field[i, :-1])
        targets.append(field[i, 1:])
        owners.append(np.full(field.shape[1] - 1, i, dtype=np.int64))
    return np.concatenate(inputs), np.concatenate(targets), np.concatenate(owners)


def fit_neural(
    model: FNO1d,
    inputs: np.ndarray,
    targets: np.ndarray,
    xi: np.ndarray | None,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    max_seconds: float,
) -> dict:
    torch.manual_seed(seed)
    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)
    xi_t = None if xi is None else torch.tensor(xi, dtype=torch.float32, device=device)
    info = train_fixed_steps(
        model, x, y, xi_t, steps=steps, batch_size=batch_size, lr=lr, seed=seed, max_seconds=max_seconds
    )
    return info


def _predict_fn(model, xi_value, device):
    model.eval()

    def step(state: torch.Tensor) -> torch.Tensor:
        xi = None
        if model.conditioned:
            xi = torch.full((state.shape[0], 1), float(xi_value), device=state.device)
        with torch.no_grad():
            return model(state, xi)

    return step


def evaluate_model_on_split(
    model: FNO1d,
    truth: np.ndarray,
    *,
    mean: float,
    std: float,
    length: float,
    horizons: list[int],
    eps: float,
    xi_value: float,
    device: str,
    divergence_rel: float,
) -> dict:
    """truth is physical HF fields (n, time, space). Model sees normalized states."""
    model.eval()
    z = (truth - mean) / std
    z_t = torch.tensor(z, dtype=torch.float32, device=device)
    step = _predict_fn(model, xi_value, device)
    one = teacher_forced(step, z_t).detach().cpu().numpy()
    one_phys = one * std + mean
    max_h = max(horizons)
    rolled = rollout(step, z_t[:, 0], max_h).detach().cpu().numpy()
    rolled_phys = rolled * std + mean
    per_traj = []
    for i in range(truth.shape[0]):
        row = {"trajectory_index": i}
        abs_e, rel_e = quadrature_errors(one_phys[i, 0], truth[i, 1], length, eps)
        row["one_step_abs"] = abs_e
        row["one_step_rel"] = rel_e
        for h in horizons:
            if h >= truth.shape[1]:
                continue
            abs_h, rel_h = quadrature_errors(rolled_phys[i, h], truth[i, h], length, eps)
            diverged = (not np.isfinite(rel_h)) or rel_h > divergence_rel
            row[f"rollout_h{h}_abs"] = abs_h
            row[f"rollout_h{h}_rel"] = rel_h
            row[f"rollout_h{h}_diverged"] = bool(diverged)
        per_traj.append(row)
    return {"per_trajectory": per_traj}


def evaluate_identity(truth: np.ndarray, length: float, horizons: list[int], eps: float, divergence_rel: float) -> dict:
    per_traj = []
    for i in range(truth.shape[0]):
        row = {"trajectory_index": i}
        abs_e, rel_e = quadrature_errors(truth[i, 0], truth[i, 1], length, eps)
        row["one_step_abs"] = abs_e
        row["one_step_rel"] = rel_e
        for h in horizons:
            if h >= truth.shape[1]:
                continue
            # persistence: the state remains the initial condition
            abs_h, rel_h = quadrature_errors(truth[i, 0], truth[i, h], length, eps)
            row[f"rollout_h{h}_abs"] = abs_h
            row[f"rollout_h{h}_rel"] = rel_h
            row[f"rollout_h{h}_diverged"] = bool(rel_h > divergence_rel)
        per_traj.append(row)
    return {"per_trajectory": per_traj}


def evaluate_field_baseline(pred: np.ndarray, truth: np.ndarray, length: float, horizons: list[int], eps: float, divergence_rel: float) -> dict:
    per_traj = []
    for i in range(truth.shape[0]):
        row = {"trajectory_index": i}
        abs_e, rel_e = quadrature_errors(pred[i, 1], truth[i, 1], length, eps)
        row["one_step_abs"] = abs_e
        row["one_step_rel"] = rel_e
        for h in horizons:
            if h >= truth.shape[1]:
                continue
            abs_h, rel_h = quadrature_errors(pred[i, h], truth[i, h], length, eps)
            row[f"rollout_h{h}_abs"] = abs_h
            row[f"rollout_h{h}_rel"] = rel_h
            row[f"rollout_h{h}_diverged"] = bool((not np.isfinite(rel_h)) or rel_h > divergence_rel)
        per_traj.append(row)
    return {"per_trajectory": per_traj}


def summarize(per_traj: list[dict], horizon: int) -> dict:
    rel = [r[f"rollout_h{horizon}_rel"] for r in per_traj]
    one = [r["one_step_rel"] for r in per_traj]
    fail = [1.0 if r[f"rollout_h{horizon}_diverged"] else 0.0 for r in per_traj]
    return {
        "mean_rollout_rel": float(np.mean(rel)),
        "median_rollout_rel": float(np.median(rel)),
        "p90_rollout_rel": float(np.quantile(rel, 0.9)),
        "mean_one_step_rel": float(np.mean(one)),
        "failure_rate": float(np.mean(fail)),
        "n": len(per_traj),
    }


def choose_pod_rank(train_trajs: list[np.ndarray], val_trajs: list[np.ndarray], ranks: list[int], length: float, eps: float) -> tuple[int, ARRidge]:
    best_rank = ranks[0]
    best_score = float("inf")
    best_model = None
    for rank in ranks:
        model = ARRidge(alpha=1.0)
        model.fit(train_trajs, rank)
        errs = []
        for traj in val_trajs:
            pred = model.step_np(traj[0][None, :])[0]
            _a, rel = quadrature_errors(pred, traj[1], length, eps)
            errs.append(rel)
        score = float(np.mean(errs))
        if score < best_score:
            best_score = score
            best_rank = rank
            best_model = model
    assert best_model is not None
    return best_rank, best_model


def rollout_numpy(step_batch, u0: np.ndarray, n_steps: int) -> np.ndarray:
    frames = [u0]
    state = u0
    for _ in range(n_steps):
        state = step_batch(state)
        frames.append(state)
    return np.stack(frames, axis=1)


def cost_report(u0: torch.Tensor, nu: float, length: float, dt_by_level: dict[str, float], t_obs: torch.Tensor, repeats: int = 3) -> dict:
    report = {}
    for name, dt in dt_by_level.items():
        report[name] = time_solver(u0, nu=nu, length=length, dt=dt, t_observe=t_obs, repeats=repeats)
    hf = report["HF"]["median_seconds"]
    lf = report["LF"]["median_seconds"]
    # separation if LF median is below HF minimum across the repeated samples
    hf_min = report["HF"]["min_seconds"]
    report["cost_separation"] = lf < hf_min and lf < hf
    report["lf_over_hf"] = lf / hf if hf > 0 else None
    return report


def json_ready_record(rec) -> dict:
    return {
        "status": rec.status,
        "dt": rec.dt,
        "n_steps": rec.n_steps,
        "max_cfl": rec.max_cfl,
        "mass_drift": rec.mass_drift,
        "energy_start": rec.energy_start,
        "energy_end": rec.energy_end,
        "wall_seconds": rec.wall_seconds,
        "nu": rec.nu,
        "nx": rec.nx,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_split_table(ids: list[str], ics: np.ndarray, seed: int) -> dict:
    labels = assign_splits(ids, seed=seed)
    hashes = {ids[i]: ic_hash(ics[i]) for i in range(len(ids))}
    violations = near_duplicate_violations(labels, hashes)
    return {"labels": labels, "ic_hashes": hashes, "near_duplicate_violations": violations}


def normalized_bundle(train_fields: np.ndarray):
    scaler = fit_standardizer(train_fields)
    return scaler


def run_method_matrix(config: dict, bundle: dict, out_dir: Path, *, open_test: bool) -> dict:
    """Train registered strategies. Test metrics are computed only if ``open_test`` is true."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("RESOURCE_LIMIT: confirmatory neural fits require CUDA on hanqing-win")
    length = float(config["length"])
    horizons = list(config["horizons"])
    eps = float(config["rel_floor"])
    divergence_rel = float(config["divergence_rel"])
    primary_h = horizons[-1]
    labels = bundle["labels"]
    ids = bundle["ids"]
    hf = bundle["HF"]
    lf = bundle["LF"]
    train_idx = [i for i, tid in enumerate(ids) if labels[tid] == "train"]
    val_idx = [i for i, tid in enumerate(ids) if labels[tid] == "val"]
    test_idx = [i for i, tid in enumerate(ids) if labels[tid] == "test"]
    c_hf = float(config["cost_per_hf"])
    c_lf = float(config["cost_per_lf"])
    budget = float(config.get("budget_seconds") or (len(train_idx) * c_hf))
    scaler = fit_standardizer(hf[train_idx])
    results = []
    per_method_val = {}

    def pack(split_idx, field):
        raw = field[split_idx]
        norm = (raw - scaler.mean) / scaler.std
        return raw, norm

    steps = int(config["train_steps"])
    batch = int(config["batch_size"])
    lr = float(config["lr"])
    max_seconds = float(config["max_fit_seconds"])
    seeds = list(config["seeds"])
    strategies = list(config["strategies"])

    for seed in seeds:
        for strategy in strategies:
            t0 = time.perf_counter()
            status = "OK"
            model = None
            xi_value = 1.0
            train_info = {"train_seconds": 0.0, "steps_completed": 0, "status": "OK"}
            if strategy == "POD_DELTA_RIDGE":
                n_pair = max(1, min(len(train_idx), int(budget // (c_hf + c_lf)) or 1))
                use_hf = train_idx[:n_pair]
                use_lf = train_idx[:n_pair]
                spent = n_pair * (c_hf + c_lf)
            elif strategy == "HF_ONLY":
                alloc = allocate_counts(budget, c_hf, c_lf, 1.0)
                use_hf = train_idx[: max(alloc["n_hf"], 1)]
                use_lf = []
                spent = len(use_hf) * c_hf
            else:
                alloc = allocate_counts(budget, c_hf, c_lf, float(config.get("hf_cost_fraction", 0.5)))
                use_hf = train_idx[: max(alloc["n_hf"], 1)]
                use_lf = train_idx[: max(alloc["n_lf"], 1)]
                spent = len(use_hf) * c_hf + len(use_lf) * c_lf
            x_hf = y_hf = x_lf = y_lf = None
            if use_hf:
                _raw, norm_hf = pack(use_hf, hf)
                x_hf, y_hf, _owners = stack_windows(norm_hf)
            if use_lf:
                _raw, norm_lf = pack(use_lf, lf)
                x_lf, y_lf, _owners = stack_windows(norm_lf)
            if strategy == "HF_ONLY":
                model = FNO1d(
                    modes=config["modes"], width=config["width"], n_blocks=config["blocks"], conditioned=False
                ).to(device)
                train_info = fit_neural(
                    model, x_hf, y_hf, None, steps=steps, batch_size=batch, lr=lr, seed=seed, device=device, max_seconds=max_seconds
                )
            elif strategy == "MIXED_CONDITIONED":
                model = FNO1d(
                    modes=config["modes"], width=config["width"], n_blocks=config["blocks"], conditioned=True
                ).to(device)
                inputs = np.concatenate([x_hf, x_lf], axis=0)
                targets = np.concatenate([y_hf, y_lf], axis=0)
                xi = np.concatenate(
                    [np.ones((x_hf.shape[0], 1), dtype=np.float32), np.zeros((x_lf.shape[0], 1), dtype=np.float32)]
                )
                train_info = fit_neural(
                    model, inputs, targets, xi, steps=steps, batch_size=batch, lr=lr, seed=seed, device=device, max_seconds=max_seconds
                )
            elif strategy == "LF_PRETRAIN_HF_FINETUNE":
                model = FNO1d(
                    modes=config["modes"], width=config["width"], n_blocks=config["blocks"], conditioned=False
                ).to(device)
                pre = max(steps // 2, 1)
                fine = steps - pre
                info_a = fit_neural(
                    model, x_lf, y_lf, None, steps=pre, batch_size=batch, lr=lr, seed=seed, device=device, max_seconds=max_seconds / 2
                )
                info_b = fit_neural(
                    model, x_hf, y_hf, None, steps=fine, batch_size=batch, lr=lr, seed=seed + 1000, device=device, max_seconds=max_seconds / 2
                )
                train_info = {
                    "status": info_b["status"] if info_a["status"] == "OK" else info_a["status"],
                    "steps_completed": info_a["steps_completed"] + info_b["steps_completed"],
                    "train_seconds": info_a["train_seconds"] + info_b["train_seconds"],
                    "pretrain_seconds": info_a["train_seconds"],
                    "finetune_seconds": info_b["train_seconds"],
                }
            elif strategy == "POD_DELTA_RIDGE":
                best = None
                for rank in list(config["pod_ranks"]):
                    model_r = ResidualRidge(alpha=1.0)
                    states = np.concatenate([lf[i, :-1] for i in use_hf])
                    lf_next = np.concatenate([lf[i, 1:] for i in use_hf])
                    hf_next = np.concatenate([hf[i, 1:] for i in use_hf])
                    model_r.fit(states, lf_next, hf_next, rank)
                    errs = []
                    for i in val_idx:
                        pred_next = model_r.correct(lf[i, :-1], lf[i, 1:])
                        _a, rel = quadrature_errors(pred_next[0], hf[i, 1], length, eps)
                        errs.append(rel)
                    score = float(np.mean(errs))
                    if best is None or score < best[0]:
                        best = (score, rank, model_r)
                assert best is not None
                _score, rank, model_r = best

                def predict_open_loop(idxs, _model=model_r):
                    preds = []
                    for i in idxs:
                        corrected_next = _model.correct(lf[i, :-1], lf[i, 1:])
                        series = np.concatenate([hf[i, :1], corrected_next], axis=0)
                        preds.append(series)
                    return evaluate_field_baseline(np.stack(preds), hf[idxs], length, horizons, eps, divergence_rel)

                val_eval = predict_open_loop(val_idx)
                test_eval = predict_open_loop(test_idx) if open_test else None
                row = _row(strategy, seed, {"status": "OK", "steps_completed": 0, "train_seconds": time.perf_counter() - t0, "rank": rank}, val_eval, test_eval, primary_h, config, time.perf_counter() - t0)
                row["C_data_seconds"] = spent
                row["n_hf"] = len(use_hf)
                row["n_lf"] = len(use_lf)
                row["pod_rank"] = rank
                row["inference_note"] = "open-loop LF solver trajectory plus train-only residual ridge; LF solve is an inference cost"
                results.append(row)
                continue
            elif strategy == "POD_RIDGE_AR":
                rank, pod = choose_pod_rank(
                    [hf[i] for i in train_idx],
                    [hf[i] for i in val_idx],
                    list(config["pod_ranks"]),
                    length,
                    eps,
                )
                train_info = {"status": "OK", "steps_completed": 0, "train_seconds": time.perf_counter() - t0, "rank": rank}

                def step_np(state, _pod=pod):
                    return _pod.step_np(state)

                def eval_idx(idxs):
                    preds = []
                    for i in idxs:
                        rolled = rollout_numpy(lambda s: step_np(s), hf[i, 0][None, :], primary_h)[0]
                        # one-step and rollout stored as a full prefix trajectory aligned with truth time
                        full = np.zeros_like(hf[i, : primary_h + 1])
                        full[: rolled.shape[0]] = rolled
                        preds.append(full)
                    pred = np.stack(preds)
                    return evaluate_field_baseline(pred, hf[idxs, : primary_h + 1], length, horizons, eps, divergence_rel)

                val_eval = eval_idx(val_idx)
                test_eval = eval_idx(test_idx) if open_test else None
                row = _row(strategy, seed, train_info, val_eval, test_eval, primary_h, config, time.perf_counter() - t0)
                row["pod_rank"] = rank
                results.append(row)
                per_method_val[f"{strategy}_seed{seed}"] = val_eval
                continue
            else:
                raise ValueError(strategy)
            status = train_info["status"]
            val_eval = evaluate_model_on_split(
                model, hf[val_idx], mean=scaler.mean, std=scaler.std, length=length, horizons=horizons, eps=eps,
                xi_value=xi_value, device=device, divergence_rel=divergence_rel,
            )
            test_eval = None
            if open_test and status == "OK":
                test_eval = evaluate_model_on_split(
                    model, hf[test_idx], mean=scaler.mean, std=scaler.std, length=length, horizons=horizons, eps=eps,
                    xi_value=xi_value, device=device, divergence_rel=divergence_rel,
                )
            row = _row(strategy, seed, train_info, val_eval, test_eval, primary_h, config, time.perf_counter() - t0)
            row["status"] = status
            row["C_data_seconds"] = spent
            row["n_hf"] = len(use_hf)
            row["n_lf"] = len(use_lf)
            results.append(row)
            per_method_val[f"{strategy}_seed{seed}"] = val_eval
            if model is not None:
                ckpt = out_dir / f"{strategy}_seed{seed}.pt"
                torch.save({"state_dict": model.state_dict(), "mean": scaler.mean, "std": scaler.std}, ckpt)

    # non-learned baselines once
    id_eval_val = evaluate_identity(hf[val_idx], length, horizons, eps, divergence_rel)
    lf_eval_val = evaluate_field_baseline(lf[val_idx], hf[val_idx], length, horizons, eps, divergence_rel)
    results.append(_row("IDENTITY", None, {"train_seconds": 0, "steps_completed": 0, "status": "OK"}, id_eval_val, None, primary_h, config, 0))
    results.append(_row("LF_SOLVER", None, {"train_seconds": 0, "steps_completed": 0, "status": "OK"}, lf_eval_val, None, primary_h, config, 0))
    if open_test:
        id_test = evaluate_identity(hf[test_idx], length, horizons, eps, divergence_rel)
        lf_test = evaluate_field_baseline(lf[test_idx], hf[test_idx], length, horizons, eps, divergence_rel)
        results[-2]["test"] = summarize(id_test["per_trajectory"], primary_h)
        results[-2]["test_trajectories"] = id_test["per_trajectory"]
        results[-1]["test"] = summarize(lf_test["per_trajectory"], primary_h)
        results[-1]["test_trajectories"] = lf_test["per_trajectory"]

    pareto_rows = []
    for row in results:
        if row.get("test"):
            pareto_rows.append({"method": row["method"], "seed": row["seed"], "cost": row["C_data_seconds"], "error": row["test"]["mean_rollout_rel"]})
    return {
        "results": results,
        "pareto": nondominated(pareto_rows) if pareto_rows else [],
        "scaler": {"mean": scaler.mean, "std": scaler.std},
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "open_test": open_test,
    }


def _row(strategy, seed, train_info, val_eval, test_eval, primary_h, config, wall):
    return {
        "method": strategy,
        "seed": seed,
        "train": train_info,
        "val": summarize(val_eval["per_trajectory"], primary_h),
        "test": None if test_eval is None else summarize(test_eval["per_trajectory"], primary_h),
        "val_trajectories": val_eval["per_trajectory"],
        "test_trajectories": None if test_eval is None else test_eval["per_trajectory"],
        "wall_seconds": wall,
        "C_train_seconds": train_info.get("train_seconds", 0.0),
        "C_data_seconds": config.get("budget_seconds"),
        "primary_horizon": primary_h,
    }


def paired_val_diff(results: list[dict], method_a: str, method_b: str, horizon_key_unused: int = 0) -> dict | None:
    return None
