"""Grading, noise floor, exact Shapley attribution, paired statistics and cost models."""

import itertools
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .ablations import BASELINE
from .runner import read_records
from .tasks import SEGMENTS, grade

N = len(SEGMENTS)
PRICE_RATIO = 5  # commercial APIs typically bill an output token at ~5x an input token
METRICS = ("passed", "decode_tokens", "prefill_tokens", "gpu_seconds", "energy_j", "price_cost")
CELL_COLS = (*METRICS, "capped", "long_tokens", "compress_ms")
DELTAS = ("decode_tokens", "prefill_tokens", "gpu_seconds", "price_cost", "energy_j")
NUMERIC = ("prefill_tokens", "decode_tokens", "prefill_ms", "decode_ms", "queue_ms", "gpu_seconds",
           "energy_j", "gpu_util_mean", "gpu_mem_max_mb", "decode_tok_s", "client_ttft_ms", "client_e2e_ms")


# ---- math --------------------------------------------------------------------------------

def _bit(i, n=N):
    return 1 << (n - 1 - i)  # segment i is character i of the "subset:xxxxxx" mask


def shapley(v, n=N):
    """Exact Shapley values; v maps an int mask to the metric measured with that subset of segments."""
    phi = np.zeros(n)
    for m in range(1 << n):
        size = bin(m).count("1")
        for i in range(n):
            if not m & _bit(i, n):
                phi[i] += (v[m | _bit(i, n)] - v[m]) / (n * comb(n - 1, size))
    return phi


def interactions(v, n=N):
    """Pairwise Shapley interaction index (Grabisch 1997): >0 synergy, <0 redundancy."""
    out = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        bi, bj = _bit(i, n), _bit(j, n)
        total = 0.0
        for m in range(1 << n):
            if not m & (bi | bj):
                size = bin(m).count("1")
                total += (v[m | bi | bj] - v[m | bi] - v[m | bj] + v[m]) / ((n - 1) * comb(n - 2, size))
        out[i, j] = out[j, i] = total
    return out


def leave_one_out(v, n=N):
    full = (1 << n) - 1
    return np.array([v[full] - v[full & ~_bit(i, n)] for i in range(n)])


def mcnemar(base, other):
    """Exact McNemar p-value for paired pass/fail outcomes."""
    base, other = np.asarray(base, bool), np.asarray(other, bool)
    b, c = int((base & ~other).sum()), int((~base & other).sum())
    return stats.binomtest(b, b + c).pvalue if b + c else 1.0


def paired_bootstrap(diffs, n=10_000, seed=0):
    """Mean of paired differences, 95% percentile CI, and a two-sided bootstrap p-value."""
    d = np.asarray(diffs, float)
    means = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = ((np.abs(means - d.mean()) >= abs(d.mean())).sum() + 1) / (n + 1)
    return d.mean(), lo, hi, min(p, 1.0)


def pareto(cost, acc):
    """Indices of points that no other point beats on both cost (lower) and accuracy (higher)."""
    cost, acc = np.asarray(cost, float), np.asarray(acc, float)
    keep, best = [], -np.inf
    for i in np.lexsort((-acc, cost)):
        if acc[i] > best:
            keep.append(i)
            best = acc[i]
    return keep


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    coef = np.polyfit(x, y, 1)
    return coef, 1 - (y - np.polyval(coef, x)).var() / y.var() if y.var() else np.nan


# ---- tables ------------------------------------------------------------------------------

def load(paths, tasks):
    """All successful records for the current task versions, graded and flattened."""
    by_id = {t["id"]: t for t in tasks}
    df = pd.DataFrame([r for p in paths for r in read_records(p)])
    if df.empty:
        raise SystemExit("no records found")
    ok = df.status.eq("ok")
    current = df.task_id.map(lambda k: by_id[k]["hash"] if k in by_id else None).eq(df.task_hash)
    print(f"{len(df)} records: {int(ok.sum())} ok, {int((~ok).sum())} errors, "
          f"{int((ok & ~current).sum())} skipped because the task changed after the run")
    df = df[ok & current].reset_index(drop=True)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    missing = [None] * len(df)
    attr = pd.DataFrame([a if isinstance(a, dict) else {} for a in df.get("attribution", missing)])
    df = pd.concat([df, attr.add_prefix("attr_")], axis=1)
    df["long_tokens"] = sum(df.get(f"attr_{s}", np.nan) for s in ("examples", "retrieved", "history"))
    df["compress_ms"] = [c.get("compress_ms") if isinstance(c, dict) else np.nan
                         for c in df.get("compression", missing)]
    df["passed"] = [float(grade(by_id[k], o, tc)) for k, o, tc in zip(df.task_id, df.output, df.tool_calls)]
    df["price_cost"] = df.prefill_tokens + PRICE_RATIO * df.decode_tokens
    df["capped"] = df.finish_reason.eq("length").astype(float)
    return df


def cells(df):
    """Mean of each metric per (model, task, config); repeats of an identical request are pooled."""
    return df.groupby(["model", "workload", "task_id", "config_id"], as_index=False)[list(CELL_COLS)].mean()


def pair_with_baseline(c):
    base = c[c.config_id == BASELINE].drop(columns=["config_id", "workload"])
    return c[c.config_id != BASELINE].merge(base, on=["model", "task_id"], suffixes=("", "_base"))


def _cv(x):
    x = x.dropna()
    return x.std(ddof=1) / x.mean() if len(x) > 1 and x.mean() else np.nan


def noise_floor(df):
    rows = []
    for (model, task), g in df[df.suite == "noise"].groupby(["model", "task_id"]):
        d = g.decode_tokens.to_numpy(float)
        diffs = np.abs(d[:, None] - d[None, :])[np.triu_indices(len(d), 1)]
        rate = g.passed.mean()
        rows.append({
            "model": model, "workload": g.workload.iat[0], "task_id": task, "repeats": len(g),
            "decode_mean": d.mean(), "decode_sd": d.std(ddof=1) if len(d) > 1 else 0.0,
            "decode_range": d.max() - d.min(),
            "decode_absdiff_p95": np.percentile(diffs, 95) if len(diffs) else 0.0,
            "identical_output_rate": g.output.eq(g.output.mode().iat[0]).mean(),
            "pass_rate": rate, "pass_flip_rate": min(rate, 1 - rate),
            "prefill_ms_cv": _cv(g.prefill_ms), "decode_ms_cv": _cv(g.decode_ms),
            "ms_per_decode_token_cv": _cv(g.decode_ms / (g.decode_tokens - 1).where(g.decode_tokens > 1)),
        })
    return pd.DataFrame(rows)


def shapley_tables(c):
    """Per-task Shapley/LOO values, workload means with bootstrap CIs, and mean pairwise interactions."""
    rows, pair_rows = [], []
    subsets = c[c.config_id.str.startswith("subset:")]
    for (model, workload, task), g in subsets.groupby(["model", "workload", "task_id"]):
        if len(g) < 1 << N:
            continue  # sweep not finished for this task yet
        g = g.set_index(g.config_id.str[len("subset:"):].map(lambda b: int(b, 2)))
        for metric in METRICS:
            if g[metric].isna().any():
                continue
            v = g[metric].to_dict()
            phi, loo, inter = shapley(v), leave_one_out(v), interactions(v)
            key = {"model": model, "workload": workload, "task_id": task, "metric": metric}
            rows += [{**key, "segment": s, "shapley": phi[i], "loo": loo[i]} for i, s in enumerate(SEGMENTS)]
            pair_rows += [{**key, "a": SEGMENTS[i], "b": SEGMENTS[j], "interaction": inter[i, j]}
                          for i, j in itertools.combinations(range(N), 2)]
    per_task, inter = pd.DataFrame(rows), pd.DataFrame(pair_rows)
    if per_task.empty:
        return per_task, pd.DataFrame(), pd.DataFrame()
    summary = []
    for key, g in pd.concat([per_task, per_task.assign(workload="all")]).groupby(
            ["model", "workload", "metric", "segment"]):
        mean, lo, hi, _ = paired_bootstrap(g.shapley)
        summary.append({**dict(zip(("model", "workload", "metric", "segment"), key)),
                        "shapley": mean, "lo": lo, "hi": hi, "loo": g.loo.mean(), "tasks": len(g)})
    inter = (pd.concat([inter, inter.assign(workload="all")])
             .groupby(["model", "workload", "metric", "a", "b"], as_index=False).interaction.mean())
    return per_task, pd.DataFrame(summary), inter


def comparisons(pairs):
    """Every config vs baseline, paired by task: McNemar on accuracy, bootstrap on costs, BH-corrected."""
    rows = []
    for (model, workload, cfg), g in pd.concat([pairs, pairs.assign(workload="all")]).groupby(
            ["model", "workload", "config_id"]):
        base_ok, ok = g.passed_base >= 0.5, g.passed >= 0.5
        row = {"model": model, "workload": workload, "config_id": cfg, "family": cfg.split(":")[0],
               "tasks": len(g), "acc_base": base_ok.mean(), "acc": ok.mean(), "acc_p": mcnemar(base_ok, ok)}
        for m in DELTAS:
            d = (g[m] - g[f"{m}_base"]).dropna()
            if len(d):
                mean, lo, hi, p = paired_bootstrap(d)
                row.update({f"d_{m}": mean, f"d_{m}_lo": lo, f"d_{m}_hi": hi, f"d_{m}_p": p,
                            f"rel_{m}": mean / g[f"{m}_base"].mean()})
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in [c for c in out.columns if c.endswith("_p")]:
        q = col[:-2] + "_q"
        out[q] = np.nan
        for idx in out.groupby(["model", "workload"]).groups.values():
            p = out.loc[idx, col].dropna()
            if len(p):
                out.loc[p.index, q] = stats.false_discovery_control(p.to_numpy(), method="bh")
    return out


def expansion(pairs):
    """Output-length ratio after each context reduction, and how often the total cost went up."""
    red = pairs[~pairs.config_id.str.startswith("reorder")].copy()
    red["family"] = red.config_id.str.split(":").str[0]
    red["ratio"] = red.decode_tokens / red.decode_tokens_base
    red["kept_long_tokens"] = red.long_tokens / red.long_tokens_base
    red["cost_up_gpu"] = red.gpu_seconds > red.gpu_seconds_base
    red["cost_up_price"] = red.price_cost > red.price_cost_base
    rows = []
    for (model, family), g in pd.concat([red, red.assign(family="all")]).groupby(["model", "family"]):
        r = g.ratio.dropna()
        rows.append({"model": model, "family": family, "cells": len(g), "median": r.median(),
                     "p90": r.quantile(0.9), "max": r.max(), "share_over_1.5x": (r > 1.5).mean(),
                     "share_over_2x": (r > 2).mean(), "capped_share": (g.capped > 0).mean(),
                     "cost_up_gpu_share": g.cost_up_gpu.mean(), "cost_up_price_share": g.cost_up_price.mean()})
    return red, pd.DataFrame(rows)


def cost_fit(df):
    """Linear per-token prefill and decode times; their ratio is the physical 'price ratio'."""
    rows = []
    for model, g in df.groupby("model"):
        a = g.dropna(subset=["prefill_tokens", "prefill_ms"])
        b = g[g.decode_tokens > 1].dropna(subset=["decode_ms"])
        if len(a) < 3 or len(b) < 3 or np.ptp(a.prefill_tokens) == 0 or np.ptp(b.decode_tokens) == 0:
            continue
        pre, pre_r2 = _fit(a.prefill_tokens, a.prefill_ms)
        dec, dec_r2 = _fit(b.decode_tokens - 1, b.decode_ms)
        rows.append({"model": model, "runs": len(g),
                     "prefill_ms_per_token": pre[0], "prefill_ms_fixed": pre[1], "prefill_r2": pre_r2,
                     "decode_ms_per_token": dec[0], "decode_ms_fixed": dec[1], "decode_r2": dec_r2,
                     "measured_decode_to_prefill": dec[0] / pre[0], "price_ratio": PRICE_RATIO})
    return pd.DataFrame(rows)


def pareto_table(c):
    agg = (pd.concat([c, c.assign(workload="all")])
           .groupby(["model", "workload", "config_id"], as_index=False)
           .agg(accuracy=("passed", "mean"), gpu_seconds=("gpu_seconds", "mean"),
                price_cost=("price_cost", "mean"), decode_tokens=("decode_tokens", "mean"),
                prefill_tokens=("prefill_tokens", "mean"), tasks=("task_id", "nunique")))
    for cost in ("gpu_seconds", "price_cost"):
        agg[f"frontier_{cost}"] = False
        for idx in agg.groupby(["model", "workload"]).groups.values():
            g = agg.loc[idx].dropna(subset=[cost])
            agg.loc[g.index[pareto(g[cost], g.accuracy)], f"frontier_{cost}"] = True
    return agg


def _md(frame):
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    return "\n".join(lines + ["| " + " | ".join(map(fmt, row)) + " |" for row in frame.itertuples(index=False)])


def summary_md(df, t):
    parts = ["# ccv analysis summary", "## Runs",
             _md(df.groupby(["model", "suite"]).size().reset_index(name="runs"))]
    if not t["cost_fit"].empty:
        parts += ["## Physical prefill vs decode cost (ms per token)", _md(t["cost_fit"])]
    if not t["noise_floor"].empty:
        nf = t["noise_floor"].groupby("model", as_index=False).agg(
            tasks=("task_id", "size"), decode_sd_median=("decode_sd", "median"),
            decode_absdiff_p95_max=("decode_absdiff_p95", "max"),
            identical_output_rate=("identical_output_rate", "mean"), pass_flip_rate=("pass_flip_rate", "mean"),
            prefill_ms_cv=("prefill_ms_cv", "median"), ms_per_decode_token_cv=("ms_per_decode_token_cv", "median"))
        parts += ["## Noise floor (identical repeats)", _md(nf)]
    if not t["shapley"].empty:
        s = t["shapley"][t["shapley"].workload == "all"]
        for metric in ("passed", "decode_tokens", "gpu_seconds"):
            piv = s[s.metric == metric].pivot(index="segment", columns="model", values="shapley").reindex(SEGMENTS)
            parts += [f"## Shapley value per segment: {metric} (all workloads)", _md(piv.reset_index())]
    if not t["comparisons"].empty:
        cmp = t["comparisons"]
        cmp = cmp[(cmp.workload == "all") & (cmp.family != "subset")]
        cols = ["model", "config_id", "tasks", "acc_base", "acc", "acc_q", "d_decode_tokens", "d_decode_tokens_q",
                "rel_gpu_seconds", "d_gpu_seconds_q", "rel_price_cost"]
        parts += ["## Truncation, LLMLingua-2 and reordering vs baseline (all workloads)",
                  _md(cmp[[c for c in cols if c in cmp]])]
    if not t["expansion"].empty:
        parts += ["## Output expansion after context reduction", _md(t["expansion"])]
    if not t["pareto"].empty:
        p = t["pareto"][t["pareto"].frontier_gpu_seconds]
        parts += ["## Pareto frontier (GPU-seconds vs accuracy)",
                  _md(p[["model", "workload", "config_id", "accuracy", "gpu_seconds", "price_cost"]])]
    return "\n\n".join(parts) + "\n"


def analyze(paths, tasks, out):
    """Write every table as CSV plus summary.md; returns the tables for plotting."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(paths, tasks)
    c = cells(df)
    pairs = pair_with_baseline(c)
    shap_tasks, shap, inter = shapley_tables(c)
    red, exp = expansion(pairs) if not pairs.empty else (pd.DataFrame(), pd.DataFrame())
    tables = {
        "runs": df.drop(columns=["output", "tool_calls", "attribution", "compression", "env"], errors="ignore"),
        "noise_floor": noise_floor(df), "shapley_per_task": shap_tasks, "shapley": shap, "interactions": inter,
        "comparisons": comparisons(pairs) if not pairs.empty else pd.DataFrame(),
        "expansion_cells": red, "expansion": exp, "cost_fit": cost_fit(df), "pareto": pareto_table(c),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "summary.md").write_text(summary_md(df, tables), encoding="utf-8")
    print(f"tables and summary written to {out}")
    return tables
