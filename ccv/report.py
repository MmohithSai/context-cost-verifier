"""Static report figures (PNG for the LaTeX report). The CSV tables from analyze() are their table view."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .tasks import SEGMENTS  # noqa: E402

# Validated reference palette: categorical slots in fixed order (never cycled), chrome, diverging pair.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("ccv_diverging", ["#2a78d6", "#f0efec", "#e34948"])
MODEL_ORDER = ["qwen2.5-7b", "llama3.1-8b", "qwen2.5-0.5b"]
FAMILIES = {"subset": ("segment removal", SERIES[0]), "truncate": ("truncation", SERIES[1]),
            "lingua2": ("LLMLingua-2", SERIES[2])}
WORKLOADS = ["qa", "extraction", "tool_call", "all"]
METRIC_LABEL = {"passed": "pass rate", "decode_tokens": "output tokens", "prefill_tokens": "input tokens",
                "gpu_seconds": "GPU-seconds", "energy_j": "energy (J)", "price_cost": "price units (in + 5×out)"}
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.labelsize": 9, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "figure.titlesize": 11, "figure.titleweight": "semibold",
    "savefig.dpi": 200, "savefig.bbox": "tight",
}


def _model_colors(models):
    order = MODEL_ORDER + sorted(set(models) - set(MODEL_ORDER))
    return {m: SERIES[order.index(m)] for m in models}  # color follows the model, not its rank


def _save(fig, path, written):
    fig.savefig(path)
    plt.close(fig)
    written.append(path)


def _workloads(frame):
    return [w for w in WORKLOADS if w in set(frame.workload)]


def _legend(fig, ax, ncols=3):
    """Figure-level legend under the panels; a single series is named by the title instead."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        fig.legend(handles, labels, loc="outside lower center", ncols=ncols)


def _named(title, colors):
    return f"{title}: {next(iter(colors))}" if len(colors) == 1 else title


def shapley_bars(shap, out, written):
    colors = _model_colors(shap.model.unique())
    for metric in ("passed", "decode_tokens", "gpu_seconds"):
        data = shap[shap.metric == metric]
        if data.empty:
            continue
        wls = _workloads(data)
        fig, axes = plt.subplots(1, len(wls), figsize=(2.6 * len(wls) + 0.8, 3.2), sharey=True,
                                 squeeze=False, layout="constrained")
        height = min(0.8 / len(colors), 0.4)  # thin bars: never fill the band
        y = np.arange(len(SEGMENTS))
        for ax, wl in zip(axes[0], wls):
            for k, (model, color) in enumerate(colors.items()):
                d = data[(data.workload == wl) & (data.model == model)].set_index("segment").reindex(SEGMENTS)
                pos = y + (k - (len(colors) - 1) / 2) * height
                ax.barh(pos, d.shapley, height=height * 0.85, color=color, label=model)
                ax.errorbar(d.shapley, pos, xerr=[d.shapley - d.lo, d.hi - d.shapley], fmt="none",
                            ecolor=INK_2, elinewidth=0.8, capsize=0)
            ax.axvline(0, color=AXIS, linewidth=0.8)
            ax.set_title(wl.replace("_", " "))
            ax.grid(axis="y", visible=False)
        axes[0][0].set_yticks(y, SEGMENTS)
        axes[0][0].invert_yaxis()
        fig.supxlabel(f"Shapley value ({METRIC_LABEL[metric]}), mean over tasks with 95% CI",
                      fontsize=9, color=INK_2)
        fig.suptitle(_named(f"Which segments earn their {METRIC_LABEL[metric]}", colors))
        _legend(fig, axes[0][0])
        _save(fig, out / f"shapley_{metric}.png", written)


def interaction_heatmaps(inter, out, written):
    for metric in ("passed", "decode_tokens"):
        for model in inter.model.unique():
            data = inter[(inter.metric == metric) & (inter.model == model)]
            if data.empty:
                continue
            wls = _workloads(data)
            fig, axes = plt.subplots(1, len(wls), figsize=(2.9 * len(wls), 3.5), squeeze=False,
                                     layout="constrained")
            mats = []
            for wl in wls:
                m = np.full((len(SEGMENTS), len(SEGMENTS)), np.nan)
                for r in data[data.workload == wl].itertuples():
                    i, j = SEGMENTS.index(r.a), SEGMENTS.index(r.b)
                    m[i, j] = m[j, i] = r.interaction
                mats.append(m)
            vmax = max(np.nanmax(np.abs(m)) for m in mats) or 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            for ax, wl, m in zip(axes[0], wls, mats):
                image = ax.imshow(m, cmap=DIVERGING, norm=norm)
                ax.set_xticks(range(len(SEGMENTS)), SEGMENTS, rotation=45, ha="right")
                ax.set_yticks(range(len(SEGMENTS)), SEGMENTS if ax is axes[0][0] else [])
                ax.grid(False)
                ax.set_title(wl.replace("_", " "))
            bar = fig.colorbar(image, ax=list(axes[0]), shrink=0.8)
            bar.set_label("interaction  (− redundant · + synergistic)", color=INK_2, fontsize=8)
            bar.outline.set_visible(False)
            fig.suptitle(f"Segment interactions on {METRIC_LABEL[metric]}: {model}")
            _save(fig, out / f"interactions_{metric}_{model}.png", written)


def expansion_ccdf(red, out, written):
    models = list(red.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.3), sharey=True,
                             squeeze=False, layout="constrained")
    for ax, model in zip(axes[0], models):
        for family, (label, color) in FAMILIES.items():
            r = np.sort(red[(red.model == model) & (red.family == family)].ratio.dropna().to_numpy())
            if len(r):
                ax.step(r, 1 - np.arange(len(r)) / len(r), where="post", color=color, label=label)
        ax.axvline(1, color=AXIS, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(model)
        ax.set_xlabel("output tokens vs full context (×)")
    axes[0][0].set_ylabel("share of reduced prompts at or above x")
    _legend(fig, axes[0][0])
    fig.suptitle("How much longer do answers get after the context shrinks?")
    _save(fig, out / "expansion_ccdf.png", written)


def truncation_vs_lingua(cmp, red, out, written):
    data = cmp[(cmp.workload == "all") & cmp.family.isin(["truncate", "lingua2"])]
    if data.empty:
        return
    kept = red.groupby(["model", "config_id"]).kept_long_tokens.mean()
    models = list(data.model.unique())
    fig, axes = plt.subplots(len(models), 2, figsize=(8, 3.1 * len(models)), squeeze=False, layout="constrained")
    for row, model in zip(axes, models):
        d = data[data.model == model]
        for family in ("truncate", "lingua2"):
            label, color = FAMILIES[family]
            f = d[d.family == family].copy()
            f["kept"] = [kept.get((model, c), float(c.split(":")[1])) for c in f.config_id]
            f = f.sort_values("kept")
            row[0].plot(f.kept, f.acc, marker="o", markersize=6, color=color, label=label)
            row[1].plot(f.kept, f.rel_gpu_seconds * 100, marker="o", markersize=6, color=color, label=label)
        row[0].axhline(d.acc_base.iat[0], color=MUTED, linewidth=1)
        row[0].annotate("full context", (1.0, d.acc_base.iat[0]), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color=INK_2)
        row[1].axhline(0, color=AXIS, linewidth=0.8)
        row[0].set_ylabel(f"{model}\npass rate")
        row[1].set_ylabel("GPU-seconds vs full context (%)")
        for ax in row:
            ax.set_xlabel("long-segment tokens kept (fraction)")
            ax.invert_xaxis()
    axes[0][0].set_title("Accuracy")
    axes[0][1].set_title("Physical cost")
    axes[0][1].legend(loc="best")
    fig.suptitle("Truncation vs LLMLingua-2 at the same token budget")
    _save(fig, out / "truncation_vs_llmlingua2.png", written)


def reorder_bars(cmp, out, written):
    data = cmp[cmp.family == "reorder"]
    if data.empty:
        return
    colors = _model_colors(data.model.unique())
    wls = _workloads(data)
    positions = ["first", "middle", "last"]
    fig, axes = plt.subplots(1, len(wls), figsize=(2.4 * len(wls) + 0.6, 3), sharey=True,
                             squeeze=False, layout="constrained")
    width = min(0.7 / len(colors), 0.35)
    for ax, wl in zip(axes[0], wls):
        for k, (model, color) in enumerate(colors.items()):
            d = data[(data.workload == wl) & (data.model == model)].set_index("config_id")
            acc = [d.acc.get(f"reorder:{p}", np.nan) for p in positions]
            ax.bar(np.arange(3) + (k - (len(colors) - 1) / 2) * width, acc, width=width * 0.85,
                   color=color, label=model)
        ax.set_xticks(range(3), positions)
        ax.set_title(wl.replace("_", " "))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("pass rate")
    fig.supxlabel("position of the answer-bearing document", fontsize=9, color=INK_2)
    fig.suptitle(_named("Lost in the middle? Accuracy by gold-document position", colors))
    _legend(fig, axes[0][0])
    _save(fig, out / "reorder_position.png", written)


def pareto_plots(par, out, written):
    for model in par.model.unique():
        data = par[par.model == model]
        wls = _workloads(data)
        fig, axes = plt.subplots(2, len(wls), figsize=(2.9 * len(wls), 6), sharey=True, squeeze=False,
                                 layout="constrained")
        for row, cost in zip(axes, ("gpu_seconds", "price_cost")):
            for ax, wl in zip(row, wls):
                d = data[data.workload == wl].dropna(subset=[cost])
                front = d[d[f"frontier_{cost}"]].sort_values(cost)
                base = d[d.config_id == "subset:111111"]
                ax.scatter(d[cost], d.accuracy, s=14, color=AXIS, linewidths=0, label="all configs")
                ax.plot(front[cost], front.accuracy, color=SERIES[0], marker="o", markersize=5,
                        drawstyle="steps-post", label="Pareto frontier")
                ax.scatter(base[cost], base.accuracy, s=60, color=SERIES[1], edgecolors=SURFACE,
                           linewidths=2, zorder=3, label="full context")
                ax.set_xlabel(METRIC_LABEL[cost])
            row[0].set_ylabel("pass rate")
        for ax, wl in zip(axes[0], wls):
            ax.set_title(wl.replace("_", " "))
        _legend(fig, axes[0][0])
        fig.suptitle(f"Cost vs accuracy for every context config: {model}")
        _save(fig, out / f"pareto_{model}.png", written)


def cost_fit_plot(runs, fit, out, written):
    colors = _model_colors(fit.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), squeeze=False, layout="constrained")
    ratio_note = []
    for model, color in colors.items():
        g, f = runs[runs.model == model], fit[fit.model == model].iloc[0]
        panels = ((g.prefill_tokens, g.prefill_ms, f.prefill_ms_per_token, f.prefill_ms_fixed),
                  (g.decode_tokens - 1, g.decode_ms, f.decode_ms_per_token, f.decode_ms_fixed))
        for ax, (x, y, slope, fixed) in zip(axes[0], panels):
            ax.scatter(x, y, s=6, color=color, alpha=0.25, linewidths=0)
            xs = np.linspace(0, np.nanmax(x), 50)
            ax.plot(xs, fixed + slope * xs, color=color, label=f"{model}: {slope:.2f} ms/token")
        ratio_note.append(f"{model} {f.measured_decode_to_prefill:.0f}×")
    axes[0][0].set(title="Prefill (time to first token)", xlabel="input tokens", ylabel="milliseconds")
    axes[0][1].set(title="Decode (after the first token)", xlabel="output tokens", ylabel="milliseconds")
    for ax in axes[0]:
        ax.legend(loc="upper left")
    fig.suptitle(f"Measured decode:prefill cost per token: {', '.join(ratio_note)} (price lists assume 5×)")
    _save(fig, out / "cost_per_token.png", written)


def plot_all(tables, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context(STYLE):
        if not tables["shapley"].empty:
            shapley_bars(tables["shapley"], out, written)
            interaction_heatmaps(tables["interactions"], out, written)
        if not tables["expansion_cells"].empty:
            expansion_ccdf(tables["expansion_cells"], out, written)
        if not tables["comparisons"].empty:
            truncation_vs_lingua(tables["comparisons"], tables["expansion_cells"], out, written)
            reorder_bars(tables["comparisons"], out, written)
        if not tables["pareto"].empty:
            pareto_plots(tables["pareto"], out, written)
        if not tables["cost_fit"].empty:
            cost_fit_plot(tables["runs"], tables["cost_fit"], out, written)
    print(f"{len(written)} figures written to {out}")
    return written
