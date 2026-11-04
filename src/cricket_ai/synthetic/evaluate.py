"""Synthetic vs real: fidelity, validity, utility and privacy.

* **Fidelity**: per-column distance (total variation for discrete, Kolmogorov-Smirnov for
  continuous) and the mean absolute difference between correlation matrices.
* **Validity**: share of rows that obey every cricket rule in ``table.VALIDITY_RULES``.
* **Utility**: train on synthetic, test on real (TSTR) against train on real, test on
  real (TRTR). Two tasks: runs in the over (MAE) and whether a wicket falls (AUC).
* **Privacy**: distance to closest real training record (DCR), compared with the same
  distance for real holdout rows, and a distance-based membership-inference attack.
  This table has many legitimately identical rows (e.g. a quiet first over), so raw
  exact-match counts mean little; the holdout comparison is the fair reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, roc_auc_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from cricket_ai.synthetic.table import COLUMNS, CONTINUOUS, DISCRETE, validity  # noqa: E402

CONTEXT = ["innings", "over", "runs_at_start", "wickets_at_start"]
REAL_REFERENCE = "real (TRTR)"


# ----- fidelity -----
def column_fidelity(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in COLUMNS:
        if c in DISCRETE:
            p = real[c].value_counts(normalize=True)
            q = synth[c].value_counts(normalize=True)
            cats = p.index.union(q.index)
            value = (
                0.5 * np.abs(p.reindex(cats, fill_value=0) - q.reindex(cats, fill_value=0)).sum()
            )
            rows.append({"column": c, "metric": "tvd", "value": round(float(value), 4)})
        else:
            stat = ks_2samp(real[c], synth[c]).statistic
            rows.append({"column": c, "metric": "ks", "value": round(float(stat), 4)})
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[COLUMNS].astype(float).corr().fillna(0.0)


def correlation_difference(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    diff = (correlation_matrix(real) - correlation_matrix(synth)).abs().to_numpy()
    upper = diff[np.triu_indices_from(diff, k=1)]
    return round(float(upper.mean()), 4)


# ----- utility -----
def utility(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, float]:
    """Fit on ``train`` (real or synthetic), score on the real holdout."""
    reg = HistGradientBoostingRegressor(max_iter=200, random_state=0)
    reg.fit(train[CONTEXT], train.runs)
    out = {
        "runs_mae": round(
            float(mean_absolute_error(holdout.runs, reg.predict(holdout[CONTEXT]))), 4
        )
    }
    y = (train.wickets > 0).astype(int)
    if y.nunique() < 2:
        out["wicket_auc"] = 0.5
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(train[CONTEXT], y)
        p = clf.predict_proba(holdout[CONTEXT])[:, 1]
        out["wicket_auc"] = round(float(roc_auc_score((holdout.wickets > 0).astype(int), p)), 4)
    return out


# ----- privacy -----
def privacy(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synth: pd.DataFrame,
    max_rows: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    mean = train[COLUMNS].mean()
    std = train[COLUMNS].std().replace(0, 1)

    def scaled(df: pd.DataFrame, n: int) -> np.ndarray:
        x = ((df[COLUMNS] - mean) / std).to_numpy(dtype=float)
        return x[rng.choice(len(x), n, replace=False)] if len(x) > n else x

    train_nn = NearestNeighbors(n_neighbors=1).fit(scaled(train, len(train)))
    dcr_synth = train_nn.kneighbors(scaled(synth, max_rows))[0][:, 0]
    dcr_hold = train_nn.kneighbors(scaled(holdout, max_rows))[0][:, 0]

    # Membership inference: are training rows closer to the synthetic data than unseen rows?
    m = min(len(train), len(holdout), max_rows)
    members, outsiders = scaled(train, m), scaled(holdout, m)
    synth_nn = NearestNeighbors(n_neighbors=1).fit(scaled(synth, len(synth)))
    d_members = synth_nn.kneighbors(members)[0][:, 0]
    d_outsiders = synth_nn.kneighbors(outsiders)[0][:, 0]
    labels = np.r_[np.ones(m), np.zeros(m)]
    mia_auc = roc_auc_score(labels, -np.r_[d_members, d_outsiders])

    med_s, med_h = float(np.median(dcr_synth)), float(np.median(dcr_hold))
    return {
        "dcr_median_synthetic": round(med_s, 4),
        "dcr_median_holdout": round(med_h, 4),
        "dcr_ratio": round(med_s / med_h, 4) if med_h > 0 else float("nan"),
        "exact_match_synthetic": round(float(np.mean(dcr_synth < 1e-9)), 4),
        "exact_match_holdout": round(float(np.mean(dcr_hold < 1e-9)), 4),
        "membership_auc": round(float(mia_auc), 4),
    }


# ----- full evaluation -----
def evaluate_all(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synthetic: dict[str, pd.DataFrame],
    report_dir: Path,
) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary, per_column, per_rule = [], {}, {}

    ref = {"model": REAL_REFERENCE, **utility(train, holdout)}
    ref["validity"] = validity(train)["all_rules"]
    summary.append(ref)

    for name, synth in synthetic.items():
        synth = synth[COLUMNS]
        fid = column_fidelity(holdout, synth)
        per_column[name] = fid.set_index("column").value.to_dict()
        rules = validity(synth)
        per_rule[name] = rules
        summary.append(
            {
                "model": name,
                "rows": len(synth),
                "mean_tvd": round(float(fid[fid.metric == "tvd"].value.mean()), 4),
                "mean_ks": round(float(fid[fid.metric == "ks"].value.mean()), 4),
                "corr_diff": correlation_difference(holdout, synth),
                "validity": rules["all_rules"],
                **utility(synth, holdout),
                **privacy(train, holdout, synth),
            }
        )

    table = pd.DataFrame(summary).set_index("model")
    columns_table = pd.DataFrame(per_column)
    rules_table = pd.DataFrame(per_rule)
    plot_marginals(holdout, synthetic, report_dir / "marginals.png")
    plot_correlations(holdout, synthetic, report_dir / "correlations.png")

    result = {
        "rows": {"train": len(train), "holdout": len(holdout)},
        "summary": table.reset_index().to_dict(orient="records"),
        "per_column": per_column,
        "validity_rules": per_rule,
    }
    (report_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    (report_dir / "report.md").write_text(_markdown(result, table, columns_table, rules_table))
    return result


# ----- plots -----
def plot_marginals(real: pd.DataFrame, synthetic: dict[str, pd.DataFrame], path: Path) -> None:
    names = list(synthetic)
    ncols = 4
    nrows = int(np.ceil(len(COLUMNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    for ax, col in zip(axes.ravel(), COLUMNS, strict=False):
        if col in CONTINUOUS:
            bins = np.linspace(real[col].min(), real[col].max(), 40)
            ax.hist(
                real[col],
                bins=bins,
                density=True,
                histtype="step",
                lw=2,
                label="real",
                color="black",
            )
            for n in names:
                ax.hist(
                    synthetic[n][col], bins=bins, density=True, histtype="step", lw=1.3, label=n
                )
        else:
            cats = np.sort(real[col].unique())[:16]
            width = 0.8 / (1 + len(names))
            series = [("real", real)] + [(n, synthetic[n]) for n in names]
            for i, (label, df) in enumerate(series):
                p = df[col].value_counts(normalize=True).reindex(cats, fill_value=0)
                color = "black" if label == "real" else None
                ax.bar(np.arange(len(cats)) + i * width, p.values, width, label=label, color=color)
            ax.set_xticks(np.arange(len(cats)) + 0.4 - width / 2, cats, fontsize=7)
        ax.set_title(col, fontsize=10)
    for ax in axes.ravel()[len(COLUMNS) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[-1].legend(handles, labels, frameon=False, fontsize=11, loc="center")
    fig.suptitle("Real holdout vs synthetic: column distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_correlations(real: pd.DataFrame, synthetic: dict[str, pd.DataFrame], path: Path) -> None:
    names = list(synthetic)
    base = correlation_matrix(real)
    fig, axes = plt.subplots(1, 1 + len(names), figsize=(4.6 * (1 + len(names)), 4.4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(base, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title("real correlations")
    for ax, n in zip(axes[1:], names, strict=True):
        diff = (correlation_matrix(synthetic[n]) - base).abs()
        im = ax.imshow(diff, vmin=0, vmax=0.5, cmap="Reds")
        ax.set_title(f"|{n} - real|")
    for ax in axes:
        ax.set_xticks(range(len(COLUMNS)), COLUMNS, rotation=90, fontsize=7)
        ax.set_yticks(range(len(COLUMNS)), COLUMNS, fontsize=7)
    if names:
        fig.colorbar(im, ax=axes[1:].tolist(), shrink=0.8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _markdown(result: dict, table: pd.DataFrame, columns: pd.DataFrame, rules: pd.DataFrame) -> str:
    rows = result["rows"]
    fidelity_cols = ["rows", "mean_tvd", "mean_ks", "corr_diff", "validity"]
    utility_cols = ["runs_mae", "wicket_auc"]
    privacy_cols = [
        "dcr_median_synthetic",
        "dcr_median_holdout",
        "dcr_ratio",
        "exact_match_synthetic",
        "exact_match_holdout",
        "membership_auc",
    ]
    synth_only = table.drop(index=REAL_REFERENCE)
    return (
        "# Synthetic over-level data: evaluation\n\n"
        f"Generators were fitted on {rows['train']} real overs; all comparisons use "
        f"{rows['holdout']} held-out real overs from unseen matches.\n\n"
        "## Fidelity and validity (lower distance is better; validity 1.0 is perfect)\n\n"
        f"{synth_only[fidelity_cols].to_markdown()}\n\n"
        "## Utility: train on synthetic, test on real\n\n"
        "Lower MAE / higher AUC is better; the real (TRTR) row is the target to match.\n\n"
        f"{table[utility_cols].to_markdown()}\n\n"
        "## Privacy\n\n"
        "DCR ratio near 1 and membership AUC near 0.5 mean the synthetic data is no closer "
        "to the training rows than unseen real rows are (no memorisation).\n\n"
        f"{synth_only[privacy_cols].to_markdown()}\n\n"
        "## Per-column distance\n\n"
        f"{columns.to_markdown()}\n\n"
        "## Validity by rule\n\n"
        f"{rules.to_markdown()}\n\n"
        "![marginals](marginals.png)\n\n![correlations](correlations.png)\n"
    )
