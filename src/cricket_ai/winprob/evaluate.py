"""Train every model on the same split, evaluate on held-out seasons and write a report."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cricket_ai.winprob import model as mlp  # noqa: E402
from cricket_ai.winprob.baselines import fit_baselines, predict  # noqa: E402
from cricket_ai.winprob.data import FEATURE_SETS, LABEL, Split, auto_split  # noqa: E402
from cricket_ai.winprob.metrics import (  # noqa: E402
    all_metrics,
    metrics_by_stage,
    reliability_table,
)

log = logging.getLogger(__name__)
MLP_NAME = "mlp_pytorch"


def run_experiment(
    states: pd.DataFrame,
    report_dir: Path,
    model_dir: Path,
    cfg: mlp.TrainConfig | None = None,
    split: Split | None = None,
) -> dict:
    cfg = cfg or mlp.TrainConfig()
    split = split or auto_split(states)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info("Split: %s %s", split.description, split.sizes())

    baselines = fit_baselines(
        split.train,
        split.val,
        seed=cfg.seed,
        half_life=cfg.recency_half_life,
        features=list(cfg.features),
    )
    net, hist = mlp.train_mlp(split.train, split.val, cfg)
    mlp.save(net, cfg, hist, model_dir / "winprob_mlp.pt")
    joblib.dump(baselines, model_dir / "winprob_baselines.joblib")

    y_test = split.test[LABEL].to_numpy()
    preds = {name: predict(m, split.test) for name, m in baselines.items()}
    preds[MLP_NAME] = mlp.predict_proba(net, split.test)

    table = pd.DataFrame({name: all_metrics(y_test, p) for name, p in preds.items()}).T
    table = table.sort_values("log_loss")
    stages = metrics_by_stage(split.test, preds[MLP_NAME])

    _plot_reliability(y_test, preds, report_dir / "reliability.png")
    _plot_history(hist, report_dir / "training_curve.png")

    result = {
        "split": split.description,
        "sizes": split.sizes(),
        "test_metrics": table.to_dict(orient="index"),
        "mlp_by_stage": stages.to_dict(orient="records"),
        "mlp_training": {
            "best_epoch": hist.best_epoch,
            "epochs_run": len(hist.val_loss),
            "calibration": {"scale": round(hist.calib_scale, 4), "bias": round(hist.calib_bias, 4)},
            "config": asdict(cfg),
        },
    }
    (report_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    (report_dir / "report.md").write_text(_markdown(result, table, stages))
    return result


ABLATION_VARIANTS: list[tuple[str, float | None]] = [
    ("core", None),
    ("core+target_vs_par", None),
    ("core+par+target_vs_par", None),
    ("core", 4.0),
    ("core+target_vs_par", 4.0),
    ("core+par+target_vs_par", 4.0),
]


def run_ablation(
    states: pd.DataFrame,
    report_dir: Path,
    cfg: mlp.TrainConfig | None = None,
    split: Split | None = None,
    variants: list[tuple[str, float | None]] | None = None,
) -> pd.DataFrame:
    """Train logistic regression and the MLP for each feature set x recency setting.

    Every variant uses the same split and Platt calibration, so differences come from
    the feature set and the weighting alone.
    """
    base = cfg or mlp.TrainConfig()
    split = split or auto_split(states)
    report_dir.mkdir(parents=True, exist_ok=True)
    y_test = split.test[LABEL].to_numpy()
    rows = []
    for feature_set, half_life in variants or ABLATION_VARIANTS:
        features = FEATURE_SETS[feature_set]
        log.info("Ablation: features=%s half_life=%s", feature_set, half_life)
        cfg_v = replace(base, features=tuple(features), recency_half_life=half_life)
        net, hist = mlp.train_mlp(split.train, split.val, cfg_v)
        logistic = fit_baselines(
            split.train,
            split.val,
            seed=base.seed,
            half_life=half_life,
            features=features,
            only=["logistic"],
        )["logistic"]
        p_mlp = mlp.predict_proba(net, split.test)
        stages = metrics_by_stage(split.test, p_mlp).set_index("stage")
        variant = {"features": feature_set, "recency": half_life or "off"}
        for name, p in (("mlp", p_mlp), ("logistic", predict(logistic, split.test))):
            rows.append({**variant, "model": name, **all_metrics(y_test, p)})
        rows[-2].update(
            {
                "early_ece": stages.loc["early", "ece"] if "early" in stages.index else None,
                "best_epoch": hist.best_epoch,
                "best_val_loss": round(min(hist.val_loss), 4),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(report_dir / "ablation.csv", index=False)
    md = (
        "# Win-probability ablation\n\n"
        f"**Split:** {split.description}. Test metrics; all variants use Platt calibration.\n\n"
        f"{table.to_markdown(index=False)}\n"
    )
    (report_dir / "ablation.md").write_text(md)
    return table


def _plot_reliability(y: np.ndarray, preds: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")
    for name, p in preds.items():
        if name == "base_rate":
            continue
        t = reliability_table(y, p)
        ax.plot(t.predicted, t.observed, marker="o", ms=4, label=name)
    ax.set(
        xlabel="predicted win probability",
        ylabel="observed win rate",
        title="Reliability on held-out seasons",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_history(hist: mlp.TrainHistory, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    epochs = range(1, len(hist.train_loss) + 1)
    ax.plot(epochs, hist.train_loss, label="train")
    ax.plot(epochs, hist.val_loss, label="validation")
    ax.axvline(hist.best_epoch, ls=":", c="grey", label=f"best epoch {hist.best_epoch}")
    ax.set(xlabel="epoch", ylabel="log loss", title="MLP training")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _markdown(result: dict, table: pd.DataFrame, stages: pd.DataFrame) -> str:
    sizes = result["sizes"]
    tr = result["mlp_training"]
    return (
        "# Win-probability model report\n\n"
        f"**Split:** {result['split']}  \n"
        f"train {sizes['train']['matches']} matches / {sizes['train']['states']} states, "
        f"val {sizes['val']['matches']} / {sizes['val']['states']}, "
        f"test {sizes['test']['matches']} / {sizes['test']['states']}\n\n"
        "## Test metrics (lower is better except AUC)\n\n"
        f"{table.to_markdown()}\n\n"
        "## MLP by stage of the chase\n\n"
        f"{stages.to_markdown(index=False)}\n\n"
        "## MLP training\n\n"
        f"Best epoch {tr['best_epoch']} of {tr['epochs_run']}, "
        f"Platt calibration scale {tr['calibration']['scale']}, "
        f"bias {tr['calibration']['bias']}. "
        f"Recency half-life: {tr['config']['recency_half_life'] or 'off'} seasons.\n\n"
        "![reliability](reliability.png)\n\n![training](training_curve.png)\n"
    )
