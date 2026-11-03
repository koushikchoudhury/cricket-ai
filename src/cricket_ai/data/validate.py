"""Data-quality checks on the parsed tables.

Errors mean the tables are unusable (duplicate keys, broken run arithmetic).
Warnings are real-world quirks worth knowing about (e.g. umpire-miscounted
seven-ball overs, which do occur in Cricsheet data).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from cricket_ai.data.parse import EXTRA_KEYS


@dataclass
class QualityReport:
    n_matches: int = 0
    n_deliveries: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(self.errors.values())

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def _record(report: QualityReport, bucket: str, name: str, bad: pd.DataFrame) -> None:
    getattr(report, bucket)[name] = int(len(bad))
    if len(bad):
        report.examples[name] = bad.head(5).to_dict(orient="records")


def validate(matches: pd.DataFrame, deliveries: pd.DataFrame) -> QualityReport:
    r = QualityReport(n_matches=len(matches), n_deliveries=len(deliveries))

    # --- errors ---
    _record(r, "errors", "duplicate_match_id", matches[matches.duplicated("match_id", keep=False)])
    key = ["match_id", "innings", "delivery_seq"]
    _record(r, "errors", "duplicate_delivery_key", deliveries[deliveries.duplicated(key, keep=False)])
    bad_total = deliveries[
        deliveries.runs_total != deliveries.runs_batter + deliveries.runs_extras
    ]
    _record(r, "errors", "runs_total_mismatch", bad_total[key])
    bad_extras = deliveries[deliveries.runs_extras != deliveries[list(EXTRA_KEYS)].sum(axis=1)]
    _record(r, "errors", "extras_breakdown_mismatch", bad_extras[key])
    orphans = deliveries[~deliveries.match_id.isin(matches.match_id)]
    _record(r, "errors", "deliveries_without_match", orphans[key])

    # --- warnings ---
    _record(r, "warnings", "missing_teams", matches[matches.team1.isna() | matches.team2.isna()])
    legal_per_over = (
        deliveries[deliveries.is_legal]
        .groupby(["match_id", "innings", "over"])
        .size()
        .rename("legal")
        .reset_index()
        .merge(matches[["match_id", "balls_per_over"]], on="match_id")
    )
    long_overs = legal_per_over[legal_per_over.legal > legal_per_over.balls_per_over]
    _record(r, "warnings", "overs_with_extra_legal_balls", long_overs)
    _record(r, "warnings", "no_result_or_unknown", matches[matches.result.isin(["no result", "unknown"])])
    return r
