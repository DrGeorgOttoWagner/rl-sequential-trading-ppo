"""
costs.py — transaction cost configuration.

Provenance
----------
Adapted from the author's earlier supervised-learning study (not part of this repository)
(``CostConfig``, ``ZERO_COST``, ``CONSERVATIVE_COST``).  The per-leg
decomposition (fee + slippage + spread, in basis points) is kept unchanged so
that the RL results are cost-comparable with the v5B ML backtests.

Additions for the RL environment: multiplicative per-leg equity factor and its
log, used by the open-to-open accounting in ``env.py``.

Frozen values (docs/methodology.md): conservative profile 0.10% fee plus
0.05% slippage allowance per purchase or sale, about 0.30% for a round trip; zero cost is an
ablation only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    """Transaction costs per leg in basis points (1 bp = 0.01 %)."""

    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    spread_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fee_bps", "slippage_bps", "spread_bps"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.one_leg_fraction >= 1.0:
            raise ValueError("per-leg cost must be below 100 %")

    @property
    def one_leg_fraction(self) -> float:
        """Total cost per leg as a decimal fraction of traded notional."""
        return (self.fee_bps + self.slippage_bps + self.spread_bps) / 10_000.0

    @property
    def round_trip_fraction(self) -> float:
        return 2.0 * self.one_leg_fraction

    @property
    def leg_factor(self) -> float:
        """Multiplicative equity factor for one leg: equity *= (1 - cost)."""
        return 1.0 - self.one_leg_fraction

    @property
    def log_leg_cost(self) -> float:
        """log(1 - cost) for one leg; a non-positive number."""
        return math.log(self.leg_factor)

    def factor_for_legs(self, legs: int) -> float:
        return self.leg_factor ** legs

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CostConfig(fee={self.fee_bps / 100:.10g}%, slippage={self.slippage_bps / 100:.10g}%, "
            f"spread={self.spread_bps / 100:.10g}%, per_leg={self.one_leg_fraction * 100:.10g}% of the traded value)"
        )


ZERO_COST = CostConfig(fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0)
CONSERVATIVE_COST = CostConfig(fee_bps=10.0, slippage_bps=5.0, spread_bps=0.0)

COST_PROFILES: dict[str, CostConfig] = {
    "zero": ZERO_COST,
    "conservative": CONSERVATIVE_COST,
}
