"""Typed records passed between the stages of the study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, TypeAlias

#: One output record; key order is the exported column order.
Row: TypeAlias = dict[str, Any]

#: An output table.
Table: TypeAlias = list[Row]

#: A daily level series keyed by trading date.
LevelByDate: TypeAlias = dict[date, float]


@dataclass(frozen=True)
class Constituent:
    """One member of the reference basket.

    Attributes:
        ticker: Market symbol, also the study identifier.
        name: Human-readable name.
        sector: Sector classification.
        iv_index: Cboe volatility index for this name, empty when none exists.
        in_iv_basket: Whether this name has an observed implied-volatility history and can
            therefore carry an option leg of the traded structure.
    """

    ticker: str
    name: str
    sector: str
    iv_index: str
    in_iv_basket: bool


@dataclass(frozen=True)
class OptionLeg:
    """One straddle leg of the dispersion structure.

    Attributes:
        ticker: Security the leg is written on.
        quantity: Straddles held. Negative is short.
        strike: Common strike of the call and the put.
        entry_spot: Underlying close at entry.
        entry_iv: Implied volatility the leg was priced at.
        entry_premium: Straddle premium per share at entry.
        entry_vega: Dollar vega at entry, signed with the position.
    """

    ticker: str
    quantity: float
    strike: float
    entry_spot: float
    entry_iv: float
    entry_premium: float
    entry_vega: float
