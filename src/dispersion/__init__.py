"""Dispersion trading and the implied-correlation risk premium.

Implied correlation is taken from Cboe's published index and realised correlation from observed
closing prices, so the premium is measured entirely on market data. See ``README.md`` for the
research design and for what the study deliberately does not claim.
"""

from __future__ import annotations

from dispersion.config import CostModel, StudyConfig
from dispersion.models import Constituent
from dispersion.pipeline import StudyResults, headline, run_study, write_outputs
from dispersion.trade import DispersionTrade

__version__ = "1.0.0"

__all__ = [
    "Constituent",
    "CostModel",
    "DispersionTrade",
    "StudyConfig",
    "StudyResults",
    "__version__",
    "headline",
    "run_study",
    "write_outputs",
]
