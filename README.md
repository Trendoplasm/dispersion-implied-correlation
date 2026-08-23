# Dispersion Trading and the Implied-Correlation Premium

Index options price a view on how much the constituents of an index move *together*. This study
measures that view against what the constituents actually did, then backtests the dispersion trade
that harvests the difference.

[![CI](https://github.com/Trendoplasm/dispersion-implied-correlation/actions/workflows/ci.yml/badge.svg)](https://github.com/Trendoplasm/dispersion-implied-correlation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![Types: mypy strict](https://img.shields.io/badge/types-mypy%20strict-1f5082)](https://mypy-lang.org/)

## The finding in three lines

1. **There is no unconditional correlation premium.** Over 3,372 trading days, implied correlation
   averaged 30.75% against 30.92% subsequently realised — a premium of **−0.17 points**, with a
   bootstrap interval of −0.65 to +0.30 that straddles zero.
2. **But the premium is real when a signal says it is.** After a high z-score on the
   implied-minus-realised spread, the premium averages **+3.98 points** (interval +2.88 to +5.03,
   clear of zero), and is positive 63% of the time. After a low z-score it averages **−3.41
   points**.
3. **Transaction costs are about the size of the edge.** The short-correlation trade earns +0.57%
   per trade before you touch the cost assumptions; double them and the whole strategy turns
   negative.

That combination — a conditional effect that a plausible cost model very nearly erases — is the
honest result, and it is the one this repository is built to demonstrate rather than obscure.

## Read this first: what is and is not measured

| Component | Status |
|---|---|
| Implied correlation | **Observed.** Cboe's COR1M / COR3M / COR6M indexes |
| Dispersion cross-check | **Observed.** Cboe's DSPX index |
| Realised correlation | **Observed.** Computed from 46 constituents' closing prices |
| Constituent option legs | **Five names.** Only AAPL, AMZN, GOOGL, GS and IBM have a published volatility index |
| Option prices | **Modelled.** Black-Scholes marks from 30-day at-the-money volatility, not historical quotes |
| Capital denominator | **A research normalisation**, not broker margin |

Both legs of the *premium* are real market data, so that measurement is a genuine empirical result.
The *backtest* carries two limits that the README, the code, and every exported table restate:

- **The traded basket is five names against a 500-name index.** Cboe publishes a volatility index
  for only five S&P 500 members, so the constituent side of the structure is a proxy for the index,
  not a replica of it. Real dispersion desks trade dozens of names. This is the single largest gap
  between the backtest and the trade it represents.
- **Option prices are model marks.** A real chain has a skew across strikes, a bid-ask spread at
  every one of them, and finite depth. Only an entry half-spread is modelled.

## What the study measures

### Implied correlation is observed, not inferred

The variance identity behind a dispersion trade is

```text
sigma_index^2 = sum_i sum_j w_i w_j sigma_i sigma_j rho_ij
```

Inverting it for a single `rho` gives implied correlation. The convenient part is that **Cboe
already publishes the answer** — the COR indexes are exactly this calculation applied to the top
S&P 500 constituents — so the study takes the observed series rather than reconstructing it from
weights it does not have. The identity is still implemented, and tested against closed-form cases,
because the realised side uses it too.

### Realised correlation two ways

| Measure | How | Why it is here |
|---|---|---|
| **Average pairwise** | Mean correlation of every constituent pair | Needs no index weights. Historical S&P 500 weights are not free, and inventing them would put a guess at the centre of the measurement |
| **Identity-implied** | Invert the variance identity on realised volatilities | The construct Cboe's index uses, so it is the like-for-like comparison |

The two differ systematically — the identity weights each pair by the product of its weights and
volatilities, so large, volatile names dominate it. The study reports both instead of choosing,
because the gap between them is a property of the basket, not an error. Over the sample they come
out at 31.3% and 30.7%.

### The observed term structure

| Series | Horizon | Mean over the sample |
|---|---:|---:|
| COR1M | 1 month | 30.8% |
| COR3M | 3 months | 35.0% |
| COR6M | 6 months | 38.3% |

Longer-dated correlation is priced higher, the same upward slope volatility itself shows.

## Results

Study period **2 January 2013 to 30 June 2026** — 3,372 panel days and 124 non-overlapping trades.

### The premium depends entirely on the regime

| Regime | Days | Mean index IV | Implied | Realised | Premium |
|---|---:|---:|---:|---:|---:|
| Low volatility | 1,133 | 12.5% | 25.3% | 27.9% | **−2.59 points** |
| Middle | 1,118 | 16.1% | 27.9% | 27.7% | **+0.26 points** |
| High volatility | 1,121 | 24.8% | 39.1% | 37.3% | **+1.85 points** |

In calm markets correlation is *underpriced* — selling it loses. Only when volatility is elevated
is it overpriced. Any claim that index options systematically overprice correlation is, on this
evidence, a claim about high-volatility periods specifically.

### The signal identifies when the premium is there

| Signal state | Days | Mean premium | Positive |
|---|---:|---:|---:|
| Short-correlation (z > 0.75) | 753 | **+3.98 points** | 63.0% |
| No signal | 1,930 | −0.63 points | 51.7% |
| Long-correlation (z < −0.90) | 689 | **−3.41 points** | 41.5% |
| All days | 3,372 | −0.17 points | 52.1% |

The spread between the first and third rows is 7.4 points of correlation, on a signal built only
from information available beforehand. The test suite enforces that: appending later data cannot
change any earlier z-score, and the standardising window ends the day *before* the value it scores.

### The trade

| Direction | Trades | Return per trade | Annualised | Win rate | Worst trade | Worst 10% |
|---|---:|---:|---:|---:|---:|---:|
| Short correlation | 72 | **+0.57%** | +6.8% | 52.8% | −6.8% | −4.9% |
| Long correlation | 52 | **−0.96%** | −11.6% | 40.4% | −13.1% | −7.5% |
| Both | 124 | −0.07% | −0.9% | 47.6% | −13.1% | −6.2% |

Only one side works. Taking both directions, as the specified rule does, gives back everything the
short side earns.

### Where the profit comes from

Mean dollars per trade, on a structure scaled to $100,000 of vega per side:

| Direction | Correlation | Volatility | Residual | Costs | Net |
|---|---:|---:|---:|---:|---:|
| Short correlation | +1,034 | +14 | +375 | −729 | **+694** |
| Long correlation | +824 | −134 | −977 | −712 | **−999** |

Two things to read here. First, the **volatility term is $14 on $100,000 of vega** — under 1% of the
gross. That is the empirical confirmation that the structure really is a correlation trade and not a
disguised bet on the level of volatility. Second, the long-correlation direction gets its
correlation call *right* on average (+824) and still loses, because the residual — realised gamma,
time decay, hedging error — costs it $977. Being short five single-name gammas is expensive.

The four columns sum to net profit by construction, and the exported table carries a `check` column
reporting the floating-point residual of that identity.

### Costs against the edge

| Cost assumption | Return per trade | Win rate | Cost per trade | As share of capital |
|---|---:|---:|---:|---:|
| Half | +0.25% | 53.2% | $361 | 0.32% |
| **Baseline** | **−0.07%** | 47.6% | $722 | 0.65% |
| Double | −0.72% | 39.5% | $1,444 | 1.29% |
| Triple | −1.37% | 33.1% | $2,166 | 1.94% |

A twelve-leg structure rebalanced daily is expensive to run. The edge and the bill are the same
order of magnitude, which is the most important practical sentence in this study.

### In sample against out of sample

The thresholds (+0.75 / −0.90) come from the original study's specification rather than being
fitted here, but the split still matters:

| Period | Trades | Return per trade | Win rate |
|---|---:|---:|---:|
| To 2020 | 75 | −0.08% | 48.0% |
| After 2020 | 49 | −0.06% | 46.9% |

Stable, and stably around zero. Worth noting that the *premium* was mildly positive before 2021
(+0.67 points) and negative after (−1.42 points), while correlation itself fell from 35% to 24% —
the era of single-stock dispersion.

## Quickstart

Requires Python 3.11 or newer; developed and validated on 3.13.

```bash
make setup      # install Python 3.13 and dependencies
make data       # download the Cboe index history and the price history
make reproduce  # run the study, writing to outputs/
```

Or with plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/fetch_cboe_data.py
python scripts/fetch_price_data.py
dispersion --output-dir outputs
```

A successful run prints:

```text
Completed 124 dispersion trades over 3372 panel days: unconditional correlation premium -0.17 points, +3.98 points after a short-correlation signal, mean trade return -0.07% on research capital.
```

`dispersion --help` documents every option. The study is importable too:

```python
from pathlib import Path
from dispersion import CostModel, StudyConfig, run_study

results = run_study(Path("data/raw"), Path("data/reference/basket.csv"), StudyConfig(), CostModel())
print(results.pooled_panel["mean_correlation_premium"])
```

### Development tasks

```bash
make lint       # ruff check + format check
make typecheck  # mypy, strict
make test       # 179 tests
make verify     # re-run and diff against outputs/
make help       # list every target
```

## Method

### The signal

- **Spread** — observed implied correlation minus trailing 60-day realised correlation.
- **Z-score** — the spread standardised against its own trailing 252-day mean and standard
  deviation, computed over a window that ends the day *before* the observation.
- **Entry** — sell index volatility against constituent volatility when z > 0.75; buy it when
  z < −0.90; otherwise stand aside.

Excluding the scored value from its own benchmark matters more than it sounds. Including it shrinks
every z-score toward zero and makes the signal look better behaved than it is.

### The structure

One straddle per leg: short the index and long each constituent for a short-correlation trade, and
the reverse for a long-correlation one. Both sides are scaled to the same dollar vega, so a parallel
move in every implied volatility nets out and only the *spread* between index and constituent
volatility can pay. Each leg is marked daily with its own observed volatility index and its own
observed close, delta-hedged in its own underlying, financed, and held 21 trading days to
settlement. Entries do not overlap.

### Costs and capital

An entry half-spread of 1% of premium, $0.65 per contract per leg, and hedge turnover at 0.5 basis
points on the index and 1.0 on single names. Capital is long premium plus 20% of short-option
notional — transparent, but explicitly **not** broker margin, which would rise precisely when the
position is losing.

## How the code is organised

| Module | Responsibility |
|---|---|
| [`config.py`](src/dispersion/config.py) | Study period, thresholds, sizing, cost model, data contract |
| [`loaders.py`](src/dispersion/loaders.py) | Reading Cboe, price, and basket inputs, failing loudly |
| [`correlation.py`](src/dispersion/correlation.py) | Both realised-correlation measures and the variance identity |
| [`blackscholes.py`](src/dispersion/blackscholes.py) | Option marks and Greeks |
| [`panel.py`](src/dispersion/panel.py) | The daily implied-versus-realised correlation panel |
| [`signals.py`](src/dispersion/signals.py) | The lagged z-score, entry states, and regimes |
| [`trade.py`](src/dispersion/trade.py) | One dispersion structure and its decomposition |
| [`aggregate.py`](src/dispersion/aggregate.py) | Summaries, regimes, tails, cost stress, bootstrap |
| [`figures.py`](src/dispersion/figures.py) | The four figures |
| [`verify.py`](src/dispersion/verify.py) | Tolerance-based comparison of two result sets |
| [`pipeline.py`](src/dispersion/pipeline.py) | End-to-end orchestration |
| [`cli.py`](src/dispersion/cli.py) | Command-line interface |

## Reproducibility

`outputs/` holds the committed result set and the test suite checks the study still produces it:

```bash
make test    # includes the end-to-end reproduction check across all 13 tables
make verify  # re-run and print the largest difference found
```

**The study period ends on a fixed date on purpose.** Both providers extend their series every
trading day, so an open-ended sample would answer differently on every download. Freezing the end
is what lets a download taken months later reproduce the published numbers.

Bit-for-bit equality is not the target, and cannot be. IEEE 754 requires `+ - * / sqrt` to be
correctly rounded, so those agree everywhere, but it deliberately imposes no such requirement on
`exp`, `log` or `erf` — each platform's maths library may use its own approximation. The
linear-algebra routines add to this, since floating-point addition is not associative and a
different summation order gives a different last digit. Identical code on macOS and on Linux
therefore disagrees at around `1e-12` relative.

Two values are treated as agreeing when

```text
|a - b| <= atol + rtol * max(|a|, |b|)
```

with `rtol = 1e-9` and `atol = 1e-10`. The absolute term is not decoration. Several exported
columns are the *residual of an identity* whose correct value is zero — `attribution_error` and
`check` report how far the leg-by-leg decomposition missed the realised profit, and a correct run
puts them at `1e-12`. Comparing `1e-12` against `0.0` relatively gives a difference of 100%, so a
relative-only check fails a study that in fact reproduced perfectly. The floor sits seven orders
of magnitude below the smallest quantity this study reports, so it cannot mask a real difference.
Both behaviours are covered by tests in `tests/test_verify.py`.

Tests build their own return series rather than reading market data. The construction is worth a
mention: rows of a Hadamard matrix are mutually orthogonal, so combining a shared factor with
distinct idiosyncratic rows plants a pairwise correlation of *exactly* 0.40 — letting tests assert
equality rather than proximity.

## Data provenance

Neither input is redistributed here.

- **Cboe indexes** — under Cboe's terms of use. `scripts/fetch_cboe_data.py` downloads COR1M,
  COR3M, COR6M, DSPX, VIX, VXAPL, VXAZN, VXGOG, VXGS, and VXIBM from
  <https://cdn.cboe.com/api/global/us_indices/daily_prices/>.
- **Prices** — Yahoo Finance's public chart endpoint, via `scripts/fetch_price_data.py`. No API key,
  which is why it is used, but it is undocumented and can change. Any provider's export in the same
  `date,close` shape reads unchanged.

One data-quality note the loader handles explicitly: Cboe's dispersion index carries a literal
`0.000000` on 8 February 2018, between values of 14.05 and 20.92. That is a missing print rather than
an observation, so it is treated as missing and logged.

The reference basket is version-controlled, in `data/reference/basket.csv`.

## Limitations

- **Five constituent legs against a 500-name index.** The binding constraint is that Cboe publishes
  a volatility index for only five members. A real dispersion book holds dozens of legs, and its
  basis risk against the index is far smaller.
- **The realised-correlation basket is today's large caps**, so it carries survivorship bias:
  companies that left the index are absent from the earlier history.
- **Average pairwise correlation is not exactly what COR1M measures.** Cboe's construct is
  weight-and-volatility weighted over the top 50; historical weights are not available free, so the
  two are close cousins rather than the same number.
- Option prices are model marks, not quotes. No skew, no per-strike spread, no depth.
- 124 trades over thirteen years is a modest sample for a strategy with a long left tail.
- The capital denominator is a research normalisation.
- The specified thresholds were not fitted here, but they were chosen by someone, on some data.

**Results are research findings, not investment advice.**

## Origin

This reimplements a study that previously existed only as a Word report and an Excel workbook, both
preserved in `deliverables/` (kept out of version control as large binaries). That original was
explicit in its own Methodology sheet that apart from three volatility indexes, "most IVs, all
returns and all strategy P&L are deterministic fixtures" — its numbers came from generated data, and
its analysis code was never delivered.

This implementation replaces the generated inputs with observed ones: Cboe's published correlation
indexes for the implied side and real closing prices for the realised side. The numbers therefore
differ from the report's, because they now rest on market data.

## License

[MIT](LICENSE).
