# The Gauntlet

**A research engine built to kill trading strategies, not to find them.**

Eight strategy families went in. All eight came out dead. That is the result, and
publishing it is the point.

---

## Where the idea came from

There is a well-known genre of Minecraft video where someone writes a search
program that grinds through hundreds of millions of world seeds looking for one
absurd seed — a world where you can beat the game without ever moving, just by
looking around. The search works because the filter is exact. You can test a
hundred million seeds only if you can reject a bad one instantly and without
mercy.

I wanted to do that with trading strategies. Search a huge space, find the one
that shouldn't exist.

I got about a week into it before the actual problem showed up: **in trading, the
filter is the hard part.** Minecraft is deterministic — a seed either lets you win
without moving or it doesn't, and the checker is never wrong. Market backtests
lie in at least four different ways, and a search that runs a hundred million
strategies through a lying filter doesn't find alpha. It manufactures an
overfit winner, with a beautiful equity curve, every single time. The more you
search, the more certain you are to be fooled.

So I stopped building the search and built the filter instead. This repo is the
filter. The search is only worth writing once the filter can be trusted, and
trusting it means proving it — which is most of what's in here.

---

## The four firewalls

Each catches a different way a backtest lies.

| Firewall | File | The lie it catches |
|---|---|---|
| **Monte Carlo Permutation Test** | `gauntlet/mcpt.py` | "This edge is real" — when the same score is achievable on patternless noise |
| **Walk-forward + consistency gate** | `gauntlet/robustness.py` | "It's profitable overall" — when one lucky window carries the whole record |
| **Regime segmentation** | `gauntlet/robustness.py` | "It's a strategy" — when it's market beta in a costume, profitable only in up-markets |
| **Look-ahead audit** | `gauntlet/lookahead.py` | "The logic is clean" — when a decision silently changes if you hide the future |

Plus a fifth, sitting above all of them:

**Persistent trial ledger** (`gauntlet/ledger.py`) — every test ever run is
recorded, and a multiple-comparison correction is applied across the whole
history. This is the firewall against myself. Run enough strategies and one will
look brilliant by chance; the ledger makes that arithmetic impossible to ignore.

The MCPT is the interesting one. It shuffles price history thousands of times in
a way that **preserves drift and volatility but destroys temporal pattern**, then
re-runs the strategy on each noise universe and asks how often noise beats
reality. It's the only firewall that re-executes the strategy logic itself, and
it's specifically what kills long-biased strategies that were just riding market
drift — permutation keeps the drift, so if the strategy still works, there was
never a pattern there.

`gauntlet/core.py` is deliberately small and boring: costs charged on every turn,
and the no-look-ahead rule enforced in exactly one place so no strategy can cheat
through it by accident. If that file is wrong, everything downstream is wrong,
which is why it has its own dedicated proof.

---

## Everything is proven on known answers first

**30 known-answer checks across 8 test files.** No component is trusted on real
market data until it has been shown correct on synthetic data where the right
answer is known in advance.

```bash
python -m tests.test_core        # the no-look-ahead math and cost model
python -m tests.test_permute     # permutation preserves drift/vol, destroys autocorrelation
python -m tests.test_mcpt        # MCPT rejects luck, noise, and pure beta
python -m tests.test_lookahead   # the audit catches a deliberately planted future-leak
python -m tests.test_robustness  # the consistency gate and regime check
```

`test_permute.py` measures the destroy side rather than asserting it: autocorrelation
goes from -0.35 to -0.001 while the marginal statistics survive.

---

## Findings — 21 years, 13 liquid ETFs

*Provenance: every figure below was produced by the engine in this repo, run over daily
bars from 2005 to 2026 on 13 liquid US ETFs, fetched with `getdata.py` (yfinance). No
number here is an estimate, a quoted result, or carried over from anywhere else — each
one is the output of a test in this codebase, and re-runnable from a clean clone.*

### What died

| Family | Verdict | Cause of death |
|---|---|---|
| Momentum (MACD / 200MA) | Killed | MCPT p = 0.38; profitable only in up-markets — beta |
| Pairs / stat-arb (EFA-EEM) | Real, then killed | Passed MCPT at p = 0.002, then decayed after 2012 |
| Pairs, all 78 ETF combinations | Killed | Every "winner" vanished under multiple-comparison correction |
| Seasonality (sell-in-May, turn-of-month, day-of-week) | Killed | Pure beta — and one had a look-ahead bug the audit caught |
| Cross-sectional, time-series, overnight | Killed | See `MASTER_INDEX.md` |
| Next-day direction | Coin flip | Well-calibrated, zero resolution — honest and useless |

The overnight effect is the one I find most instructive: **real (+432%) and
completely uncapturable (-79% after transaction costs).** A true statistical
anomaly that a cost model erases. That gap between "real" and "tradeable" is the
entire reason `Costs` lives in `core.py` and not in an optional flag.

### What survived

Not direction. Risk and structure.

| Component | Result |
|---|---|
| Volatility forecasting (EWMA/GARCH, `probability/volatility.py`) | **0.65 correlation out-of-sample**, +27% over naive |
| Regime detection (HMM from scratch, `probability/regime.py`) | Identified every crisis — 2008, 2011, 2020, 2022 — from returns alone |
| Volatility targeting (`run_voltarget.py`) | Max drawdown -55% to -35%, steadier in 5 of 5 windows |
| Regime-conditioned forecasting (`run_regime_calibration.py`) | +14% sharper volatility forecasts |

This matches the literature, which is itself a check that the engine works: simple
price-based directional alpha in liquid ETFs is gone. Volatility clusters and is
predictable. Both are exactly what a correct engine should conclude.

---

## Read this file

**[`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)** documents what this engine has
*not* proven. The headline item: I attempted a clean synthetic positive control
for MCPT five separate times and could not build one. The file explains precisely
why — every synthetic edge I constructed leaked into the *marginal* statistics
that permutation preserves, rather than living purely in the temporal ordering it
destroys — and argues that a sixth attempt risked a misleading pass I wouldn't
understand, which is worse than an honest deferral.

I'd rather ship that file than quietly delete it. An engine whose stated purpose
is catching self-deception has to apply that standard to itself first.

---

## Setup

```bash
pip install -r requirements.txt
python getdata.py          # fetches ~21 years of daily ETF bars via yfinance
```

Market data is not committed — it is regenerated from the fetch scripts. Only the
computed result tables (`batch_results*.csv`) ship with the repo.

## Run strategies through the gauntlet

```bash
python run_macd.py          # momentum — watch it get killed as beta
python run_pairs.py         # stat-arb — a real signal that decayed
python scan_pairs.py        # all 78 ETF pairs, multiple-comparison corrected
python run_seasonality.py   # calendar effects
python run_voltarget.py     # volatility targeting — a real risk benefit
```

`MASTER_INDEX.md` maps every file to the question it answers.

## Results, visually

<!-- PNG goes here: ![MCPT permutation distribution](docs/mcpt.png) -->

*Charts pending.* The figure worth looking at first is the MCPT permutation
histogram — the noise distribution from a thousand shuffled universes, with the
real strategy's score marked on it. When the real score sits inside that
distribution, the edge was never there.

---

## Methods

Permutation testing after Timothy Masters; multiple-testing discipline after
Lopez de Prado; standard volatility and regime econometrics (GARCH, Hidden Markov
Models). The core math — permutation, Baum-Welch, forward-backward, Viterbi, and
the Brier decomposition — is implemented from scratch rather than imported, so
that each piece could be verified against a known answer.

## How this was built

AI-assisted, over roughly a week of iteration, and I want to be straightforward
about the split. I brought the framing (the Minecraft-seed search, and the
realization that the filter had to come before the search), the research direction,
the decision to structure it as adversarial firewalls rather than a scoring model,
and the calls on what to trust, what to kill, and what to defer. The
implementation was written with Claude. I directed it, tested it, and I can defend
every design decision in here — but I didn't type most of these lines, and it
would be dishonest to imply otherwise.

Author: **Frank Crilley** — Clemson University, Computer Science.

## License

MIT — see [LICENSE](LICENSE).
