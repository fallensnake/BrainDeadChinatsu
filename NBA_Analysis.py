#!/usr/bin/env python3
"""
Test the Burgi-Deng-Whelan (2026) favorite-longshot bias finding on NBA markets.

Core question: grouping by price-at-tip-off, what fraction of markets resolve YES, and does
that line up with the price? FLB = low-price contracts win LESS than their price implies and
high-price contracts win MORE (curve below the 45-deg line on the left, above on the right;
regression Y-P = a + b*P has a<0, b>0, joint F-test rejects a=b=0).

This is the fee-independent test (the paper's Figure 3 + Mincer-Zarnowitz regression). A
returns section is included at the end but is only a rough cut -- Kalshi's NBA-era fees differ
from the pre-2025 taker-only schedule the paper modeled, so don't lean on it.

Inputs: nba_markets_meta.csv (Phase 1) + nba_tipoff_prices.csv (Phase 2), joined on market_ticker.

  pip install pandas numpy statsmodels matplotlib
  python nba_flb_analysis.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
META   = "nba_markets_meta.csv"
PRICES = "nba_tipoff_prices.csv"

# Which price represents "the price at tip-off". mid = (bid+ask)/2 is the standing quote;
# last = last trade in the tip-off candle. We use mid, falling back to last when mid is missing.
PRICE_MID  = "price_at_start_mid"
PRICE_LAST = "price_at_start_last"

MIN_VOLUME = 1000      # paper kept contracts with >= $1000 total volume
MAX_SPREAD = 0.20      # paper dropped final bid-ask spreads > 20c
PLOT_BIN   = 0.02      # width of price bins for the calibration curve
OUT_PLOT   = "nba_flb_calibration.png"
OUT_TABLE  = "nba_flb_by_type.csv"

# ---------------- LOAD + CLEAN ----------------
meta = pd.read_csv(META)
prices = pd.read_csv(PRICES)
df = meta.merge(prices, on="market_ticker", how="inner")
print(f"joined rows: {len(df)}")

# settled only, with a real 0/1 outcome
df = df[df["result_binary"].isin([0, 1])].copy()
df["result_binary"] = df["result_binary"].astype(int)

# representative tip-off price
df["price"] = df[PRICE_MID]
if PRICE_LAST in df.columns:
    df["price"] = df["price"].fillna(df[PRICE_LAST])
df = df[df["price"].notna() & (df["price"] > 0) & (df["price"] < 1)]

# liquidity filters (paper's defaults)
if {"price_at_start_yes_ask", "price_at_start_yes_bid"}.issubset(df.columns):
    spread = df["price_at_start_yes_ask"] - df["price_at_start_yes_bid"]
    df = df[spread.isna() | (spread <= MAX_SPREAD)]
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
df = df[df["volume"] >= MIN_VOLUME]
if "n_candles_in_window" in df.columns:                      # drop stale tip-off prices
    df = df[df["n_candles_in_window"].fillna(0) > 0]

print(f"after cleaning: {len(df)} settled NBA markets")
print(df["market_type"].value_counts().to_string())

# ---------------- BUILD ANALYSIS FRAMES ----------------
# YES side only -> answers "P(resolves yes | yes-price)" and is used for the regression
# (paper uses Yes-only to avoid mechanically doubling the sample and shrinking std errors).
yes = pd.DataFrame({
    "event": df["event_ticker"], "market_type": df["market_type"],
    "price": df["price"], "y": df["result_binary"],
})
# SYMMETRIZED (Yes + No) -> fills out the calibration curve across the full price range,
# exactly as the paper's Figure 3 does. A market at yes-price P with outcome Y also gives a
# NO observation at price (1-P) with outcome (1-Y).
no = pd.DataFrame({
    "event": df["event_ticker"], "market_type": df["market_type"],
    "price": 1 - df["price"], "y": 1 - df["result_binary"],
})
both = pd.concat([yes, no], ignore_index=True)

# ---------------- CALIBRATION CURVE ----------------
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h

def calibration(frame, w=PLOT_BIN):
    edges = np.arange(0, 1 + w, w)
    frame = frame.assign(bin=pd.cut(frame["price"], edges, include_lowest=True))
    rows = []
    for b, g in frame.groupby("bin", observed=True):
        n, k = len(g), int(g["y"].sum())
        lo, hi = wilson(k, n)
        rows.append({"price_mid": b.mid, "n": n, "win_rate": k/n if n else np.nan,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)

cal = calibration(both)            # Fig 3 uses both sides
print("\n=== Calibration (both sides) ===")
print(cal.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

plt.figure(figsize=(7, 7))
plt.plot([0, 1], [0, 1], color="green", lw=1, label="45-deg (price = win rate)")
plt.fill_between(cal["price_mid"], cal["ci_lo"], cal["ci_hi"], alpha=0.2, color="gray")
plt.plot(cal["price_mid"], cal["win_rate"], color="crimson", marker="o", ms=3, label="NBA empirical")
plt.xlabel("Price at tip-off"); plt.ylabel("Fraction resolving YES")
plt.title("NBA: win rate vs. tip-off price"); plt.legend(); plt.xlim(0, 1); plt.ylim(0, 1)
plt.tight_layout(); plt.savefig(OUT_PLOT, dpi=130)
print(f"saved {OUT_PLOT}")

# ---------------- MINCER-ZARNOWITZ REGRESSION ----------------
# Y - P = a + b*P, clustered by event. Unbiased <=> a=b=0. FLB <=> a<0, b>0.
def mz(frame, label):
    d = frame.dropna(subset=["price", "y"])
    if len(d) < 30 or d["price"].nunique() < 5:
        print(f"  {label:18s} n={len(d):5d}  (too few to fit)")
        return None
    X = sm.add_constant(d["price"])
    m = sm.OLS(d["y"] - d["price"], X).fit(cov_type="cluster",
                                           cov_kwds={"groups": d["event"]})
    a, b = m.params["const"], m.params["price"]
    F = m.f_test("const = 0, price = 0")
    fp = float(np.ravel(F.pvalue))
    flb = "FLB pattern" if (a < 0 and b > 0) else "no FLB pattern"
    print(f"  {label:18s} n={len(d):5d}  a={a:+.3f}  b={b:+.3f}  "
          f"F p={fp:.4g}  -> {'REJECT unbiased' if fp < 0.05 else 'cannot reject'}, {flb}")
    return {"sample": label, "n": len(d), "alpha": a, "beta": b, "F_pvalue": fp}

print("\n=== Mincer-Zarnowitz (Yes contracts, clustered by event) ===")
results = [mz(yes, "ALL NBA")]
for mt, g in yes.groupby("market_type"):
    results.append(mz(g, mt))
pd.DataFrame([r for r in results if r]).to_csv(OUT_TABLE, index=False)
print(f"saved {OUT_TABLE}")

# ---------------- RETURNS (secondary; fee model is pre-2025, see header) ----------------
def returns_by_band(frame):
    P, Y = frame["price"], frame["y"]
    pre = (Y - P) / P
    fee = 0.07 * P * (1 - P)              # paper's pre-2025 taker fee -- NBA-era differs
    post = (Y - P - fee) / (P + fee)
    band = (P * 10).clip(upper=9).astype(int)
    out = pd.DataFrame({"band": band, "pre": pre, "post": post})
    return out.groupby("band").agg(n=("pre", "size"),
                                   pre_fee_return=("pre", "mean"),
                                   post_fee_return=("post", "mean"))

print("\n=== Returns by 10c band (both sides) -- ROUGH, fee caveat applies ===")
print(returns_by_band(both).to_string(float_format=lambda x: f"{x:.3f}"))
print(f"\noverall pre-fee mean return: {((both['y']-both['price'])/both['price']).mean():.3f}")