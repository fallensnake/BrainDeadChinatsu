#!/usr/bin/env python3
"""
NBA Kalshi favorite-longshot dashboard.

Pick any players, teams, and stat categories, then it builds the research-paper curve
(win rate vs price) for that selection, ranks the best positive-edge prices, and lists the
specific markets with the biggest realized edges.

  pip install streamlit pandas numpy matplotlib
  streamlit run nba_dashboard.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

CSV = "nba_markets_enriched.csv"

PLAYER_SERIES = {"KXNBAPTS","KXNBAREB","KXNBAAST","KXNBA3PT","KXNBASTL","KXNBABLK",
                 "KXNBA2D","KXNBA3D","KXNBAFTM"}
CATEGORY = {
    "KXNBAPTS":"Points","KXNBAREB":"Rebounds","KXNBAAST":"Assists","KXNBA3PT":"Threes Made",
    "KXNBASTL":"Steals","KXNBABLK":"Blocks","KXNBAFTM":"Free Throws Made",
    "KXNBA2D":"Double-Double","KXNBA3D":"Triple-Double",
    "KXNBASPREAD":"Spread (Full Game)","KXNBA1HSPREAD":"Spread (1st Half)",
    "KXNBA2HSPREAD":"Spread (2nd Half)","KXNBA1QSPREAD":"Spread (Q1)","KXNBA2QSPREAD":"Spread (Q2)",
    "KXNBA3QSPREAD":"Spread (Q3)","KXNBA4QSPREAD":"Spread (Q4)",
    "KXNBATOTAL":"Game Total","KXNBA1HTOTAL":"Total (1st Half)","KXNBA2HTOTAL":"Total (2nd Half)",
    "KXNBA1QTOTAL":"Total (Q1)","KXNBA2QTOTAL":"Total (Q2)","KXNBA3QTOTAL":"Total (Q3)",
    "KXNBA4QTOTAL":"Total (Q4)","KXNBATEAMTOTAL":"Team Total",
    "KXNBAGAME":"Game Winner","KXNBA1HWINNER":"Winner (1st Half)","KXNBA2HWINNER":"Winner (2nd Half)",
    "KXNBA1QWINNER":"Winner (Q1)","KXNBA2QWINNER":"Winner (Q2)","KXNBA3QWINNER":"Winner (Q3)",
    "KXNBA4QWINNER":"Winner (Q4)","KXNBAOVERTIME":"Overtime",
    "KXNBAH2HPTS":"H2H Points","KXNBAH2H3PT":"H2H Threes","KXNBAH2HPRA":"H2H PRA",
    "KXNBAH2HBENCHPTS":"H2H Bench Points","KXNBAH2HTEAM3PT":"H2H Team Threes",
}
MONTHS = {m: i+1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}

@st.cache_data(show_spinner="Parsing markets...")
def load_data(path):
    df = pd.read_csv(path)
    df["category"] = df["series_ticker"].map(lambda s: CATEGORY.get(s, s.replace("KXNBA","")))
    df["is_pp"] = df["series_ticker"].isin(PLAYER_SERIES)

    # player + the player's team (leading 3 chars of the player code; NYJ is a KAT typo for NYK)
    sub = df["yes_sub_title"].astype(str)
    df["player"] = np.where(df["is_pp"], sub.str.split(":").str[0].str.strip(), None)
    df["line"]   = np.where(df["is_pp"] & sub.str.contains(":"), sub.str.split(":").str[1].str.strip(), "")
    pcode = df["market_ticker"].str.split("-").str[2].fillna("")
    df["player_team"] = np.where(df["is_pp"], pcode.str[:3].replace({"NYJ":"NYK"}), None)

    # the two teams in the game (event ticker = ...-YYMMMDD + away(3) + home(3))
    suf = df["event_ticker"].str.split("-", n=1).str[1].fillna("")
    df["away"], df["home"] = suf.str[7:10], suf.str[10:13]

    # readable date from the event code
    yy = pd.to_numeric(suf.str[0:2], errors="coerce")
    mm = suf.str[2:5].map(MONTHS)
    dd = pd.to_numeric(suf.str[5:7], errors="coerce")
    df["date"] = pd.to_datetime(dict(year=2000+yy, month=mm, day=dd), errors="coerce")

    df["price"] = df["price_at_start_mid"].fillna(df["price_at_start_last"])
    df["result_binary"] = pd.to_numeric(df["result_binary"], errors="coerce")
    return df

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-h, c+h

def usable(df):
    return df[df["result_binary"].isin([0,1]) & df["price"].between(0,1,inclusive="neither")].copy()

def per_cent(df):
    """Symmetrized (Yes+No) win rate by integer cent, plus edge vs price and significance."""
    d = usable(df)
    both = pd.concat([
        pd.DataFrame({"price": d["price"],     "y": d["result_binary"].astype(int)}),
        pd.DataFrame({"price": 1 - d["price"], "y": 1 - d["result_binary"].astype(int)}),
    ], ignore_index=True)
    both["c"] = (both["price"]*100).round().clip(1,99).astype(int)
    g = both.groupby("c")["y"].agg(n="size", emp="mean").reset_index()
    g["kalshi"] = g["c"]/100
    g["edge"] = g["emp"] - g["kalshi"]
    g[["lo","hi"]] = g.apply(lambda r: pd.Series(wilson(r["emp"]*r["n"], r["n"])), axis=1)
    g["z"] = g["edge"] / np.sqrt(g["kalshi"]*(1-g["kalshi"])/g["n"])
    return g

# ----------------------------------------------------------------------------- UI
st.set_page_config(page_title="NBA Kalshi FLB", layout="wide")
st.title("NBA Kalshi — Favorite–Longshot Explorer")

df = load_data(CSV)

players = sorted(df["player"].dropna().unique())
teams   = sorted(set(df["away"].dropna()) | set(df["home"].dropna()) - {""})
cats    = sorted(df["category"].dropna().unique())

with st.sidebar:
    st.header("Filters")
    sel_players = st.multiselect("Players (type to search)", players)
    sel_teams   = st.multiselect("Teams", teams)
    sel_cats    = st.multiselect("Market categories", cats,
                                 help="Leave empty for all. e.g. Points, Rebounds, Assists.")
    st.divider()
    min_n = st.slider("Min sample size per price (edge significance)", 5, 200, 25, 5)
    top_n = st.slider("Rows in tables", 5, 50, 20, 5)

# apply filters: a market passes if it matches every active filter
mask = pd.Series(True, index=df.index)
if sel_cats:
    mask &= df["category"].isin(sel_cats)
if sel_players:
    mask &= df["player"].isin(sel_players)          # team markets (no player) excluded
if sel_teams:
    team_mask = (df["is_pp"] & df["player_team"].isin(sel_teams)) | \
                (~df["is_pp"] & (df["away"].isin(sel_teams) | df["home"].isin(sel_teams)))
    mask &= team_mask
sub = df[mask]
u = usable(sub)

# ----- summary -----
who = ", ".join(sel_players) if sel_players else (", ".join(sel_teams) if sel_teams else "All NBA")
st.subheader(f"Selection: {who}")
if u.empty:
    st.warning("No settled markets with a usable tip-off price match these filters. Loosen them.")
    st.stop()

ret = np.where(u["result_binary"]==1, 1-u["price"], u["price"]).mean()  # avg gross on correct side
c1, c2, c3, c4 = st.columns(4)
c1.metric("Settled markets", f"{len(u):,}")
c2.metric("Distinct players", f"{u['player'].dropna().nunique():,}")
c3.metric("Categories", f"{u['category'].nunique()}")
c4.metric("Avg pre-fee return", f"{((u['result_binary']-u['price'])/u['price']).mean():+.1%}")

# ----- FLB curve -----
st.markdown("### Favorite–longshot curve")
g = per_cent(sub)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0,100],[0,100], color="green", lw=1, label="45° (efficient)")
ax.fill_between(g["c"], g["lo"]*100, g["hi"]*100, color="gray", alpha=0.25, label="95% CI")
ax.plot(g["c"], g["emp"]*100, color="crimson", lw=1.2, label="Empirical")
ax.set_xlabel("Price (cents)"); ax.set_ylabel("Win %")
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.legend()
left, right = st.columns([3, 2])
left.pyplot(fig)
right.markdown("**How to read it**\n\nBelow the green line at low prices and above it at high prices = "
               "favorite–longshot bias. Where the red line sits **above** green, that price is "
               "underpriced — a positive edge for buying YES (before fees).")

# ----- best positive edges -----
st.markdown("### Best positive-edge prices")
st.caption("Cents where the outcome won MORE than the price implied (buy YES = +EV pre-fee). "
           f"Filtered to n ≥ {min_n}; ✓ = clears 95% significance (|z|>1.96).")
edges = g[(g["n"]>=min_n) & (g["edge"]>0)].sort_values("edge", ascending=False).head(top_n).copy()
if edges.empty:
    st.info("No positive-edge prices clear the sample-size floor for this selection. Lower Min sample size.")
else:
    show = pd.DataFrame({
        "Price (¢)": edges["c"],
        "Kalshi implied": edges["kalshi"].map("{:.0%}".format),
        "Actual win rate": edges["emp"].map("{:.1%}".format),
        "Edge": edges["edge"].map("{:+.1%}".format),
        "z": edges["z"].map("{:+.2f}".format),
        "Significant": np.where(edges["z"].abs()>1.96, "✓", ""),
        "n": edges["n"],
    })
    st.dataframe(show, hide_index=True, width='stretch')

# ----- best recorded markets -----
st.markdown("### Best recorded markets (biggest realized edges)")
st.caption("Individual markets where the correct side paid off most vs its price — the biggest "
           "mispricings that actually hit. Realized = gross return on the winning side, pre-fee.")
m = u.copy()
m["realized"] = np.where(m["result_binary"]==1, 1-m["price"], m["price"])
m["outcome"]  = np.where(m["result_binary"]==1, "YES", "NO")
best = m.sort_values("realized", ascending=False).head(top_n)
tbl = pd.DataFrame({
    "Date": best["date"].dt.strftime("%Y-%m-%d"),
    "Category": best["category"],
    "Market": best["title"],
    "Matchup": best["away"] + " @ " + best["home"],
    "Price": best["price"].map("{:.0%}".format),
    "Result": best["outcome"],
    "Realized edge": best["realized"].map("{:+.0%}".format),
})
st.dataframe(tbl, hide_index=True, width='stretch')