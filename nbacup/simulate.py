"""Mô phỏng Monte Carlo toàn bộ NBA Cup (vòng bảng còn lại + vòng knock-out).

Thể thức NBA Cup:
  - 30 đội, 6 bảng x 5 đội (3 bảng Đông, 3 bảng Tây), mỗi đội đá 4 trận vòng bảng.
  - Mỗi miền: 3 đội nhất bảng (hạt giống 1-3) + 1 wild card (hạt giống 4) vào tứ kết.
  - Tứ kết: 1 vs 4, 2 vs 3, sân nhà của hạt giống cao hơn.
  - Bán kết & chung kết: sân trung lập (Las Vegas).
Xếp hạng bảng: số trận thắng -> đối đầu -> hiệu số -> ngẫu nhiên.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import NormalDist

import numpy as np
import pandas as pd

SIGMA = 12.5  # độ lệch chuẩn cách biệt điểm một trận NBA
STAGES = ["group_win", "knockout", "semis", "final", "champion"]


def _rank(teams, wins, diff, h2h, rng):
    """Xếp hạng danh sách đội theo luật tiebreak."""
    noise = {t: rng.random() for t in teams}
    by_wins = defaultdict(list)
    for t in teams:
        by_wins[wins[t]].append(t)
    out = []
    for w in sorted(by_wins, reverse=True):
        tied = by_wins[w]
        h = {t: sum(h2h.get((t, o), 0) for o in tied if o != t) for t in tied}
        out += sorted(tied, key=lambda t: (h[t], diff[t], noise[t]), reverse=True)
    return out


def simulate(standings: pd.DataFrame, group_games: pd.DataFrame, P: dict,
             n_sims: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nd = NormalDist()
    groups = standings.groupby("istGroup").teamAbbreviation.apply(list).to_dict()
    conf = standings.set_index("teamAbbreviation").conference.to_dict()

    done = group_games[group_games.FINAL]
    todo = group_games[~group_games.FINAL].reset_index(drop=True)
    # Cách biệt mô phỏng: N(mu, SIGMA) với mu sao cho P(cách biệt > 0) = p của mô hình
    mu = np.array([SIGMA * nd.inv_cdf(min(max(P[(h, a, 1.0)], 1e-6), 1 - 1e-6))
                   for h, a in zip(todo.HOME, todo.AWAY)])
    sim_margins = rng.normal(mu, SIGMA, size=(n_sims, len(todo)))

    counts = {s: Counter() for s in STAGES}
    for k in range(n_sims):
        wins, diff, h2h = Counter(), Counter(), Counter()
        results = list(zip(done.HOME, done.AWAY, done.HOME_MARGIN)) + \
            list(zip(todo.HOME, todo.AWAY, sim_margins[k]))
        for h, a, m in results:
            w, l = (h, a) if m > 0 else (a, h)
            wins[w] += 1
            h2h[(w, l)] += 1
            diff[h] += m
            diff[a] -= m

        seeds = {"East": [], "West": []}
        runners = {"East": [], "West": []}
        for g, teams in groups.items():
            r = _rank(teams, wins, diff, h2h, rng)
            counts["group_win"][r[0]] += 1
            seeds[conf[r[0]]].append(r[0])
            runners[conf[r[0]]] += r[1:]

        finalists = []
        for c in ("East", "West"):
            winners = sorted(seeds[c], key=lambda t: (wins[t], diff[t], rng.random()), reverse=True)
            wc = max(runners[c], key=lambda t: (wins[t], diff[t], rng.random()))
            s = winners + [wc]
            for t in s:
                counts["knockout"][t] += 1
            semi = []
            for hi, lo in ((s[0], s[3]), (s[1], s[2])):
                semi.append(hi if rng.random() < P[(hi, lo, 1.0)] else lo)
            for t in semi:
                counts["semis"][t] += 1
            a, b = semi
            finalists.append(a if rng.random() < P[(a, b, 0.0)] else b)
        for t in finalists:
            counts["final"][t] += 1
        a, b = finalists
        counts["champion"][a if rng.random() < P[(a, b, 0.0)] else b] += 1

    st = standings.set_index("teamAbbreviation")
    df = pd.DataFrame({s: pd.Series(counts[s]) / n_sims for s in STAGES}).reindex(st.index).fillna(0)
    df.insert(0, "group", st.istGroup)
    df.insert(1, "record", st.wins.astype(str) + "-" + st.losses.astype(str))
    return df.sort_values("champion", ascending=False)
