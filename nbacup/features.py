"""Tạo feature cho mỗi trận, chỉ dùng thông tin CÓ TRƯỚC trận đấu (không rò rỉ dữ liệu).

Đầu mùa chỉ số rất nhiễu (1-2 trận), nên chỉ số mùa hiện tại được "co" (shrink)
về phía chỉ số mùa trước:  blended = (n*hiện_tại + K*mùa_trước) / (n + K)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_SHRINK = 12  # sau ~12 trận, mùa hiện tại và mùa trước có trọng số bằng nhau
FEATURES = ["d_margin", "d_winpct", "d_last10", "d_rest"]


def _long(games: pd.DataFrame) -> pd.DataFrame:
    """Mỗi trận -> 2 dòng, 1 dòng cho mỗi đội."""
    h = games.assign(TEAM=games.HOME, OPP=games.AWAY, MARGIN=games.HOME_PTS - games.AWAY_PTS)
    a = games.assign(TEAM=games.AWAY, OPP=games.HOME, MARGIN=games.AWAY_PTS - games.HOME_PTS)
    lg = pd.concat([h, a], ignore_index=True)[["SEASON", "GAME_ID", "GAME_DATE", "TEAM", "MARGIN"]]
    lg["WIN"] = (lg.MARGIN > 0).astype(float)
    return lg.sort_values(["TEAM", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def _prior(lg: pd.DataFrame) -> pd.DataFrame:
    """Chỉ số cả mùa trước của mỗi đội (map sang mùa kế tiếp)."""
    seasons = sorted(lg.SEASON.unique())
    nxt = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}
    p = lg.groupby(["SEASON", "TEAM"]).agg(P_MARGIN=("MARGIN", "mean"), P_WIN=("WIN", "mean")).reset_index()
    p["P_MARGIN"] *= 0.7  # hồi quy về trung bình giữa các mùa (đổi đội hình, chấn thương...)
    p["P_WIN"] = 0.5 + 0.7 * (p.P_WIN - 0.5)
    p["SEASON"] = p.SEASON.map(nxt)
    return p.dropna(subset=["SEASON"])


def team_states(games: pd.DataFrame) -> pd.DataFrame:
    """Trạng thái của mỗi đội NGAY TRƯỚC mỗi trận."""
    lg = _long(games)
    g = lg.groupby(["SEASON", "TEAM"])
    lg["N"] = g.cumcount()
    lg["CUR_MARGIN"] = g.MARGIN.transform(lambda s: s.shift().expanding().mean())
    lg["CUR_WIN"] = g.WIN.transform(lambda s: s.shift().expanding().mean())
    lg["LAST10"] = g.MARGIN.transform(lambda s: s.shift().rolling(10, min_periods=1).mean())
    lg["REST"] = g.GAME_DATE.diff().dt.days.clip(upper=4).fillna(4)
    return _blend(lg.merge(_prior(lg), on=["SEASON", "TEAM"], how="left"))


def _blend(lg: pd.DataFrame) -> pd.DataFrame:
    lg["P_MARGIN"] = lg.P_MARGIN.fillna(0.0)
    lg["P_WIN"] = lg.P_WIN.fillna(0.5)
    n = lg.N
    lg["S_MARGIN"] = (n * lg.CUR_MARGIN.fillna(0) + K_SHRINK * lg.P_MARGIN) / (n + K_SHRINK)
    lg["S_WIN"] = (n * lg.CUR_WIN.fillna(0) + K_SHRINK * lg.P_WIN) / (n + K_SHRINK)
    lg["S_LAST10"] = np.where(n > 0, lg.LAST10, lg.P_MARGIN)
    return lg


def current_states(games: pd.DataFrame, season: str) -> pd.DataFrame:
    """Trạng thái hiện tại (sau trận gần nhất) của mỗi đội ở mùa `season`."""
    lg = _long(games)
    cur = lg[lg.SEASON == season]
    agg = cur.groupby("TEAM").agg(
        N=("MARGIN", "size"), CUR_MARGIN=("MARGIN", "mean"), CUR_WIN=("WIN", "mean"),
        LAST10=("MARGIN", lambda s: s.tail(10).mean()),
    ).reset_index()
    agg["SEASON"] = season
    agg["REST"] = 2.0  # giả định nghỉ ngơi như nhau cho các trận mô phỏng
    st = _blend(agg.merge(_prior(lg), on=["SEASON", "TEAM"], how="left"))
    return st.set_index("TEAM")


def pair_features(a: pd.Series | pd.DataFrame, b: pd.Series | pd.DataFrame) -> dict:
    return {
        "d_margin": a["S_MARGIN"] - b["S_MARGIN"],
        "d_winpct": a["S_WIN"] - b["S_WIN"],
        "d_last10": a["S_LAST10"] - b["S_LAST10"],
        "d_rest": a["REST"] - b["REST"],
    }


def training_table(games: pd.DataFrame) -> pd.DataFrame:
    """Mỗi trận -> 1 dòng góc nhìn chủ nhà, cộng thêm dòng 'lật' (góc nhìn đội khách)
    để mô hình đối xứng và có thể dự đoán cả trận sân trung lập (LOC = 0)."""
    st = team_states(games).set_index(["GAME_ID", "TEAM"])
    h = st.loc[list(zip(games.GAME_ID, games.HOME))].reset_index(drop=True)
    a = st.loc[list(zip(games.GAME_ID, games.AWAY))].reset_index(drop=True)
    X = pd.DataFrame(pair_features(h, a))
    X["LOC"] = 1.0
    y = (games.HOME_PTS > games.AWAY_PTS).astype(int).to_numpy()
    flipped = -X
    both = pd.concat([X, flipped], ignore_index=True)
    both["y"] = np.concatenate([y, 1 - y])
    both["SEASON"] = np.concatenate([games.SEASON.to_numpy()] * 2)
    return both
