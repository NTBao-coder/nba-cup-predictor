"""Lấy dữ liệu từ NBA Stats API (qua thư viện nba_api) và cache ra CSV."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import iststandings, leaguegamefinder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _fetch_team_rows(season: str) -> pd.DataFrame:
    """Mỗi dòng = 1 đội trong 1 trận (mùa giải chính thức)."""
    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
        timeout=60,
    ).get_data_frames()[0]
    time.sleep(0.8)  # tránh bị stats.nba.com chặn
    return df


def _to_games(rows: pd.DataFrame, season: str) -> pd.DataFrame:
    """Gộp 2 dòng (chủ nhà / khách) thành 1 dòng mỗi trận."""
    rows = rows.dropna(subset=["WL"])
    home = rows[rows["MATCHUP"].str.contains("vs.")]
    away = rows[rows["MATCHUP"].str.contains("@")]
    cols = ["GAME_ID", "TEAM_ABBREVIATION", "PTS"]
    g = home[["GAME_ID", "GAME_DATE"] + cols[1:]].merge(
        away[cols], on="GAME_ID", suffixes=("_H", "_A")
    )
    g = g.rename(columns={
        "TEAM_ABBREVIATION_H": "HOME", "TEAM_ABBREVIATION_A": "AWAY",
        "PTS_H": "HOME_PTS", "PTS_A": "AWAY_PTS",
    })
    g["SEASON"] = season
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def _read_games_csv(path: Path) -> pd.DataFrame:
    # GAME_ID phải đọc dạng chuỗi, nếu không "0022600030" thành 22600030
    return pd.read_csv(path, parse_dates=["GAME_DATE"], dtype={"GAME_ID": str})


def _warn_stale(path: Path, err: Exception) -> None:
    print(f"    ! NBA API lỗi ({type(err).__name__}), dùng dữ liệu cache cũ: {path.name}")


def load_games(season: str, refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / f"games_{season}.csv"
    if path.exists() and not refresh:
        return _read_games_csv(path)
    try:
        games = _to_games(_fetch_team_rows(season), season)
    except Exception as err:  # stats.nba.com hay chặn hoặc timeout
        if not path.exists():
            raise
        _warn_stale(path, err)
        return _read_games_csv(path)
    games.to_csv(path, index=False)
    return games


def load_all_games(seasons: list[str], current: str, refresh: bool = False) -> pd.DataFrame:
    """Mùa hiện tại luôn tải lại (refresh) vì có trận mới mỗi ngày."""
    frames = []
    for s in seasons:
        print(f"  - tải mùa {s} ...")
        frames.append(load_games(s, refresh=refresh or s == current))
    return pd.concat(frames, ignore_index=True)


def read_cup_standings_csv(path: Path) -> pd.DataFrame:
    """Đọc bảng NBA Cup đã cache. Cột gameId1..4 phải là chuỗi để giữ số 0 ở đầu,
    nếu không cup_group_games sẽ không khớp được trận nào với bảng kết quả."""
    return pd.read_csv(path, dtype={f"gameId{i}": str for i in range(1, 5)})


def load_cup_standings(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"cup_standings_{season}.csv"
    try:
        df = iststandings.ISTStandings(season=season, timeout=60).get_data_frames()[0]
    except Exception as err:  # stats.nba.com hay chặn hoặc timeout
        if not path.exists():
            raise
        _warn_stale(path, err)
        return read_cup_standings_csv(path)
    df.to_csv(path, index=False)
    return df


def cup_group_games(standings: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Danh sách 60 trận vòng bảng (đã đá + chưa đá) của NBA Cup."""
    margin = games.set_index("GAME_ID").eval("HOME_PTS - AWAY_PTS")
    out = {}
    for _, r in standings.iterrows():
        for i in range(1, 5):
            gid = r[f"gameId{i}"]
            if not isinstance(gid, str) or gid in out:
                continue
            team, opp = r["teamAbbreviation"], r[f"opponentTeamAbbreviation{i}"]
            home, away = (team, opp) if r[f"location{i}"] == "H" else (opp, team)
            final = int(r[f"gameStatus{i}"]) == 3 and gid in margin.index
            out[gid] = {
                "GAME_ID": gid, "HOME": home, "AWAY": away, "FINAL": final,
                "HOME_MARGIN": float(margin[gid]) if final else None,
            }
    return pd.DataFrame(out.values())
