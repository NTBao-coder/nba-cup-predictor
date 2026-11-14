"""Dự đoán nhà vô địch NBA Cup bằng Logistic Regression + mô phỏng Monte Carlo.

Chạy:  python predict.py            (mặc định mùa 2026-27, 20.000 lần mô phỏng)
       python predict.py --sims 50000 --refresh
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nbacup import data, features, model, simulate


def season_str(start: int) -> str:
    return f"{start}-{str(start + 1)[-2:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--history", type=int, default=6, help="số mùa lịch sử để train")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--refresh", action="store_true", help="tải lại cả dữ liệu các mùa cũ")
    args = ap.parse_args()

    start = int(args.season[:4])
    seasons = [season_str(y) for y in range(start - args.history, start + 1)]

    print("1) Lấy dữ liệu từ NBA API")
    games = data.load_all_games(seasons, args.season, refresh=args.refresh)
    standings = data.load_cup_standings(args.season)
    print(f"   {len(games)} trận, {games.SEASON.nunique()} mùa")

    print("2) Tạo feature + kiểm định mô hình (train mùa cũ, test mùa gần nhất)")
    table = features.training_table(games)
    table = table[table.SEASON != seasons[0]]  # mùa đầu tiên không có dữ liệu mùa trước
    ev = model.evaluate(table, seasons[-2])
    print(f"   Test {ev['test_season']} ({ev['n_games']} trận): accuracy={ev['accuracy']:.3f} "
          f"| log-loss={ev['log_loss']:.3f} | brier={ev['brier']:.3f} "
          f"| baseline 'chủ nhà luôn thắng'={ev['baseline_home_acc']:.3f}")

    print("3) Train mô hình cuối trên toàn bộ dữ liệu")
    m = model.fit(table)
    print("   Hệ số (đã chuẩn hoá):", model.coefficients(m).round(3).to_dict())

    print(f"4) Mô phỏng NBA Cup {args.season} ({args.sims} lần)")
    states = features.current_states(games, args.season)
    P = model.prob_matrix(m, states, features.pair_features)
    gg = data.cup_group_games(standings, games)
    print(f"   Vòng bảng: {gg.FINAL.sum()} trận đã đá, {(~gg.FINAL).sum()} trận còn lại")
    res = simulate.simulate(standings, gg, P, n_sims=args.sims)

    out = Path("output"); out.mkdir(exist_ok=True)
    res.to_csv(out / f"cup_prediction_{args.season}.csv")
    show = res.copy()
    for c in simulate.STAGES:
        show[c] = (show[c] * 100).map("{:5.1f}%".format)
    pd.set_option("display.width", 140)
    print("\nXác suất (nhất bảng / vào tứ kết / bán kết / chung kết / VÔ ĐỊCH):")
    print(show.head(15).to_string())
    top = res.index[0]
    print(f"\n=> Ứng viên số 1: {top} ({res.loc[top, 'champion']:.1%} vô địch)")


if __name__ == "__main__":
    main()
