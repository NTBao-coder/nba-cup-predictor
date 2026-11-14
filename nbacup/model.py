"""Logistic Regression dự đoán xác suất đội A thắng đội B."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

COLS = FEATURES + ["LOC"]


def build() -> object:
    # with_mean=False + fit_intercept=False giữ mô hình đối xứng: P(A thắng B) = 1 - P(B thắng A)
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000),
    )


def evaluate(table: pd.DataFrame, test_season: str) -> dict:
    """Train trên các mùa trước, test trên `test_season` (chỉ góc nhìn chủ nhà)."""
    train = table[table.SEASON < test_season]
    test = table[(table.SEASON == test_season) & (table.LOC == 1)]
    m = build().fit(train[COLS], train.y)
    p = m.predict_proba(test[COLS])[:, 1]
    return {
        "test_season": test_season,
        "n_games": len(test),
        "accuracy": accuracy_score(test.y, p > 0.5),
        "log_loss": log_loss(test.y, p),
        "brier": brier_score_loss(test.y, p),
        "baseline_home_acc": test.y.mean(),
    }


def fit(table: pd.DataFrame):
    return build().fit(table[COLS], table.y)


def coefficients(model) -> pd.Series:
    lr = model[-1]
    return pd.Series(lr.coef_[0], index=COLS)


def win_prob(model, feats: dict, loc: float) -> float:
    x = pd.DataFrame([{**feats, "LOC": loc}])[COLS]
    return float(model.predict_proba(x)[0, 1])


def prob_matrix(model, states: pd.DataFrame, pair_features) -> dict:
    """P[(A, B, loc)] cho mọi cặp đội và loc ∈ {1, 0, -1}."""
    teams = list(states.index)
    rows, keys = [], []
    for a in teams:
        for b in teams:
            if a == b:
                continue
            f = pair_features(states.loc[a], states.loc[b])
            for loc in (1.0, 0.0, -1.0):
                rows.append({**f, "LOC": loc})
                keys.append((a, b, loc))
    p = model.predict_proba(pd.DataFrame(rows)[COLS])[:, 1]
    return dict(zip(keys, np.asarray(p)))
