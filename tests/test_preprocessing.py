"""前處理修正的單元測試。

鎖定兩個關鍵方法學修正，避免未來回歸（regression）：
1. Z-score 只用訓練期統計量（避免資料洩漏）
2. 預測目標 shift(-1)（避免 look-ahead bias）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.preprocessing import train_only_zscore, add_next_day_target


def _make_df():
    """5 個訓練日 + 3 個測試日；測試期數值刻意極端。"""
    dates = pd.to_datetime([
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2025-01-01", "2025-01-02", "2025-01-03",
    ])
    return pd.DataFrame({
        "date": dates,
        "ai_raw": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0, 2000.0, 3000.0],
        "bi_raw": [10.0, 12.0, 14.0, 16.0, 18.0, -500.0, -600.0, -700.0],
        "pi_raw": [0.0, 1.0, 0.0, 1.0, 0.0, 99.0, 99.0, 99.0],
        "trend_label": [1, -1, 0, 1, -1, 1, 0, -1],
    })


def test_zscore_uses_only_train_statistics():
    """訓練期 z-score 應由訓練期 mean/std 計算。"""
    df = _make_df()
    out = train_only_zscore(df, test_start="2025-01-01")

    train = df[df["date"] < "2025-01-01"]
    mu, sigma = train["ai_raw"].mean(), train["ai_raw"].std()  # pandas std: ddof=1
    expected_train_z = (train["ai_raw"].values - mu) / sigma

    np.testing.assert_allclose(out["ai_zscore"].values[:5], expected_train_z)


def test_zscore_no_leakage_from_test_period():
    """改變測試期的 raw 值，不應改變訓練期的 z-score（無洩漏）。"""
    df = _make_df()
    out1 = train_only_zscore(df, test_start="2025-01-01")

    df2 = df.copy()
    df2.loc[df2["date"] >= "2025-01-01", "ai_raw"] *= 999  # 測試期亂動
    out2 = train_only_zscore(df2, test_start="2025-01-01")

    np.testing.assert_allclose(
        out1["ai_zscore"].values[:5], out2["ai_zscore"].values[:5]
    )


def test_train_portion_is_standardized():
    """標準化後，訓練期切片的 mean≈0、std≈1。"""
    df = _make_df()
    out = train_only_zscore(df, test_start="2025-01-01")
    train_z = out.loc[out["date"] < "2025-01-01", "ai_zscore"]
    assert abs(train_z.mean()) < 1e-9
    assert abs(train_z.std() - 1.0) < 1e-9


def test_zero_variance_guard():
    """訓練期無變異時不應產生 inf / NaN。"""
    df = _make_df()
    df["ai_raw"] = [7.0] * 8  # 全常數
    out = train_only_zscore(df, test_start="2025-01-01")
    assert np.isfinite(out["ai_zscore"].values).all()


def test_train_only_zscore_does_not_mutate_input():
    df = _make_df()
    cols_before = set(df.columns)
    _ = train_only_zscore(df, test_start="2025-01-01")
    assert set(df.columns) == cols_before  # 未就地新增欄位


def test_target_is_next_day_label():
    """target[t] 應等於 label[t+1]，且尾列被移除。"""
    df = _make_df()
    out = add_next_day_target(df)

    original_labels = df["trend_label"].values
    assert len(out) == len(df) - 1
    np.testing.assert_array_equal(out["target"].values, original_labels[1:])


def test_target_is_integer():
    df = _make_df()
    out = add_next_day_target(df)
    assert np.issubdtype(out["target"].dtype, np.integer)
