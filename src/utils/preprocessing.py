"""共用前處理工具：集中實作兩個關鍵的方法學修正。

本模組將原本散落在 `scripts/05`、`06`、`07` 中重複的前處理邏輯抽出，
讓「避免資料洩漏」與「避免 look-ahead bias」這兩個修正：

1. 只有單一實作（DRY），不會出現各 script 不一致的情況；
2. 可被單元測試覆蓋（見 ``tests/test_preprocessing.py``）。

兩個修正：
- :func:`train_only_zscore` — Z-score 標準化僅使用訓練期的 mean/std，
  再套用至全期（含測試期），避免測試期統計資訊洩漏進標準化。
- :func:`add_next_day_target` — 以 ``shift(-1)`` 建立預測目標，
  確保用第 t 天的情緒特徵預測第 t+1 天的漲跌，避免 look-ahead bias。
"""

from typing import List, Optional, Tuple

import pandas as pd

# (raw 欄位, 對應的 z-score 欄位)
DEFAULT_ZSCORE_COLUMNS: List[Tuple[str, str]] = [
    ("ai_raw", "ai_zscore"),
    ("bi_raw", "bi_zscore"),
    ("pi_raw", "pi_zscore"),
]


def train_only_zscore(
    df: pd.DataFrame,
    test_start: str,
    date_col: str = "date",
    columns: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """以「僅訓練期統計量」對指定欄位做 Z-score 標準化。

    標準化所用的 mean / std 只由 ``date < test_start`` 的訓練期資料計算，
    再套用到全期資料，避免把測試期的分布資訊洩漏給標準化過程。

    Args:
        df: 含 ``date_col`` 與各 raw 欄位的 DataFrame。
        test_start: 測試期起始日（字串，如 ``"2025-01-01"``）；早於此日者為訓練期。
        date_col: 日期欄位名稱。
        columns: ``(raw_col, zscore_col)`` 配對清單；預設為 AI/BI/PI 三組。

    Returns:
        新的 DataFrame（不就地修改輸入），新增各 ``zscore_col`` 欄位。
    """
    df = df.copy()
    if columns is None:
        columns = DEFAULT_ZSCORE_COLUMNS

    train_mask = df[date_col] < pd.Timestamp(test_start)
    for raw_col, z_col in columns:
        train_mean = df.loc[train_mask, raw_col].mean()
        train_std = df.loc[train_mask, raw_col].std()
        # 防呆：訓練期無變異（或無資料）時避免除以零 / 產生 NaN
        if train_std == 0 or pd.isna(train_std):
            train_std = 1.0
        df[z_col] = (df[raw_col] - train_mean) / train_std

    return df


def add_next_day_target(
    df: pd.DataFrame,
    label_col: str = "trend_label",
    target_col: str = "target",
) -> pd.DataFrame:
    """以 ``shift(-1)`` 建立「隔日漲跌」預測目標，避免 look-ahead bias。

    令 ``target[t] = label[t+1]``，即用第 t 天的特徵預測第 t+1 天的趨勢。
    最後一列因無隔日標籤而被移除。

    Args:
        df: 含 ``label_col`` 的 DataFrame（應已依日期排序）。
        label_col: 當日趨勢標籤欄位名稱。
        target_col: 產生的目標欄位名稱。

    Returns:
        新的 DataFrame（不就地修改輸入），含整數型 ``target_col``，且已去除尾列。
    """
    df = df.copy()
    df[target_col] = df[label_col].shift(-1)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df[target_col] = df[target_col].astype(int)
    return df
