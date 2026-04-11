"""看漲指數（Bullish Index）建構模組。

將 PTT Stock 板的推噓文數據轉換為標準化的看漲指數。
公式：BI = ln[(1 + push_count) / (1 + boo_count)]
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils.config_loader import get_config, get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("bullish_builder")


class BullishBuilder:
    """看漲指數建構器。

    將 PTT 推噓文數據計算為看漲指數。

    Attributes:
        output_dir: 輸出目錄。
    """

    def __init__(self) -> None:
        self.output_dir = get_data_dir("processed")
        self.raw_dir = get_data_dir("raw/ptt")

    def build(self, aligned_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """建構看漲指數。

        Args:
            aligned_df: 已對齊到交易日且包含 push_count, boo_count 的 DataFrame。
                        若為 None 則從原始檔案載入。

        Returns:
            包含 date, push_total, boo_total, bi_raw, bi_zscore 的 DataFrame。
        """
        if aligned_df is None:
            aligned_df = self._load_and_aggregate()

        if aligned_df is None or len(aligned_df) == 0:
            logger.error("無 PTT 資料可建構指數")
            return pd.DataFrame()

        logger.info("建構看漲指數")

        # 確保有推噓文數欄位
        push_col = [c for c in aligned_df.columns if "push" in c.lower()]
        boo_col = [c for c in aligned_df.columns if "boo" in c.lower()]

        if push_col and boo_col:
            push_total = aligned_df[push_col[0]]
            boo_total = aligned_df[boo_col[0]]
        elif "push_total" in aligned_df.columns:
            push_total = aligned_df["push_total"]
            boo_total = aligned_df["boo_total"]
        else:
            logger.error("找不到推噓文數欄位")
            return pd.DataFrame()

        # BI = ln[(1 + push) / (1 + boo)]
        bi_raw = np.log((1 + push_total) / (1 + boo_total))

        # Z-score 標準化
        bi_zscore = scipy_stats.zscore(bi_raw, nan_policy="omit")

        result = pd.DataFrame({
            "date": aligned_df.index if isinstance(aligned_df.index, pd.DatetimeIndex)
            else pd.to_datetime(aligned_df.index),
            "push_total": push_total.values,
            "boo_total": boo_total.values,
            "bi_raw": bi_raw.values,
            "bi_zscore": bi_zscore,
        })

        self._save(result)
        logger.info(f"看漲指數建構完成，共 {len(result)} 筆")
        return result

    def _load_and_aggregate(self) -> Optional[pd.DataFrame]:
        """從原始檔案載入並按日彙總 PTT 資料。

        Returns:
            按日彙總的 DataFrame 或 None。
        """
        raw_path = self.raw_dir / "ptt_stock_articles.csv"
        if not raw_path.exists():
            return None

        df = pd.read_csv(raw_path, encoding="utf-8-sig")

        if "date" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"])

        # 按日彙總推噓文數
        daily = df.groupby("date").agg(
            push_total=("push_count", "sum"),
            boo_total=("boo_count", "sum"),
            article_count=("title", "count"),
        )

        return daily

    def _save(self, df: pd.DataFrame) -> None:
        """儲存看漲指數。

        Args:
            df: 要儲存的 DataFrame。
        """
        output_path = self.output_dir / "bullish_index.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"已儲存至 {output_path}")
