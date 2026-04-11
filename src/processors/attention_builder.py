"""關注度指數（Attention Index）建構模組。

將 Google Trends 原始資料轉換為標準化的關注度指數。
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA

from src.utils.config_loader import get_config, get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("attention_builder")


class AttentionBuilder:
    """關注度指數建構器。

    將 Google Trends 多關鍵字資料彙總為單一關注度指數。

    Attributes:
        aggregation: 彙總方式（'sum', 'max', 'pca'）。
        output_dir: 輸出目錄。
    """

    def __init__(self) -> None:
        config = get_config()
        self.aggregation: str = config["index_construction"]["attention_index"][
            "aggregation"
        ]
        self.output_dir = get_data_dir("processed")
        self.raw_dir = get_data_dir("raw/google_trends")

    def build(self, aligned_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """建構關注度指數。

        Args:
            aligned_df: 已對齊到交易日的 Google Trends DataFrame。
                        若為 None 則從原始檔案載入。

        Returns:
            包含 date, ai_raw, ai_zscore 的 DataFrame。
        """
        if aligned_df is None:
            aligned_df = self._load_raw_data()

        if aligned_df is None or len(aligned_df) == 0:
            logger.error("無 Google Trends 資料可建構指數")
            return pd.DataFrame()

        logger.info(f"建構關注度指數，彙總方式：{self.aggregation}")

        # 彙總多關鍵字
        numeric_cols = aligned_df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            logger.error("無數值欄位可彙總")
            return pd.DataFrame()

        if self.aggregation == "sum":
            ai_raw = aligned_df[numeric_cols].sum(axis=1)
        elif self.aggregation == "max":
            ai_raw = aligned_df[numeric_cols].max(axis=1)
        elif self.aggregation == "pca":
            pca = PCA(n_components=1)
            filled = aligned_df[numeric_cols].fillna(0)
            ai_raw = pd.Series(
                pca.fit_transform(filled).flatten(),
                index=aligned_df.index,
            )
            logger.info(
                f"PCA 第一主成分解釋變異比例：{pca.explained_variance_ratio_[0]:.4f}"
            )
        else:
            ai_raw = aligned_df[numeric_cols].sum(axis=1)

        # Z-score 標準化
        ai_zscore = scipy_stats.zscore(ai_raw, nan_policy="omit")

        result = pd.DataFrame({
            "date": aligned_df.index if isinstance(aligned_df.index, pd.DatetimeIndex)
            else pd.to_datetime(aligned_df.index),
            "ai_raw": ai_raw.values,
            "ai_zscore": ai_zscore,
        })

        self._save(result)
        logger.info(f"關注度指數建構完成，共 {len(result)} 筆")
        return result

    def _load_raw_data(self) -> Optional[pd.DataFrame]:
        """從原始檔案載入 Google Trends 資料。

        Returns:
            合併後的 DataFrame 或 None。
        """
        csv_files = list(self.raw_dir.glob("trends_*.csv"))
        if not csv_files:
            return None

        dfs = {}
        for f in csv_files:
            keyword = f.stem.replace("trends_", "")
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            dfs[keyword] = df.rename(columns={"value": keyword})

        # 合併所有關鍵字
        merged = pd.concat(
            [df for df in dfs.values()],
            axis=1,
        )
        return merged

    def _save(self, df: pd.DataFrame) -> None:
        """儲存關注度指數。

        Args:
            df: 要儲存的 DataFrame。
        """
        output_path = self.output_dir / "attention_index.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"已儲存至 {output_path}")
