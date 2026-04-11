"""傳播指數（Propagation Index）建構模組。

將 PTT Stock 板的互動數據（推/噓/箭頭）透過 PCA 降維為單一傳播指數。

設計理念：
- BI 使用 push/boo 的 **比值**（方向性指標，量綱消除）
- PI 使用 push/boo/arrow 的 **絕對量**（強度指標，衡量傳播規模）
- 兩者概念正交：高 PI 不代表高 BI（活躍討論可能多空激辯）

時間對齊策略：
- PTT 在非交易日（週末、假日）仍有活躍討論
- 非交易日的互動量歸併至下一個交易日，避免資訊損失
- 使用 TAIEX 交易日曆作為對齊基準
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import get_config, get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("propagation_builder")


class PropagationBuilder:
    """傳播指數建構器。

    使用 PCA 將 PTT 的每日推/噓/箭頭互動量
    降維為單一傳播指數。

    Attributes:
        features: 用於 PCA 的特徵欄位。
        min_variance: 最低解釋變異比例門檻。
        output_dir: 輸出目錄。
    """

    def __init__(self) -> None:
        config = get_config()
        pi_config = config["index_construction"]["propagation_index"]

        self.features = pi_config["features"]
        self.min_variance: float = pi_config["min_variance_explained"]
        self.output_dir = get_data_dir("processed")
        self.raw_ptt_dir = get_data_dir("raw/ptt")
        self.raw_taiex_dir = get_data_dir("raw/taiex")

    def build(self, aligned_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """建構傳播指數。

        Args:
            aligned_df: 已對齊到交易日且包含互動欄位的 DataFrame。
                        若為 None 則從原始檔案載入。

        Returns:
            包含 date, push_total, boo_total, arrow_total,
            article_count, pi_raw, pi_zscore, variance_explained 的 DataFrame。
        """
        if aligned_df is None:
            aligned_df = self._load_and_aggregate()

        if aligned_df is None or len(aligned_df) == 0:
            logger.error("無 PTT 資料可建構傳播指數")
            return pd.DataFrame()

        logger.info("建構傳播指數")

        # PCA 降維
        feature_cols = [c for c in self.features if c in aligned_df.columns]

        if len(feature_cols) < 2:
            logger.warning("特徵欄位不足，直接使用互動總量作為指數")
            pi_raw = aligned_df[feature_cols].sum(axis=1) if feature_cols else pd.Series(0, index=aligned_df.index)
            variance_explained = 1.0
        else:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(
                aligned_df[feature_cols].fillna(0)
            )

            pca = PCA(n_components=1)
            pi_raw_values = pca.fit_transform(scaled_data).flatten()
            pi_raw = pd.Series(pi_raw_values, index=aligned_df.index)

            variance_explained = pca.explained_variance_ratio_[0]

            if variance_explained < self.min_variance:
                logger.warning(
                    f"PCA 第一主成分解釋變異 {variance_explained:.4f} "
                    f"低於門檻 {self.min_variance}"
                )
            else:
                logger.info(
                    f"PCA 第一主成分解釋變異：{variance_explained:.4f}"
                )

        # Z-score 標準化
        pi_zscore = scipy_stats.zscore(pi_raw, nan_policy="omit")

        result = pd.DataFrame({
            "date": aligned_df.index if isinstance(aligned_df.index, pd.DatetimeIndex)
            else pd.to_datetime(aligned_df.index),
            "push_total": aligned_df["push_total"].values,
            "boo_total": aligned_df["boo_total"].values,
            "arrow_total": aligned_df["arrow_total"].values,
            "article_count": aligned_df["article_count"].values,
            "pi_raw": pi_raw.values,
            "pi_zscore": pi_zscore,
            "variance_explained": variance_explained,
        })

        self._save(result)
        logger.info(f"傳播指數建構完成，共 {len(result)} 筆")
        return result

    def _load_and_aggregate(self) -> Optional[pd.DataFrame]:
        """從原始 PTT 檔案載入，按日彙總互動量，並對齊至交易日。

        非交易日（週末、假日）的互動量歸併至下一個交易日，
        確保時間維度的資訊不會因日曆對齊而損失。

        Returns:
            按交易日彙總的 DataFrame 或 None。
        """
        raw_path = self.raw_ptt_dir / "ptt_stock_articles.csv"
        if not raw_path.exists():
            logger.error(f"找不到 PTT 資料：{raw_path}")
            return None

        df = pd.read_csv(raw_path, encoding="utf-8-sig")

        if "date" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"])

        # 確保 arrow_count 欄位存在
        if "arrow_count" not in df.columns:
            df["arrow_count"] = 0

        # 按日曆日彙總
        daily = df.groupby("date").agg(
            push_total=("push_count", "sum"),
            boo_total=("boo_count", "sum"),
            arrow_total=("arrow_count", "sum"),
            article_count=("title", "count"),
        )

        # 載入 TAIEX 交易日曆進行非交易日歸併
        daily = self._align_to_trading_days(daily)

        return daily

    def _align_to_trading_days(self, daily: pd.DataFrame) -> pd.DataFrame:
        """將非交易日的互動量歸併至下一個交易日。

        例如週六、週日的 PTT 討論量會累加到下週一，
        反映了「週末累積的情緒在開盤日釋放」的時間效應。

        Args:
            daily: 按日曆日彙總的 DataFrame（index 為日期）。

        Returns:
            按交易日重新彙總的 DataFrame。
        """
        taiex_path = self.raw_taiex_dir / "taiex_daily.csv"
        if not taiex_path.exists():
            logger.warning("找不到 TAIEX 資料，無法對齊交易日，使用所有日曆日")
            return daily

        taiex = pd.read_csv(taiex_path, encoding="utf-8-sig")
        taiex["date"] = pd.to_datetime(taiex["date"])
        trading_days = sorted(taiex["date"].unique())

        if len(trading_days) == 0:
            return daily

        # 為每個日曆日找到對應的下一個交易日
        trading_days_set = set(trading_days)

        def map_to_next_trading_day(cal_date):
            """將日曆日映射到最近的下一個交易日。"""
            if cal_date in trading_days_set:
                return cal_date
            # 往後找最近的交易日（最多找 10 天，涵蓋長假）
            for offset in range(1, 11):
                next_day = cal_date + pd.Timedelta(days=offset)
                if next_day in trading_days_set:
                    return next_day
            return None

        daily = daily.copy()
        daily["trading_date"] = [
            map_to_next_trading_day(d) for d in daily.index
        ]

        # 移除無法映射的日期（超出 TAIEX 資料範圍）
        daily = daily.dropna(subset=["trading_date"])

        # 按交易日重新彙總（非交易日的量累加到對應交易日）
        aligned = daily.groupby("trading_date").agg(
            push_total=("push_total", "sum"),
            boo_total=("boo_total", "sum"),
            arrow_total=("arrow_total", "sum"),
            article_count=("article_count", "sum"),
        )

        aligned.index = pd.to_datetime(aligned.index)
        aligned.index.name = "date"

        non_trading_days = len(daily) - len(
            [d for d in daily.index if d in trading_days_set]
        )
        logger.info(
            f"交易日對齊完成：{len(daily)} 個日曆日 → {len(aligned)} 個交易日"
            f"（{non_trading_days} 個非交易日的互動量已歸併）"
        )

        return aligned

    def _save(self, df: pd.DataFrame) -> None:
        """儲存傳播指數。

        Args:
            df: 要儲存的 DataFrame。
        """
        output_path = self.output_dir / "propagation_index.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"已儲存至 {output_path}")
