"""時間對齊模組。

將不同資料來源的時間序列對齊到台灣交易日基準軸，
處理非交易日資料合併與缺失值填充。
"""

import datetime
from typing import Dict, Optional

import pandas as pd

from src.utils.calendar_utils import TaiwanTradingCalendar
from src.utils.logging_utils import setup_logger

logger = setup_logger("time_aligner")


class TimeAligner:
    """時間序列對齊器。

    依據台灣交易日曆將多個資料來源的 DataFrame 對齊。

    Attributes:
        calendar: 台灣交易日曆。
        merge_method: 非交易日資料合併方式（'sum' 或 'mean'）。
        max_ffill: 前向填充的最大天數。
    """

    def __init__(
        self,
        merge_method: str = "sum",
        max_ffill: int = 3,
    ) -> None:
        self.calendar = TaiwanTradingCalendar()
        self.merge_method = merge_method
        self.max_ffill = max_ffill

    def align(
        self,
        dfs: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """將多個 DataFrame 對齊到交易日基準軸。

        Args:
            dfs: 以來源名稱為 key、DataFrame 為 value 的字典。
                 每個 DataFrame 需有 DatetimeIndex 或 'date' 欄位。
            start_date: 起始日期。
            end_date: 結束日期。

        Returns:
            對齊後的 DataFrame，index 為交易日。
        """
        trading_days = self.calendar.get_trading_day_index(
            datetime.date.fromisoformat(start_date),
            datetime.date.fromisoformat(end_date),
        )

        aligned_dfs = []

        for name, df in dfs.items():
            logger.info(f"對齊 {name} 資料（{len(df)} 筆）")
            aligned = self._align_single(df, trading_days)
            aligned.columns = [f"{name}_{col}" for col in aligned.columns]
            aligned_dfs.append(aligned)

        result = pd.concat(aligned_dfs, axis=1)
        logger.info(
            f"對齊完成：{len(result)} 個交易日，{len(result.columns)} 個欄位"
        )
        return result

    def _align_single(
        self,
        df: pd.DataFrame,
        trading_days: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """將單一 DataFrame 對齊到交易日。

        Args:
            df: 原始 DataFrame。
            trading_days: 交易日 DatetimeIndex。

        Returns:
            對齊後的 DataFrame。
        """
        df = df.copy()

        # 確保有 DatetimeIndex
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 只保留數值欄位
        numeric_cols = df.select_dtypes(include="number").columns
        df = df[numeric_cols]

        # 將每一天的資料對齊到交易日
        aligned_data = {}
        for trading_day in trading_days:
            td = trading_day.date()

            # 找出應合併到此交易日的所有日期
            # （從前一個交易日之後到當天）
            prev_td = self._get_prev_trading_day(td)
            if prev_td is not None:
                merge_start = pd.Timestamp(prev_td) + pd.Timedelta(days=1)
            else:
                merge_start = trading_day

            merge_end = trading_day

            mask = (df.index >= merge_start) & (df.index <= merge_end)
            period_data = df.loc[mask]

            if len(period_data) > 0:
                if self.merge_method == "sum":
                    aligned_data[trading_day] = period_data.sum()
                else:
                    aligned_data[trading_day] = period_data.mean()
            else:
                aligned_data[trading_day] = pd.Series(
                    {col: float("nan") for col in numeric_cols}
                )

        result = pd.DataFrame(aligned_data).T
        result.index.name = "date"

        # 前向填充（最多 max_ffill 天）
        result = result.ffill(limit=self.max_ffill)

        return result

    def _get_prev_trading_day(
        self,
        date: datetime.date,
    ) -> Optional[datetime.date]:
        """取得前一個交易日。

        Args:
            date: 參考日期。

        Returns:
            前一個交易日或 None。
        """
        current = date - datetime.timedelta(days=1)
        for _ in range(10):
            if self.calendar.is_trading_day(current):
                return current
            current -= datetime.timedelta(days=1)
        return None
