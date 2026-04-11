"""台灣交易日曆工具模組。

提供台灣證交所交易日的判斷、查詢、對齊功能，
涵蓋 2021-2026 年的國定假日與特殊休市日。
"""

import datetime
from typing import List, Optional

import pandas as pd


# 台灣證交所休市日（國定假日 + 補班日等），2021-2026
# 不含週六、週日（已自動排除）
_TAIWAN_HOLIDAYS = [
    # ===== 2021 =====
    "2021-01-01",  # 元旦
    "2021-02-08", "2021-02-09", "2021-02-10",  # 春節前
    "2021-02-11", "2021-02-12", "2021-02-15", "2021-02-16",  # 春節
    "2021-02-28",  # 二二八（日，補假3/1）
    "2021-03-01",  # 二二八補假
    "2021-04-02",  # 清明節前一日
    "2021-04-05",  # 清明節補假
    "2021-06-14",  # 端午節
    "2021-09-20", "2021-09-21",  # 中秋節
    "2021-10-11",  # 國慶日補假
    "2021-12-31",  # 彈性放假

    # ===== 2022 =====
    "2022-01-31",  # 春節前
    "2022-02-01", "2022-02-02", "2022-02-03", "2022-02-04",  # 春節
    "2022-02-28",  # 二二八
    "2022-04-04", "2022-04-05",  # 清明節、兒童節
    "2022-06-03",  # 端午節
    "2022-09-09",  # 中秋節（補假）
    "2022-10-10",  # 國慶日

    # ===== 2023 =====
    "2023-01-02",  # 元旦補假
    "2023-01-20",  # 春節前
    "2023-01-23", "2023-01-24", "2023-01-25",
    "2023-01-26", "2023-01-27",  # 春節
    "2023-02-27",  # 二二八補假
    "2023-02-28",  # 二二八
    "2023-04-03", "2023-04-04", "2023-04-05",  # 清明節、兒童節
    "2023-06-22", "2023-06-23",  # 端午節
    "2023-09-29",  # 中秋節
    "2023-10-09", "2023-10-10",  # 國慶日

    # ===== 2024 =====
    "2024-01-01",  # 元旦
    "2024-02-08", "2024-02-09",  # 春節前
    "2024-02-12", "2024-02-13", "2024-02-14",  # 春節
    "2024-02-28",  # 二二八
    "2024-04-04", "2024-04-05",  # 清明節、兒童節
    "2024-06-10",  # 端午節
    "2024-09-17",  # 中秋節
    "2024-10-10",  # 國慶日

    # ===== 2025 =====
    "2025-01-01",  # 元旦
    "2025-01-27", "2025-01-28", "2025-01-29",
    "2025-01-30", "2025-01-31",  # 春節
    "2025-02-28",  # 二二八
    "2025-04-03", "2025-04-04",  # 清明節、兒童節
    "2025-05-30",  # 端午節（補假）
    "2025-06-02",  # 端午節補假
    "2025-10-06",  # 中秋節補假
    "2025-10-10",  # 國慶日

    # ===== 2026 =====
    "2026-01-01",  # 元旦
    "2026-01-02",  # 彈性放假
    "2026-02-16", "2026-02-17", "2026-02-18",
    "2026-02-19", "2026-02-20",  # 春節
    "2026-02-27",  # 二二八補假（週六不影響）
    "2026-03-02",  # 二二八補假（週一）
    "2026-04-03",  # 兒童節補假
    "2026-04-06",  # 清明節補假
    "2026-05-25",  # 端午節（週一）
    "2026-10-05",  # 中秋節（週一）
    "2026-10-09",  # 國慶日補假
]

_HOLIDAY_SET = {datetime.date.fromisoformat(d) for d in _TAIWAN_HOLIDAYS}


class TaiwanTradingCalendar:
    """台灣證交所交易日曆。

    提供交易日判斷、查詢與對齊功能。

    Attributes:
        holidays: 台灣證交所國定假日集合。
    """

    def __init__(self) -> None:
        self.holidays = _HOLIDAY_SET

    def is_trading_day(self, date: datetime.date) -> bool:
        """判斷指定日期是否為交易日。

        Args:
            date: 要判斷的日期。

        Returns:
            True 表示為交易日，False 表示為休市日。
        """
        if isinstance(date, datetime.datetime):
            date = date.date()
        # 週六日不交易
        if date.weekday() >= 5:
            return False
        # 國定假日不交易
        if date in self.holidays:
            return False
        return True

    def get_trading_days(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[datetime.date]:
        """取得指定期間內的所有交易日。

        Args:
            start_date: 起始日期（含）。
            end_date: 結束日期（含）。

        Returns:
            交易日列表。
        """
        if isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.date.fromisoformat(end_date)

        trading_days = []
        current = start_date
        while current <= end_date:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += datetime.timedelta(days=1)
        return trading_days

    def get_next_trading_day(self, date: datetime.date) -> datetime.date:
        """取得指定日期之後的下一個交易日。

        Args:
            date: 參考日期。

        Returns:
            下一個交易日。
        """
        if isinstance(date, str):
            date = datetime.date.fromisoformat(date)

        current = date + datetime.timedelta(days=1)
        while not self.is_trading_day(current):
            current += datetime.timedelta(days=1)
        return current

    def align_to_trading_day(self, date: datetime.date) -> datetime.date:
        """將日期對齊到交易日。

        若該日期本身為交易日則直接回傳，否則對齊到下一個交易日。

        Args:
            date: 要對齊的日期。

        Returns:
            對齊後的交易日。
        """
        if isinstance(date, str):
            date = datetime.date.fromisoformat(date)

        if self.is_trading_day(date):
            return date
        return self.get_next_trading_day(date)

    def get_trading_day_index(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> pd.DatetimeIndex:
        """取得交易日的 pandas DatetimeIndex。

        Args:
            start_date: 起始日期。
            end_date: 結束日期。

        Returns:
            交易日的 DatetimeIndex。
        """
        days = self.get_trading_days(start_date, end_date)
        return pd.DatetimeIndex([pd.Timestamp(d) for d in days])
