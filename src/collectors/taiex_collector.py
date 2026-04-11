"""TAIEX 台灣加權股價指數資料收集模組。

從 TWSE OpenAPI 或 yfinance 取得每日收盤價，
計算日報酬率與趨勢標籤。
"""

import datetime
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# 修復路徑含中文時 curl/libcurl 無法讀取 SSL 憑證的問題
_ca_bundle = Path.home() / ".cacert.pem"
if _ca_bundle.exists():
    os.environ.setdefault("CURL_CA_BUNDLE", str(_ca_bundle))

from src.utils.config_loader import get_config, get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("taiex_collector")


class TaiexCollector:
    """TAIEX 每日收盤價收集器。

    優先使用 TWSE OpenAPI，失敗時使用 yfinance 作為備援。

    Attributes:
        start_date: 資料起始日期。
        end_date: 資料結束日期。
        output_dir: 輸出目錄。
    """

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        config = get_config()
        self.start_date = start_date or config["project"]["train_start_date"]
        self.end_date = end_date or config["project"]["test_end_date"]
        self.output_dir = get_data_dir("raw/taiex")

    def collect(self) -> pd.DataFrame:
        """收集 TAIEX 資料。

        優先從 TWSE 取得，失敗時使用 yfinance。

        Returns:
            包含 date, close, daily_return, trend_label 的 DataFrame。
        """
        logger.info(f"開始收集 TAIEX 資料：{self.start_date} ~ {self.end_date}")

        try:
            df = self._collect_from_twse()
            if df is not None and len(df) > 0:
                logger.info(f"TWSE 取得 {len(df)} 筆資料")
            else:
                raise ValueError("TWSE 回傳空資料")
        except Exception as e:
            logger.warning(f"TWSE 取得失敗：{e}，改用 yfinance")
            df = self._collect_from_yfinance()

        if df is None or len(df) == 0:
            logger.error("無法取得 TAIEX 資料")
            return pd.DataFrame()

        df = self._process(df)
        self._save(df)
        return df

    def _collect_from_twse(self) -> Optional[pd.DataFrame]:
        """從 TWSE OpenAPI 按月取得 TAIEX 每日收盤價。

        Returns:
            原始資料 DataFrame 或 None。
        """
        all_data = []
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)

        current = start.replace(day=1)
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            url = (
                "https://www.twse.com.tw/exchangeReport/FMTQIK"
                f"?response=json&date={date_str}"
            )
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if data.get("stat") == "OK" and data.get("data"):
                    for row in data["data"]:
                        # row: [日期, 開盤, 最高, 最低, 收盤]
                        # 民國年轉西元年
                        roc_date = row[0].strip()
                        parts = roc_date.split("/")
                        year = int(parts[0]) + 1911
                        month = int(parts[1])
                        day = int(parts[2])
                        close_str = row[4].replace(",", "")
                        all_data.append({
                            "date": datetime.date(year, month, day),
                            "close": float(close_str),
                        })
            except Exception as e:
                logger.debug(f"TWSE 月份 {date_str} 失敗：{e}")

            # 下一個月
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

            time.sleep(3)  # 避免被封鎖

        if not all_data:
            return None

        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 篩選日期範圍
        mask = (df["date"] >= self.start_date) & (df["date"] <= self.end_date)
        return df.loc[mask].reset_index(drop=True)

    def _collect_from_yfinance(self) -> Optional[pd.DataFrame]:
        """使用 yfinance 取得 ^TWII 資料。

        Returns:
            原始資料 DataFrame 或 None。
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker("^TWII")
            hist = ticker.history(start=self.start_date, end=self.end_date)

            if hist.empty:
                return None

            df = pd.DataFrame({
                "date": hist.index.date,
                "close": hist["Close"].values,
            })
            df["date"] = pd.to_datetime(df["date"])
            return df

        except ImportError:
            logger.error("yfinance 未安裝，請執行 pip install yfinance")
            return None

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算日報酬率和趨勢標籤。

        Args:
            df: 包含 date, close 的 DataFrame。

        Returns:
            加入 daily_return 和 trend_label 的 DataFrame。
        """
        df = df.sort_values("date").reset_index(drop=True)
        df["daily_return"] = df["close"].pct_change()

        # 趨勢標籤：+1 上漲, 0 持平, -1 下跌
        df["trend_label"] = 0
        df.loc[df["daily_return"] > 0, "trend_label"] = 1
        df.loc[df["daily_return"] < 0, "trend_label"] = -1

        return df

    def _save(self, df: pd.DataFrame) -> None:
        """儲存為 CSV（UTF-8-BOM 編碼）。

        Args:
            df: 要儲存的 DataFrame。
        """
        output_path = self.output_dir / "taiex_daily.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"TAIEX 資料已儲存至 {output_path}（{len(df)} 筆）")
