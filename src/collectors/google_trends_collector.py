"""Google Trends 資料收集模組。

使用 pytrends 套件分段查詢日頻率資料，
並透過重疊期間的比例縮放拼接（overlap normalization）
將多個時段的資料串接成完整的時間序列。
"""

import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pytrends.request import TrendReq

from src.utils.config_loader import get_config, get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("google_trends_collector")


class GoogleTrendsCollector:
    """Google Trends 資料收集器。

    透過分段查詢取得日頻率的搜尋趨勢資料，
    並使用重疊期間做比例縮放拼接。

    Attributes:
        keywords: 要查詢的關鍵字列表。
        geo: 地區代碼。
        chunk_months: 每次查詢的月份數。
        overlap_days: 重疊天數。
        delay_range: 請求延遲範圍（秒）。
        output_dir: 輸出目錄。
    """

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        config = get_config()
        gt_config = config["data_sources"]["google_trends"]

        self.start_date = start_date or config["project"]["train_start_date"]
        self.end_date = end_date or config["project"]["test_end_date"]
        self.keywords: List[str] = (
            gt_config["keywords"]["primary"]
            + gt_config["keywords"]["secondary"]
        )
        self.geo: str = gt_config["geo"]
        self.chunk_months: int = gt_config["chunk_months"]
        self.overlap_days: int = gt_config["overlap_days"]
        self.delay_range: List[float] = gt_config["request_delay_sec"]
        self.output_dir = get_data_dir("raw/google_trends")

        self.pytrends = TrendReq(hl="zh-TW", tz=480, timeout=(10, 30))

    def collect(self) -> Dict[str, pd.DataFrame]:
        """收集所有關鍵字的 Google Trends 資料。

        Returns:
            以關鍵字為 key、DataFrame 為 value 的字典。
        """
        results: Dict[str, pd.DataFrame] = {}

        for keyword in self.keywords:
            logger.info(f"開始收集關鍵字：{keyword}")
            try:
                df = self._collect_keyword(keyword)
                if df is not None and len(df) > 0:
                    results[keyword] = df
                    self._save(keyword, df)
                    logger.info(f"關鍵字 '{keyword}' 完成，共 {len(df)} 筆")
                else:
                    logger.warning(f"關鍵字 '{keyword}' 未取得資料")
            except Exception as e:
                logger.error(f"關鍵字 '{keyword}' 收集失敗：{e}")

        return results

    def _collect_keyword(self, keyword: str) -> Optional[pd.DataFrame]:
        """收集單一關鍵字的 Google Trends 資料。

        使用分段查詢與重疊拼接策略。

        Args:
            keyword: 要查詢的關鍵字。

        Returns:
            包含日頻率趨勢資料的 DataFrame。
        """
        chunks = self._generate_time_chunks()
        chunk_dfs: List[pd.DataFrame] = []

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            timeframe = f"{chunk_start} {chunk_end}"
            logger.debug(f"  查詢時段 {i + 1}/{len(chunks)}：{timeframe}")

            df = self._query_with_retry(keyword, timeframe)
            if df is not None and len(df) > 0:
                chunk_dfs.append(df)

            # 隨機延遲
            delay = random.uniform(*self.delay_range)
            time.sleep(delay)

        if not chunk_dfs:
            return None

        # 重疊拼接
        return self._stitch_chunks(chunk_dfs)

    def _generate_time_chunks(self) -> List[Tuple[str, str]]:
        """產生分段查詢的時間區間（含重疊）。

        Returns:
            時間區間列表，每個元素為 (start, end) 字串。
        """
        chunks = []
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)

        current_start = start
        while current_start < end:
            current_end = current_start + pd.DateOffset(months=self.chunk_months)
            if current_end > end:
                current_end = end

            chunks.append((
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            ))

            # 已到達結束日，不需要再分段
            if current_end >= end:
                break

            # 下一段的起始日 = 當前結束日 - overlap_days
            current_start = current_end - pd.Timedelta(days=self.overlap_days)

        return chunks

    def _query_with_retry(
        self,
        keyword: str,
        timeframe: str,
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """查詢 Google Trends，遇到 429 自動重試。

        Args:
            keyword: 關鍵字。
            timeframe: 時間範圍字串。
            max_retries: 最大重試次數。

        Returns:
            查詢結果 DataFrame 或 None。
        """
        for attempt in range(max_retries):
            try:
                self.pytrends.build_payload(
                    [keyword],
                    cat=0,
                    timeframe=timeframe,
                    geo=self.geo,
                )
                df = self.pytrends.interest_over_time()

                if df is not None and not df.empty:
                    df = df[[keyword]].copy()
                    df.columns = ["value"]
                    return df

            except Exception as e:
                wait_time = (attempt + 1) * 60
                logger.warning(
                    f"  查詢失敗（第 {attempt + 1} 次）：{e}，"
                    f"等待 {wait_time} 秒後重試"
                )
                time.sleep(wait_time)

        return None

    def _stitch_chunks(self, chunk_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """使用重疊期間的比例因子將多段資料拼接。

        Args:
            chunk_dfs: 各時段的 DataFrame 列表。

        Returns:
            拼接後的完整 DataFrame。
        """
        if len(chunk_dfs) == 1:
            return chunk_dfs[0]

        # 以第一段為基準
        result = chunk_dfs[0].copy()

        for i in range(1, len(chunk_dfs)):
            current = chunk_dfs[i].copy()

            # 找出重疊期間
            overlap_start = current.index.min()
            overlap_end = result.index.max()

            if overlap_start > overlap_end:
                # 沒有重疊，直接串接
                result = pd.concat([result, current])
                continue

            # 取得重疊區間的資料
            overlap_mask_result = (result.index >= overlap_start) & (
                result.index <= overlap_end
            )
            overlap_mask_current = (current.index >= overlap_start) & (
                current.index <= overlap_end
            )

            overlap_result = result.loc[overlap_mask_result, "value"]
            overlap_current = current.loc[overlap_mask_current, "value"]

            # 對齊索引
            common_idx = overlap_result.index.intersection(overlap_current.index)

            if len(common_idx) == 0 or overlap_current.loc[common_idx].mean() == 0:
                result = pd.concat([result, current[current.index > overlap_end]])
                continue

            # 計算比例因子
            scale_factor = (
                overlap_result.loc[common_idx].mean()
                / overlap_current.loc[common_idx].mean()
            )

            # 縮放當前段落並只取非重疊部分
            current["value"] = current["value"] * scale_factor
            non_overlap = current[current.index > overlap_end]
            result = pd.concat([result, non_overlap])

        result = result.sort_index()
        result = result[~result.index.duplicated(keep="first")]
        return result

    def _save(self, keyword: str, df: pd.DataFrame) -> None:
        """儲存為 CSV。

        Args:
            keyword: 關鍵字名稱。
            df: 要儲存的 DataFrame。
        """
        safe_name = keyword.replace(" ", "_").replace("/", "_")
        output_path = self.output_dir / f"trends_{safe_name}.csv"
        df.to_csv(output_path, encoding="utf-8-sig")
        logger.info(f"已儲存至 {output_path}")
