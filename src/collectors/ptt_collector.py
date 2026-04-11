"""PTT Stock 板資料收集模組。

爬取 PTT Stock 板的文章標題、推噓文數等資訊，
支援斷點續爬功能。
"""

import csv
import datetime
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.config_loader import get_config, get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("ptt_collector")

PTT_BASE = "https://www.ptt.cc"


class PttCollector:
    """PTT Stock 板文章收集器。

    爬取文章標題、作者、日期、推噓文數等資訊。
    支援斷點續爬。

    Attributes:
        board: PTT 板名。
        delay_range: 請求延遲範圍。
        collect_comments: 是否收集推文內容。
        output_dir: 輸出目錄。
    """

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        config = get_config()
        ptt_config = config["data_sources"]["ptt"]

        self.board: str = ptt_config["board"]
        self.base_url: str = ptt_config["base_url"]
        self.delay_range: List[float] = ptt_config["request_delay_sec"]
        self.collect_comments: bool = ptt_config["collect_comments"]

        self.start_date = datetime.date.fromisoformat(
            start_date or config["project"]["train_start_date"]
        )
        self.end_date = datetime.date.fromisoformat(
            end_date or config["project"]["test_end_date"]
        )

        self.output_dir = get_data_dir("raw/ptt")
        self.session = requests.Session()
        self.session.cookies.set("over18", "1")
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

        # 斷點續爬
        self.checkpoint_file = self.output_dir / "checkpoint.json"

        # 用於反向爬取時推斷年份（從今天的年份開始）
        self._current_year = datetime.date.today().year
        self._last_month: Optional[int] = None

    def collect(self) -> pd.DataFrame:
        """收集 PTT Stock 板文章。

        Returns:
            包含文章資訊的 DataFrame。
        """
        logger.info(
            f"開始收集 PTT {self.board} 板資料：{self.start_date} ~ {self.end_date}"
        )

        # 載入斷點
        checkpoint = self._load_checkpoint()
        start_page = checkpoint.get("last_page", None)
        articles = checkpoint.get("articles", [])

        # 取得最新頁碼
        latest_page = self._get_latest_page()
        if latest_page is None:
            logger.error("無法取得 PTT 最新頁碼")
            return pd.DataFrame()

        logger.info(f"PTT {self.board} 板最新頁碼：{latest_page}")

        # 從最新頁往回爬
        current_page = start_page if start_page else latest_page
        reached_start = False

        while current_page > 0 and not reached_start:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    page_articles = self._parse_page(current_page)

                    for article in page_articles:
                        if article["date"] is None:
                            continue

                        if article["date"] < self.start_date:
                            reached_start = True
                            break

                        if article["date"] <= self.end_date:
                            # 取得推噓文數
                            if self.collect_comments and article.get("url"):
                                comment_data = self._parse_article_comments(
                                    article["url"]
                                )
                                article.update(comment_data)
                                time.sleep(random.uniform(*self.delay_range))

                            articles.append(article)

                    # 儲存斷點
                    self._save_checkpoint(current_page, articles)

                    current_page -= 1
                    time.sleep(random.uniform(*self.delay_range))

                    if len(articles) > 0 and len(articles) % 100 == 0:
                        logger.info(f"已收集 {len(articles)} 篇文章")

                    break  # 成功，跳出重試迴圈

                except Exception as e:
                    wait = (attempt + 1) * 15
                    logger.error(
                        f"頁面 {current_page} 解析失敗"
                        f"（第 {attempt+1}/{max_retries} 次）：{e}"
                    )
                    if attempt < max_retries - 1:
                        logger.info(f"等待 {wait} 秒後重試...")
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"頁面 {current_page} 已重試 {max_retries} 次仍失敗，跳過"
                        )
                        current_page -= 1

        df = pd.DataFrame(articles)
        if len(df) > 0:
            df = df.sort_values("date").reset_index(drop=True)

        self._save(df)
        # 清除斷點
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

        return df

    def _get_latest_page(self) -> Optional[int]:
        """取得板上最新頁碼。

        Returns:
            最新頁碼或 None。
        """
        for attempt in range(5):
            try:
                resp = self.session.get(self.base_url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # 找到「上頁」按鈕的連結
                prev_link = soup.select_one(
                    'div.btn-group-paging a:nth-of-type(2)'
                )
                if prev_link and prev_link.get("href"):
                    match = re.search(r"index(\d+)", prev_link["href"])
                    if match:
                        return int(match.group(1)) + 1

            except Exception as e:
                wait = (attempt + 1) * 10
                logger.error(f"取得最新頁碼失敗（第 {attempt+1} 次）：{e}")
                if attempt < 4:
                    logger.info(f"等待 {wait} 秒後重試...")
                    time.sleep(wait)

        return None

    def _parse_page(self, page_num: int) -> List[Dict[str, Any]]:
        """解析單一頁面的文章列表。

        Args:
            page_num: 頁碼。

        Returns:
            該頁文章資訊列表。
        """
        url = f"{PTT_BASE}/bbs/{self.board}/index{page_num}.html"
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []

        # 只取分隔線之前的一般文章，跳過置頂/公告文
        entries = []
        for element in soup.select("div.r-ent, div.r-list-sep"):
            if "r-list-sep" in element.get("class", []):
                break  # 分隔線之後都是置頂文，停止
            entries.append(element)

        # 反轉順序：頁面內文章原本由舊到新排列，
        # 反轉為由新到舊，與跨頁的反向爬取方向一致，
        # 確保年份推斷邏輯（月份從小跳到大 = 跨年）正確運作。
        for entry in reversed(entries):
            try:
                article = self._parse_entry(entry)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.debug(f"解析文章條目失敗：{e}")

        return articles

    def _parse_entry(self, entry: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """解析單一文章條目。

        Args:
            entry: BeautifulSoup 元素。

        Returns:
            文章資訊字典或 None。
        """
        # 標題與連結
        title_elem = entry.select_one("div.title a")
        if title_elem is None:
            return None

        title = title_elem.text.strip()
        url = PTT_BASE + title_elem["href"]

        # 推文數
        nrec_elem = entry.select_one("div.nrec span")
        nrec_text = nrec_elem.text.strip() if nrec_elem else "0"

        if nrec_text == "爆":
            push_count = 100
        elif nrec_text.startswith("X"):
            push_count = -10 if nrec_text == "X" else -int(nrec_text[1:]) * 10
        elif nrec_text.isdigit():
            push_count = int(nrec_text)
        else:
            push_count = 0

        # 作者
        author_elem = entry.select_one("div.meta div.author")
        author = author_elem.text.strip() if author_elem else ""

        # 日期（格式 M/DD）
        date_elem = entry.select_one("div.meta div.date")
        date_str = date_elem.text.strip() if date_elem else ""

        article_date = self._parse_ptt_date(date_str)

        # 文章類別標籤
        tag_match = re.match(r"\[(.+?)\]", title)
        tag = tag_match.group(1) if tag_match else ""

        return {
            "title": title,
            "author": author,
            "date": article_date,
            "url": url,
            "tag": tag,
            "push_count": max(push_count, 0),
            "boo_count": abs(min(push_count, 0)),
            "arrow_count": 0,
        }

    def _parse_ptt_date(self, date_str: str) -> Optional[datetime.date]:
        """解析 PTT 的日期格式（M/DD）。

        爬蟲從最新頁往回爬，因此文章日期是由新到舊。
        當偵測到月份從小跳到大（例如 1 月 → 12 月），
        代表跨越了年份邊界，需要將年份減 1。

        Args:
            date_str: PTT 日期字串。

        Returns:
            完整日期或 None。
        """
        if not date_str:
            return None

        try:
            parts = date_str.strip().split("/")
            month = int(parts[0])
            day = int(parts[1])

            # 反向遍歷時推斷年份：月份從小跳到大代表跨年
            if self._last_month is not None and month > self._last_month:
                self._current_year -= 1
            self._last_month = month

            try:
                return datetime.date(self._current_year, month, day)
            except ValueError:
                return None

        except (ValueError, IndexError):
            return None

    def _parse_article_comments(self, url: str) -> Dict[str, int]:
        """解析文章內頁的推噓文數量。

        Args:
            url: 文章 URL。

        Returns:
            包含 push_count, boo_count, arrow_count 的字典。
        """
        result = {"push_count": 0, "boo_count": 0, "arrow_count": 0}

        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                for push in soup.select("div.push"):
                    tag_elem = push.select_one("span.push-tag")
                    if tag_elem is None:
                        continue

                    tag = tag_elem.text.strip()
                    if tag == "推":
                        result["push_count"] += 1
                    elif tag == "噓":
                        result["boo_count"] += 1
                    elif tag == "→":
                        result["arrow_count"] += 1

                return result

            except Exception as e:
                if attempt < 2:
                    time.sleep((attempt + 1) * 5)
                else:
                    logger.debug(f"解析文章推文失敗 {url}：{e}")

        return result

    def _load_checkpoint(self) -> Dict[str, Any]:
        """載入斷點續爬資訊。

        Returns:
            斷點資訊字典。
        """
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(
                    f"載入斷點：頁碼 {data.get('last_page')}，"
                    f"已有 {len(data.get('articles', []))} 篇"
                )
                # 從 checkpoint 恢復年份追蹤狀態
                if "current_year" in data and "last_month" in data:
                    self._current_year = data["current_year"]
                    self._last_month = data["last_month"]
                else:
                    # 向下相容：舊格式從最後一篇文章推斷
                    articles = data.get("articles", [])
                    if articles:
                        last_date_str = articles[-1].get("date")
                        if last_date_str:
                            last_date = datetime.date.fromisoformat(
                                last_date_str
                            )
                            self._current_year = last_date.year
                            self._last_month = last_date.month
                return data
        return {}

    def _save_checkpoint(
        self,
        page: int,
        articles: List[Dict[str, Any]],
    ) -> None:
        """儲存斷點。

        Args:
            page: 當前頁碼。
            articles: 已收集的文章列表。
        """
        data = {
            "last_page": page,
            "current_year": self._current_year,
            "last_month": self._last_month,
            "articles": articles,
        }

        class DateEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, datetime.date):
                    return obj.isoformat()
                return super().default(obj)

        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, cls=DateEncoder)

    def _save(self, df: pd.DataFrame) -> None:
        """儲存為 CSV。

        Args:
            df: 要儲存的 DataFrame。
        """
        output_path = self.output_dir / "ptt_stock_articles.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"PTT 資料已儲存至 {output_path}（{len(df)} 筆）")

