"""爬取測試期每日最高互動 PTT 文章的全文與留言。

為 LLM 情緒分類（BI 強化）準備資料。
每個交易日取互動量最高的 1 篇文章，爬取全文及所有推文。
支援斷點續爬，中斷後重新執行即可自動接續。

用法:
    python scripts/fetch_top_articles.py
    python scripts/fetch_top_articles.py --top-n 3    # 每天取 top-3
    python scripts/fetch_top_articles.py --period all  # 全期間（含訓練期）
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# SSL 修復：中文路徑 CA bundle 問題
_ca_bundle = Path.home() / ".cacert.pem"
if _ca_bundle.exists():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_ca_bundle))

from src.utils.config_loader import get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("fetch_top_articles")

OUTPUT_DIR = get_data_dir("raw/ptt")
CHECKPOINT_FILE = OUTPUT_DIR / "fetch_top_checkpoint.json"
OUTPUT_FILE = OUTPUT_DIR / "top_articles_content.jsonl"


def select_target_articles(top_n: int = 1, period: str = "test") -> pd.DataFrame:
    """從已有的文章列表中，選出每個交易日互動量最高的 N 篇。

    Args:
        top_n: 每天取幾篇。
        period: 'test' 只取測試期，'all' 取全期間。

    Returns:
        包含目標文章資訊的 DataFrame。
    """
    articles_path = OUTPUT_DIR / "ptt_stock_articles.csv"
    taiex_path = get_data_dir("raw/taiex") / "taiex_daily.csv"

    df = pd.read_csv(articles_path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["total_interaction"] = df["push_count"] + df["boo_count"] + df["arrow_count"]

    taiex = pd.read_csv(taiex_path, encoding="utf-8-sig")
    taiex["date"] = pd.to_datetime(taiex["date"])
    trading_days = set(taiex["date"].dt.date)

    # 只保留交易日的文章
    df["date_only"] = df["date"].dt.date
    df = df[df["date_only"].isin(trading_days)]

    if period == "test":
        df = df[df["date"] >= "2025-01-01"]

    # 每天取 top-N
    targets = df.groupby("date_only").apply(
        lambda g: g.nlargest(top_n, "total_interaction"),
        include_groups=False,
    ).reset_index(drop=True)

    return targets[["date", "title", "url", "tag", "push_count",
                     "boo_count", "arrow_count", "total_interaction"]]


def load_checkpoint() -> set:
    """載入已完成的 URL 集合。"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            done = set(data.get("done_urls", []))
            logger.info(f"載入斷點：已完成 {len(done)} 篇")
            return done
    return set()


def save_checkpoint(done_urls: set) -> None:
    """儲存斷點。"""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"done_urls": list(done_urls)}, f, ensure_ascii=False)


def fetch_article(
    url: str, session: requests.Session, max_comments: int = 100,
) -> dict:
    """爬取單篇文章的全文與推文。

    Args:
        url: 文章 URL。
        session: requests Session。
        max_comments: 最多取幾則推文（節省 token）。

    Returns:
        包含 body 和 comments 的字典。
    """
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. 先提取推文（在 decompose 之前）
    push_elements = soup.select("div.push")
    total_push = 0
    total_boo = 0
    total_arrow = 0
    comments = []
    for p in push_elements:
        tag_e = p.select_one("span.push-tag")
        userid = p.select_one("span.push-userid")
        content = p.select_one("span.push-content")
        if tag_e and content:
            tag_text = tag_e.text.strip()
            if tag_text == "推":
                total_push += 1
            elif tag_text == "噓":
                total_boo += 1
            elif tag_text == "→":
                total_arrow += 1
            # 只保留前 max_comments 則留言的文字
            if len(comments) < max_comments:
                comments.append({
                    "type": tag_text,
                    "user": userid.text.strip() if userid else "",
                    "text": content.text.strip().lstrip(": "),
                })

    # 2. 提取文章本體
    main_content = soup.select_one("div#main-content")
    body = ""
    if main_content:
        for elem in main_content.select(
            "div.article-metaline, div.article-metaline-right, div.push"
        ):
            elem.decompose()
        body = main_content.get_text()
        # 截到簽名檔分隔線
        if "\n--\n" in body:
            body = body[: body.index("\n--\n")]
        body = body.strip()

    return {
        "body": body,
        "comments": comments,
        "actual_push": total_push,
        "actual_boo": total_boo,
        "actual_arrow": total_arrow,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=1, help="每天取幾篇")
    parser.add_argument("--period", default="test", choices=["test", "all"])
    args = parser.parse_args()

    print("=" * 60)
    print("  爬取 PTT 高互動文章全文與留言")
    print("=" * 60)

    # 選取目標文章
    targets = select_target_articles(top_n=args.top_n, period=args.period)
    print(f"  目標文章: {len(targets)} 篇")
    print(f"  期間: {targets['date'].min().date()} ~ {targets['date'].max().date()}")
    print(f"  平均互動量: {targets['total_interaction'].mean():.0f}")

    # 載入斷點
    done_urls = load_checkpoint()
    remaining = targets[~targets["url"].isin(done_urls)]
    print(f"  已完成: {len(done_urls)} 篇")
    print(f"  待爬取: {len(remaining)} 篇")

    if len(remaining) == 0:
        print("\n  所有文章已爬取完成！")
        return

    # 建立 session
    session = requests.Session()
    session.cookies.set("over18", "1")
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })

    # 開始爬取
    print(f"\n  開始爬取...")
    success = 0
    fail = 0

    for idx, (_, row) in enumerate(remaining.iterrows()):
        url = row["url"]
        try:
            result = fetch_article(url, session)

            # 寫入 JSONL（追加模式）
            record = {
                "date": row["date"].strftime("%Y-%m-%d"),
                "title": row["title"],
                "url": url,
                "tag": row["tag"],
                "push_count": int(row["push_count"]),
                "boo_count": int(row["boo_count"]),
                "arrow_count": int(row["arrow_count"]),
                "body": result["body"],
                "comments": result["comments"],
                "n_comments": len(result["comments"]),
                "actual_push": result["actual_push"],
                "actual_boo": result["actual_boo"],
                "actual_arrow": result["actual_arrow"],
            }

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            done_urls.add(url)
            success += 1

            if success % 10 == 0:
                save_checkpoint(done_urls)
                elapsed_pct = (success + fail) / len(remaining) * 100
                print(
                    f"  [{success + fail}/{len(remaining)}] "
                    f"({elapsed_pct:.0f}%) "
                    f"成功 {success}, 失敗 {fail}"
                )

        except Exception as e:
            fail += 1
            logger.warning(f"爬取失敗 {url}: {e}")
            if "Connection" in str(e) or "Timeout" in str(e):
                time.sleep(10)

        # 延遲，避免被封鎖
        time.sleep(random.uniform(1.5, 3.0))

    # 最終儲存斷點
    save_checkpoint(done_urls)

    print(f"\n  爬取完成: 成功 {success}, 失敗 {fail}")
    print(f"  結果儲存至: {OUTPUT_FILE}")
    print(f"  總計已爬取: {len(done_urls)} 篇")

    # 清除斷點（如果全部完成）
    if fail == 0 and len(done_urls) >= len(targets):
        CHECKPOINT_FILE.unlink(missing_ok=True)
        print("  斷點檔已清除（全部完成）")


if __name__ == "__main__":
    main()
