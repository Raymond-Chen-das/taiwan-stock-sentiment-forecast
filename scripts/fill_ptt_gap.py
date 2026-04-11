"""補爬 PTT 缺失日期區間的文章。

用法：
    python scripts/fill_ptt_gap.py --start 2024-10-10 --end 2024-10-31

從最新頁往回爬，只保留指定日期範圍內的文章，
爬到 start 之前自動停止。完成後合併至主 CSV。
"""

import argparse
import datetime
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PTT_BASE = "https://www.ptt.cc"
BOARD = "Stock"
DELAY = (0.5, 1.5)


def create_session():
    session = requests.Session()
    session.cookies.set("over18", "1")
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })
    return session


def get_latest_page(session):
    for attempt in range(5):
        try:
            resp = session.get(f"{PTT_BASE}/bbs/{BOARD}/index.html", timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            prev_link = soup.select_one("div.btn-group-paging a:nth-of-type(2)")
            if prev_link and prev_link.get("href"):
                match = re.search(r"index(\d+)", prev_link["href"])
                if match:
                    return int(match.group(1)) + 1
        except Exception as e:
            wait = (attempt + 1) * 10
            print(f"取得最新頁碼失敗（第 {attempt+1} 次）：{e}")
            if attempt < 4:
                time.sleep(wait)
    return None


def parse_ptt_date(date_str, state):
    """解析 PTT 日期，state = {'year': int, 'last_month': int or None}"""
    if not date_str:
        return None
    try:
        parts = date_str.strip().split("/")
        month = int(parts[0])
        day = int(parts[1])
        if state["last_month"] is not None and month > state["last_month"]:
            state["year"] -= 1
        state["last_month"] = month
        try:
            return datetime.date(state["year"], month, day)
        except ValueError:
            return None
    except (ValueError, IndexError):
        return None


def parse_page(session, page_num, state):
    url = f"{PTT_BASE}/bbs/{BOARD}/index{page_num}.html"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    articles = []
    entries = []
    for element in soup.select("div.r-ent, div.r-list-sep"):
        if "r-list-sep" in element.get("class", []):
            break
        entries.append(element)

    for entry in reversed(entries):
        try:
            title_elem = entry.select_one("div.title a")
            if title_elem is None:
                continue

            title = title_elem.text.strip()
            article_url = PTT_BASE + title_elem["href"]

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

            author_elem = entry.select_one("div.meta div.author")
            author = author_elem.text.strip() if author_elem else ""

            date_elem = entry.select_one("div.meta div.date")
            date_str = date_elem.text.strip() if date_elem else ""
            article_date = parse_ptt_date(date_str, state)

            tag_match = re.match(r"\[(.+?)\]", title)
            tag = tag_match.group(1) if tag_match else ""

            articles.append({
                "title": title,
                "author": author,
                "date": article_date,
                "url": article_url,
                "tag": tag,
                "push_count": max(push_count, 0),
                "boo_count": abs(min(push_count, 0)),
                "arrow_count": 0,
            })
        except Exception:
            continue

    return articles


def parse_article_comments(session, url):
    result = {"push_count": 0, "boo_count": 0, "arrow_count": 0}
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=15)
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
    return result


def fill_gap(start_date, end_date):
    session = create_session()
    latest_page = get_latest_page(session)
    if latest_page is None:
        print("無法取得最新頁碼")
        return

    print(f"最新頁碼: {latest_page}")
    print(f"目標範圍: {start_date} ~ {end_date}")

    state = {"year": datetime.date.today().year, "last_month": None}
    current_page = latest_page
    articles = []
    reached_start = False

    while current_page > 0 and not reached_start:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                page_articles = parse_page(session, current_page, state)

                for article in page_articles:
                    if article["date"] is None:
                        continue
                    if article["date"] < start_date:
                        reached_start = True
                        break
                    if start_date <= article["date"] <= end_date:
                        comment_data = parse_article_comments(session, article["url"])
                        article.update(comment_data)
                        time.sleep(random.uniform(*DELAY))
                        articles.append(article)

                current_page -= 1
                time.sleep(random.uniform(*DELAY))

                if len(articles) > 0 and len(articles) % 50 == 0:
                    print(f"已收集 {len(articles)} 篇 (頁碼 {current_page})")

                break
            except Exception as e:
                wait = (attempt + 1) * 15
                print(f"頁面 {current_page} 失敗（{attempt+1}/{max_retries}）：{e}")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    print(f"頁面 {current_page} 跳過")
                    current_page -= 1

    if not articles:
        print("未收集到任何文章")
        return

    gap_df = pd.DataFrame(articles)
    gap_df["date"] = gap_df["date"].astype(str)
    print(f"\n補爬完成：{len(gap_df)} 篇 ({gap_df['date'].min()} ~ {gap_df['date'].max()})")

    # 合併至主 CSV
    csv_path = Path("data/raw/ptt/ptt_stock_articles.csv")
    main_df = pd.read_csv(csv_path)
    combined = pd.concat([main_df, gap_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="url", keep="first")
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"已合併並儲存：{len(combined)} 篇")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="補爬起始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="補爬結束日期 YYYY-MM-DD")
    args = parser.parse_args()

    fill_gap(
        datetime.date.fromisoformat(args.start),
        datetime.date.fromisoformat(args.end),
    )
