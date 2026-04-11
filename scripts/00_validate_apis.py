"""Day 1-2: API 可用性驗證腳本。

逐一測試三個資料來源的可用性，只取最近 1 個月的資料做測試。
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

# 將專案根目錄加入 path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import setup_logger

logger = setup_logger("validate_apis")


def validate_taiex() -> dict:
    """測試 TAIEX 資料源。"""
    print("\n" + "=" * 60)
    print("[TAIEX] 測試台灣加權指數")
    print("=" * 60)

    try:
        from src.collectors.taiex_collector import TaiexCollector

        end = date.today()
        start = end - timedelta(days=30)

        collector = TaiexCollector(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

        df = collector.collect()

        if df is not None and len(df) > 0:
            print(f"  [OK] 連線成功")
            print(f"  取得 {len(df)} 筆資料")
            print(f"  範例資料：")
            print(df.head().to_string(index=False))
            print(f"\n  預估全期間（5年）收集時間：約 2-5 分鐘")
            return {"status": "OK", "count": len(df)}
        else:
            print(f"  [FAIL] 未取得資料")
            return {"status": "FAIL", "count": 0}

    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "ERROR", "error": str(e)}


def validate_google_trends() -> dict:
    """測試 Google Trends 資料源。"""
    print("\n" + "=" * 60)
    print("[Google Trends] 測試搜尋趨勢")
    print("=" * 60)

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="zh-TW", tz=480)

        end = date.today()
        start = end - timedelta(days=30)

        pytrends.build_payload(
            ["台股"],
            cat=0,
            timeframe=f"{start.isoformat()} {end.isoformat()}",
            geo="TW",
        )
        df = pytrends.interest_over_time()

        if df is not None and not df.empty:
            print(f"  [OK] 連線成功")
            print(f"  取得 {len(df)} 筆資料")
            print(f"  範例資料：")
            print(df.head().to_string())
            print(f"\n  預估全期間（5年 x 5關鍵字）收集時間：約 30-60 分鐘")
            return {"status": "OK", "count": len(df)}
        else:
            print(f"  [FAIL] 未取得資料")
            return {"status": "FAIL", "count": 0}

    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "ERROR", "error": str(e)}


def validate_ptt() -> dict:
    """測試 PTT Stock 板。"""
    print("\n" + "=" * 60)
    print("[PTT] 測試 PTT Stock 板")
    print("=" * 60)

    try:
        import requests
        from bs4 import BeautifulSoup

        session = requests.Session()
        session.cookies.set("over18", "1")
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

        url = "https://www.ptt.cc/bbs/Stock/index.html"
        resp = session.get(url, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.select("div.r-ent")

        if articles:
            print(f"  [OK] 連線成功")
            print(f"  當前頁面有 {len(articles)} 篇文章")
            print(f"  最新文章標題：")
            for a in articles[:5]:
                title_elem = a.select_one("div.title a")
                if title_elem:
                    print(f"    - {title_elem.text.strip()}")
            print(f"\n  預估全期間（5年）收集時間：約 2-6 小時")
            return {"status": "OK", "count": len(articles)}
        else:
            print(f"  [FAIL] 未取得文章")
            return {"status": "FAIL", "count": 0}

    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"status": "ERROR", "error": str(e)}


def main() -> None:
    """執行所有 API 驗證。"""
    print("=" * 60)
    print("  API 可用性驗證報告")
    print("=" * 60)

    results = {
        "TAIEX": validate_taiex(),
        "Google Trends": validate_google_trends(),
        "PTT Stock": validate_ptt(),
    }

    # 總結報告
    print("\n" + "=" * 60)
    print("  驗證總結")
    print("=" * 60)
    print(f"{'資料來源':<20} {'狀態':<10} {'資料量':<10}")
    print("-" * 40)
    for source, result in results.items():
        status = result["status"]
        count = result.get("count", "-")
        icon = {"OK": "V", "FAIL": "X", "SKIP": "-", "ERROR": "!"}
        print(f"  [{icon.get(status, '?')}] {source:<16} {status:<10} {count}")

    print("=" * 60)

    # 檢查是否有失敗
    failed = [s for s, r in results.items() if r["status"] == "FAIL"]
    errors = [s for s, r in results.items() if r["status"] == "ERROR"]

    if failed or errors:
        print(f"\n注意：有 {len(failed) + len(errors)} 個資料來源需要排除障礙")
    else:
        print(f"\n所有資料來源驗證通過（或已跳過）！")


if __name__ == "__main__":
    main()
