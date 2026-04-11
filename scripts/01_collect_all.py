"""批次資料收集腳本。

支援命令列參數選擇資料來源和時間範圍。
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import get_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("collect_all")


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="批次資料收集")
    parser.add_argument(
        "--source",
        choices=["google_trends", "ptt", "taiex", "all"],
        default="all",
        help="要收集的資料來源",
    )
    parser.add_argument("--start-date", type=str, default=None, help="起始日期")
    parser.add_argument("--end-date", type=str, default=None, help="結束日期")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="測試模式：只取 3 個月資料",
    )
    return parser.parse_args()


def collect_taiex(start_date: str, end_date: str) -> None:
    """收集 TAIEX 資料。"""
    from src.collectors.taiex_collector import TaiexCollector

    print("\n[1/3] 收集 TAIEX 資料...")
    collector = TaiexCollector(start_date=start_date, end_date=end_date)
    df = collector.collect()
    print(f"  完成：{len(df)} 筆")


def collect_google_trends(start_date: str, end_date: str) -> None:
    """收集 Google Trends 資料。"""
    from src.collectors.google_trends_collector import GoogleTrendsCollector

    print("\n[2/3] 收集 Google Trends 資料...")
    collector = GoogleTrendsCollector(start_date=start_date, end_date=end_date)
    results = collector.collect()
    total = sum(len(df) for df in results.values())
    print(f"  完成：{len(results)} 個關鍵字，共 {total} 筆")


def collect_ptt(start_date: str, end_date: str) -> None:
    """收集 PTT 資料。"""
    from src.collectors.ptt_collector import PttCollector

    print("\n[3/3] 收集 PTT Stock 板資料...")
    collector = PttCollector(start_date=start_date, end_date=end_date)
    df = collector.collect()
    print(f"  完成：{len(df)} 篇文章")


def main() -> None:
    """主流程。"""
    args = parse_args()
    config = get_config()

    start_date = args.start_date or config["project"]["train_start_date"]
    end_date = args.end_date or config["project"]["test_end_date"]

    if args.test_mode:
        from datetime import date, timedelta

        end_dt = date.fromisoformat(end_date)
        start_date = (end_dt - timedelta(days=90)).isoformat()
        print(f"[測試模式] 資料範圍：{start_date} ~ {end_date}")

    print(f"資料收集範圍：{start_date} ~ {end_date}")
    print(f"目標來源：{args.source}")

    collectors = {
        "taiex": collect_taiex,
        "google_trends": collect_google_trends,
        "ptt": collect_ptt,
    }

    if args.source == "all":
        for name, func in collectors.items():
            try:
                func(start_date, end_date)
            except Exception as e:
                logger.error(f"{name} 收集失敗：{e}")
                print(f"  [ERROR] {name}：{e}")
    else:
        try:
            collectors[args.source](start_date, end_date)
        except Exception as e:
            logger.error(f"{args.source} 收集失敗：{e}")
            print(f"  [ERROR] {e}")

    print("\n收集完成！")


if __name__ == "__main__":
    main()
