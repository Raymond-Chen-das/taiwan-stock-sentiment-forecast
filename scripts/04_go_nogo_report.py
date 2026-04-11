"""Day 7: Go/No-Go 決策報告腳本。

自動檢查所有條件並輸出決策報告。
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy import stats

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("go_nogo_report")


def check_condition(name: str, passed: bool, detail: str) -> dict:
    """建立檢查結果。"""
    return {"name": name, "passed": passed, "detail": detail}


def main() -> None:
    """產生 Go/No-Go 報告。"""
    processed_dir = get_data_dir("processed")
    raw_dir = get_data_dir("raw")

    checks = []

    # 1. Google Trends 資料可取得
    gt_dir = raw_dir / "google_trends"
    gt_files = list(gt_dir.glob("trends_*.csv")) if gt_dir.exists() else []
    gt_ok = len(gt_files) > 0
    if gt_ok:
        total_rows = sum(len(pd.read_csv(f)) for f in gt_files)
        gt_ok = total_rows > 0
    checks.append(check_condition(
        "Google Trends 資料可取得",
        gt_ok,
        f"{'有' if gt_ok else '無'}資料檔案（{len(gt_files)} 個）",
    ))

    # 2. PTT 日均文章數 > 30
    ptt_path = raw_dir / "ptt" / "ptt_stock_articles.csv"
    if ptt_path.exists():
        ptt_df = pd.read_csv(ptt_path, encoding="utf-8-sig")
        ptt_df["date"] = pd.to_datetime(ptt_df["date"])
        daily_count = ptt_df.groupby("date").size()
        avg_articles = daily_count.mean()
        ptt_ok = avg_articles > 30
        checks.append(check_condition(
            "PTT 日均文章數 > 30",
            ptt_ok,
            f"日均 {avg_articles:.1f} 篇",
        ))
    else:
        checks.append(check_condition("PTT 日均文章數 > 30", False, "無資料"))

    # 3. 各指標與日報酬相關性
    taiex_path = raw_dir / "taiex" / "taiex_daily.csv"
    taiex_df = None
    if taiex_path.exists():
        taiex_df = pd.read_csv(taiex_path, encoding="utf-8-sig")
        taiex_df["date"] = pd.to_datetime(taiex_df["date"])

    for name, file_name, zscore_col in [
        ("AI", "attention_index.csv", "ai_zscore"),
        ("BI", "bullish_index.csv", "bi_zscore"),
        ("PI", "propagation_index.csv", "pi_zscore"),
    ]:
        idx_path = processed_dir / file_name
        if idx_path.exists() and taiex_df is not None:
            idx_df = pd.read_csv(idx_path, encoding="utf-8-sig")
            idx_df["date"] = pd.to_datetime(idx_df["date"])

            merged = taiex_df.merge(idx_df[["date", zscore_col]], on="date")
            valid = merged.dropna(subset=[zscore_col, "daily_return"])

            if len(valid) > 10:
                r, p = stats.pearsonr(valid[zscore_col], valid["daily_return"])
                corr_ok = abs(r) > 0.05
                checks.append(check_condition(
                    f"{name} 與日報酬 |r| > 0.05",
                    corr_ok,
                    f"r={r:.4f}, p={p:.2e}",
                ))
            else:
                checks.append(check_condition(
                    f"{name} 與日報酬 |r| > 0.05",
                    False,
                    "資料不足",
                ))
        else:
            checks.append(check_condition(
                f"{name} 與日報酬 |r| > 0.05",
                False,
                "無資料",
            ))

    # 7. 三指標最大互相關 < 0.70
    indicator_data = {}
    for name, file_name, zscore_col in [
        ("AI", "attention_index.csv", "ai_zscore"),
        ("BI", "bullish_index.csv", "bi_zscore"),
        ("PI", "propagation_index.csv", "pi_zscore"),
    ]:
        idx_path = processed_dir / file_name
        if idx_path.exists():
            df = pd.read_csv(idx_path, encoding="utf-8-sig")
            df["date"] = pd.to_datetime(df["date"])
            indicator_data[name] = df.set_index("date")[zscore_col]

    if len(indicator_data) >= 2:
        combined = pd.DataFrame(indicator_data).dropna()
        corr_matrix = combined.corr()
        # 取上三角（排除對角線）
        max_corr = 0.0
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                max_corr = max(max_corr, abs(corr_matrix.iloc[i, j]))

        intercorr_ok = max_corr < 0.70
        checks.append(check_condition(
            "三指標最大互相關 < 0.70",
            intercorr_ok,
            f"最大 |r| = {max_corr:.4f}",
        ))
    else:
        checks.append(check_condition(
            "三指標最大互相關 < 0.70",
            False,
            "指標不足",
        ))

    # 輸出報告
    report_lines = [
        "=" * 60,
        "  Go/No-Go 決策報告",
        f"  產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    all_passed = True
    for c in checks:
        icon = "[PASS]" if c["passed"] else "[FAIL]"
        if not c["passed"]:
            all_passed = False
        report_lines.append(f"  {icon} {c['name']}")
        report_lines.append(f"         {c['detail']}")
        report_lines.append("")

    report_lines.append("=" * 60)
    if all_passed:
        report_lines.append("  決策：GO - 所有條件通過，可進入 Phase 2")
    else:
        failed_count = sum(1 for c in checks if not c["passed"])
        report_lines.append(
            f"  決策：NO-GO - 有 {failed_count} 項未通過，需調整後重新評估"
        )
    report_lines.append("=" * 60)

    report = "\n".join(report_lines)
    print(report)

    # 儲存報告
    report_path = project_root / "experiments" / "go_nogo_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n報告已儲存至 {report_path}")


if __name__ == "__main__":
    main()
