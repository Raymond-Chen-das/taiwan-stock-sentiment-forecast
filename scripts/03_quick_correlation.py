"""Day 5-6: 快速相關性檢驗腳本。

計算三個指標與 TAIEX 的相關係數，繪製熱力圖和時間序列圖。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("quick_correlation")


def load_data() -> dict:
    """載入已建構的指標資料。"""
    processed_dir = get_data_dir("processed")
    raw_dir = get_data_dir("raw/taiex")

    data = {}

    # TAIEX
    taiex_path = raw_dir / "taiex_daily.csv"
    if taiex_path.exists():
        data["taiex"] = pd.read_csv(taiex_path, encoding="utf-8-sig")
        data["taiex"]["date"] = pd.to_datetime(data["taiex"]["date"])
    else:
        print("  [警告] 找不到 TAIEX 資料")

    # AI
    ai_path = processed_dir / "attention_index.csv"
    if ai_path.exists():
        data["ai"] = pd.read_csv(ai_path, encoding="utf-8-sig")
        data["ai"]["date"] = pd.to_datetime(data["ai"]["date"])

    # BI
    bi_path = processed_dir / "bullish_index.csv"
    if bi_path.exists():
        data["bi"] = pd.read_csv(bi_path, encoding="utf-8-sig")
        data["bi"]["date"] = pd.to_datetime(data["bi"]["date"])

    # PI
    pi_path = processed_dir / "propagation_index.csv"
    if pi_path.exists():
        data["pi"] = pd.read_csv(pi_path, encoding="utf-8-sig")
        data["pi"]["date"] = pd.to_datetime(data["pi"]["date"])

    return data


def compute_correlations(data: dict) -> None:
    """計算相關係數。"""
    print("\n" + "=" * 60)
    print("  相關性分析")
    print("=" * 60)

    if "taiex" not in data:
        print("  無 TAIEX 資料，無法計算相關性")
        return

    taiex = data["taiex"]
    results = []

    for name, zscore_col in [("AI", "ai_zscore"), ("BI", "bi_zscore"), ("PI", "pi_zscore")]:
        if name.lower() not in data:
            continue

        df = data[name.lower()]
        if zscore_col not in df.columns:
            continue

        # 合併
        merged = taiex.merge(df[["date", zscore_col]], on="date", how="inner")

        if len(merged) < 10:
            print(f"  {name}: 合併後資料不足（{len(merged)} 筆）")
            continue

        # 與收盤價的相關
        r_close, p_close = stats.pearsonr(
            merged[zscore_col].dropna(),
            merged["close"].loc[merged[zscore_col].notna()],
        )

        # 與日報酬的相關
        valid = merged.dropna(subset=[zscore_col, "daily_return"])
        if len(valid) > 0:
            r_return, p_return = stats.pearsonr(
                valid[zscore_col], valid["daily_return"]
            )
        else:
            r_return, p_return = 0, 1

        results.append({
            "指標": name,
            "與收盤價 r": f"{r_close:.4f}",
            "p-value": f"{p_close:.2e}",
            "與日報酬 r": f"{r_return:.4f}",
            "p-value ": f"{p_return:.2e}",
        })

        print(f"\n  {name}:")
        print(f"    與收盤價: r={r_close:.4f}, p={p_close:.2e}")
        print(f"    與日報酬: r={r_return:.4f}, p={p_return:.2e}")

    # 指標間相關矩陣
    print("\n  --- 指標間相關矩陣 ---")
    indicator_data = {}
    for name, zscore_col in [("AI", "ai_zscore"), ("BI", "bi_zscore"), ("PI", "pi_zscore")]:
        if name.lower() in data and zscore_col in data[name.lower()].columns:
            df = data[name.lower()]
            indicator_data[name] = df.set_index("date")[zscore_col]

    if len(indicator_data) >= 2:
        indicator_df = pd.DataFrame(indicator_data)
        corr_matrix = indicator_df.corr()
        print(corr_matrix.to_string())

        # 繪製熱力圖
        try:
            from src.visualization.plots import (
                plot_correlation_heatmap,
                plot_time_series_comparison,
            )

            output_dir = get_data_dir("processed")

            fig = plot_correlation_heatmap(
                corr_matrix,
                title="情緒指標相關矩陣",
                output_path=str(output_dir / "correlation_heatmap.html"),
            )
            print(f"\n  熱力圖已儲存至 data/processed/correlation_heatmap.html")

        except Exception as e:
            print(f"\n  繪圖失敗：{e}")

    # Go/No-Go 建議
    print("\n" + "=" * 60)
    print("  Go/No-Go 初步建議")
    print("=" * 60)

    if results:
        for r in results:
            r_val = abs(float(r["與日報酬 r"]))
            status = "PASS" if r_val > 0.05 else "WEAK"
            print(f"  {r['指標']}: |r| = {r_val:.4f} {'[OK]' if status == 'PASS' else '[需注意]'}")
    else:
        print("  資料不足，無法進行分析")


def main() -> None:
    """主流程。"""
    print("=" * 60)
    print("  快速相關性檢驗")
    print("=" * 60)

    data = load_data()
    print(f"  已載入 {len(data)} 個資料來源")

    compute_correlations(data)


if __name__ == "__main__":
    main()
