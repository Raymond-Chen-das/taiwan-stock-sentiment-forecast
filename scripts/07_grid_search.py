"""Phase 2c: Alpha/Theta 參數 Grid Search。

對 Baseline 和 LLM-BI 兩個版本同時進行參數調優，
搜尋最佳 alpha（樣本比例門檻）和 theta（機率門檻）組合。

參數空間：
  alpha: [0.15, 0.20, 0.25, 0.30]
  theta: [0.60, 0.65, 0.70, 0.75]
  window: [20, 40, 60] 天

共 4 × 4 × 3 = 48 組 × 2 版本 = 96 次實驗
"""

import sys
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.sds_detector import SDSDetector
from src.models.evaluator import Evaluator
from src.utils.config_loader import get_config, get_data_dir
from src.utils.logging_utils import setup_logger
from src.utils.preprocessing import train_only_zscore, add_next_day_target

logger = setup_logger("grid_search")


def load_baseline_data(test_start: str = "2025-01-01") -> pd.DataFrame:
    """載入 Baseline 版本資料（與 05_run_model.py 相同邏輯）。"""
    processed_dir = get_data_dir("processed")
    raw_dir = get_data_dir("raw/taiex")

    ai = pd.read_csv(processed_dir / "attention_index.csv", encoding="utf-8-sig")
    bi = pd.read_csv(processed_dir / "bullish_index.csv", encoding="utf-8-sig")
    pi = pd.read_csv(processed_dir / "propagation_index.csv", encoding="utf-8-sig")
    taiex = pd.read_csv(raw_dir / "taiex_daily.csv", encoding="utf-8-sig")

    for df in [ai, bi, pi, taiex]:
        df["date"] = pd.to_datetime(df["date"])

    merged = ai[["date", "ai_raw"]].merge(
        bi[["date", "bi_raw"]], on="date"
    ).merge(
        pi[["date", "pi_raw"]], on="date"
    ).merge(
        taiex[["date", "close", "daily_return", "trend_label"]], on="date"
    ).sort_values("date").reset_index(drop=True)

    merged = train_only_zscore(merged, test_start)
    merged = add_next_day_target(merged)

    return merged


def load_llm_data(test_start: str = "2025-01-01") -> pd.DataFrame:
    """載入 LLM-BI 版本資料（與 06_llm_bi_model.py 相同邏輯）。"""
    import json

    processed_dir = get_data_dir("processed")
    raw_dir = get_data_dir("raw/taiex")

    ai = pd.read_csv(processed_dir / "attention_index.csv", encoding="utf-8-sig")
    pi = pd.read_csv(processed_dir / "propagation_index.csv", encoding="utf-8-sig")
    taiex = pd.read_csv(raw_dir / "taiex_daily.csv", encoding="utf-8-sig")
    bi_orig = pd.read_csv(processed_dir / "bullish_index.csv", encoding="utf-8-sig")

    for df in [ai, pi, taiex, bi_orig]:
        df["date"] = pd.to_datetime(df["date"])

    # LLM BI: 訓練期用原始，測試期用 LLM 標籤
    with open(processed_dir / "llm_sentiment.json", "r", encoding="utf-8") as f:
        llm_results = json.load(f)

    llm_df = pd.DataFrame(llm_results)[["date", "sentiment"]]
    llm_df["date"] = pd.to_datetime(llm_df["date"])
    llm_df = llm_df.rename(columns={"sentiment": "bi_raw"})

    train_bi = bi_orig[bi_orig["date"] < test_start][["date", "bi_raw"]].copy()
    test_bi = llm_df.copy()
    bi_combined = pd.concat([train_bi, test_bi], ignore_index=True).sort_values("date").reset_index(drop=True)

    merged = ai[["date", "ai_raw"]].merge(
        bi_combined[["date", "bi_raw"]], on="date"
    ).merge(
        pi[["date", "pi_raw"]], on="date"
    ).merge(
        taiex[["date", "close", "daily_return", "trend_label"]], on="date"
    ).sort_values("date").reset_index(drop=True)

    merged = train_only_zscore(merged, test_start)
    merged = add_next_day_target(merged)

    return merged


def run_single(
    merged: pd.DataFrame,
    window_size: int,
    alpha: float,
    theta: float,
    test_start: str = "2025-01-01",
) -> dict:
    """對單一參數組合執行預測並回傳評估指標。"""
    ai = merged["ai_zscore"].values
    bi = merged["bi_zscore"].values
    pi = merged["pi_zscore"].values
    labels = merged["target"].values
    dates = merged["date"].values

    test_mask = merged["date"] >= pd.Timestamp(test_start)
    test_start_idx = test_mask.idxmax() if test_mask.any() else len(merged)

    detector = SDSDetector()
    predictions = []

    start_t = max(window_size + 1, test_start_idx)

    for t in range(start_t, len(merged)):
        model_results = detector.detect_all_models(
            ai[:t], bi[:t], pi[:t], labels[:t],
            window_size=window_size,
            alpha=alpha,
            theta=theta,
        )

        sample = np.array([ai[t], bi[t], pi[t]])
        pred = detector.predict_single(sample, model_results)

        predictions.append({
            "date": dates[t],
            "forecast": pred["forecast"],
            "actual": int(labels[t]),
        })

    pred_df = pd.DataFrame(predictions)

    if len(pred_df) == 0:
        return {"coverage": 0, "accuracy": 0, "up_accuracy": 0, "down_accuracy": 0,
                "valid_predictions": 0, "total_samples": 0}

    evaluator = Evaluator(pred_df)
    return evaluator.evaluate()


def main() -> None:
    """Grid Search 主流程。"""
    print("=" * 70)
    print("  Phase 2c: Alpha/Theta Grid Search")
    print("=" * 70)

    config = get_config()
    model_config = config["model"]

    alpha_values = model_config["sds_detection"]["alpha_values"]
    theta_values = model_config["sds_detection"]["theta_values"]
    window_sizes = [m * 20 for m in model_config["sliding_window_months"]]

    total_combos = len(alpha_values) * len(theta_values) * len(window_sizes)
    print(f"\n  參數空間: alpha {alpha_values} × theta {theta_values} × window {window_sizes}")
    print(f"  共 {total_combos} 組 × 2 版本 = {total_combos * 2} 次實驗")

    # 載入資料（只載入一次）
    print("\n[1] 載入資料...")
    t0 = time.time()
    baseline_data = load_baseline_data()
    llm_data = load_llm_data()
    print(f"  資料載入完成 ({time.time() - t0:.1f}s)")
    print(f"  Baseline: {len(baseline_data)} 筆, LLM-BI: {len(llm_data)} 筆")

    # Grid search
    results = []
    run_count = 0

    for version, data in [("baseline", baseline_data), ("llm_bi", llm_data)]:
        print(f"\n[{'2' if version == 'baseline' else '3'}] Grid Search — {version.upper()}")
        print("-" * 70)

        for w in window_sizes:
            for alpha, theta in itertools.product(alpha_values, theta_values):
                run_count += 1
                t0 = time.time()

                metrics = run_single(data, window_size=w, alpha=alpha, theta=theta)
                elapsed = time.time() - t0

                row = {
                    "version": version,
                    "window": w,
                    "alpha": alpha,
                    "theta": theta,
                    "coverage": metrics["coverage"],
                    "accuracy": metrics["accuracy"],
                    "up_accuracy": metrics["up_accuracy"],
                    "down_accuracy": metrics["down_accuracy"],
                    "valid_predictions": metrics["valid_predictions"],
                    "total_samples": metrics["total_samples"],
                    "elapsed_sec": round(elapsed, 1),
                }
                results.append(row)

                # 簡短進度
                print(
                    f"  [{run_count:3d}/{total_combos*2}] "
                    f"{version:>8s} w={w:2d} α={alpha:.2f} θ={theta:.2f} → "
                    f"cov={metrics['coverage']:.1%} acc={metrics['accuracy']:.1%} "
                    f"({metrics['valid_predictions']:3d}筆) {elapsed:.0f}s"
                )

    # 儲存完整結果
    results_df = pd.DataFrame(results)
    output_dir = get_data_dir("processed")
    results_df.to_csv(
        output_dir / "grid_search_results.csv",
        index=False, encoding="utf-8-sig",
    )

    # 分析與排名
    print("\n" + "=" * 70)
    print("  Grid Search 結果摘要")
    print("=" * 70)

    for version in ["baseline", "llm_bi"]:
        vdf = results_df[results_df["version"] == version]
        # 只看有足夠預測量的（coverage > 5%）
        vdf_valid = vdf[vdf["coverage"] > 0.05]

        if len(vdf_valid) == 0:
            print(f"\n  {version.upper()}: 無有效結果（覆蓋率均 < 5%）")
            continue

        print(f"\n  === {version.upper()} Top 5（按準確率排序，覆蓋率 > 5%）===")
        top5 = vdf_valid.nlargest(5, "accuracy")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(
                f"  #{i} w={int(row['window']):2d} α={row['alpha']:.2f} θ={row['theta']:.2f} | "
                f"覆蓋率={row['coverage']:.1%} 準確率={row['accuracy']:.1%} "
                f"看漲={row['up_accuracy']:.1%} 看跌={row['down_accuracy']:.1%} "
                f"({int(row['valid_predictions'])}筆)"
            )

        # 也顯示按 coverage × accuracy 綜合指標排序
        vdf_valid = vdf_valid.copy()
        vdf_valid["score"] = vdf_valid["coverage"] * vdf_valid["accuracy"]
        print(f"\n  === {version.upper()} Top 5（按覆蓋率×準確率綜合分排序）===")
        top5s = vdf_valid.nlargest(5, "score")
        for i, (_, row) in enumerate(top5s.iterrows(), 1):
            print(
                f"  #{i} w={int(row['window']):2d} α={row['alpha']:.2f} θ={row['theta']:.2f} | "
                f"覆蓋率={row['coverage']:.1%} 準確率={row['accuracy']:.1%} "
                f"綜合={row['score']:.4f}"
            )

    # Baseline vs LLM-BI 最佳配對對比
    print(f"\n  === 最佳配置 Baseline vs LLM-BI 對比 ===")
    for w in window_sizes:
        bl = results_df[(results_df["version"] == "baseline") & (results_df["window"] == w)]
        lm = results_df[(results_df["version"] == "llm_bi") & (results_df["window"] == w)]

        bl_valid = bl[bl["coverage"] > 0.05]
        lm_valid = lm[lm["coverage"] > 0.05]

        if len(bl_valid) > 0 and len(lm_valid) > 0:
            bl_best = bl_valid.loc[bl_valid["accuracy"].idxmax()]
            lm_best = lm_valid.loc[lm_valid["accuracy"].idxmax()]

            print(f"\n  w={w} ({w//20}M):")
            print(f"    Baseline 最佳: α={bl_best['alpha']:.2f} θ={bl_best['theta']:.2f} "
                  f"→ cov={bl_best['coverage']:.1%} acc={bl_best['accuracy']:.1%}")
            print(f"    LLM-BI  最佳: α={lm_best['alpha']:.2f} θ={lm_best['theta']:.2f} "
                  f"→ cov={lm_best['coverage']:.1%} acc={lm_best['accuracy']:.1%}")
            print(f"    Delta 準確率: {lm_best['accuracy'] - bl_best['accuracy']:+.1%}")

    print(f"\n  完整結果已儲存至 data/processed/grid_search_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
