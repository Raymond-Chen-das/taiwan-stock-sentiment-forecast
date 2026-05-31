"""Phase 2b: LLM 強化 BI 版本的 SDS 模型。

使用 Claude (Opus 4.6) 對 PTT 高互動文章全文+留言進行情緒分類，
重建 BI 指標，並與 baseline（推噓比 BI）進行 A/B 對比。

LLM-BI 定義：
  - 每個交易日選取互動數最高的文章
  - Claude 基於文章內容+留言分類為 bullish(+1)/neutral(0)/bearish(-1)
  - BI_llm = 該日情緒分數
  - 測試期：直接使用 LLM 標籤
  - 訓練期：無 LLM 標籤，回退到 push/boo ratio BI（保持訓練邏輯不變）
"""

import sys
import time
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
import json

logger = setup_logger("llm_bi_model")


def build_llm_bi(test_start: str = "2025-01-01") -> pd.DataFrame:
    """建構 LLM 強化版 BI。

    測試期使用 LLM 情緒分類，訓練期使用原始 push/boo ratio BI。

    Returns:
        包含 date, bi_raw 的完整 DataFrame。
    """
    processed_dir = get_data_dir("processed")

    # 載入原始 BI（訓練期使用）
    bi_orig = pd.read_csv(processed_dir / "bullish_index.csv", encoding="utf-8-sig")
    bi_orig["date"] = pd.to_datetime(bi_orig["date"])

    # 載入 LLM 情緒分類（測試期使用）
    with open(processed_dir / "llm_sentiment.json", "r", encoding="utf-8") as f:
        llm_results = json.load(f)

    llm_df = pd.DataFrame(llm_results)[["date", "sentiment"]]
    llm_df["date"] = pd.to_datetime(llm_df["date"])
    llm_df = llm_df.rename(columns={"sentiment": "bi_raw"})

    # 合併：訓練期用原始 BI，測試期用 LLM BI
    train_bi = bi_orig[bi_orig["date"] < test_start][["date", "bi_raw"]].copy()
    test_bi = llm_df.copy()

    combined = pd.concat([train_bi, test_bi], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # 檢查
    n_train = (combined["date"] < test_start).sum()
    n_test = (combined["date"] >= test_start).sum()
    print(f"  LLM-BI: 訓練期 {n_train} 筆 (push/boo ratio), 測試期 {n_test} 筆 (LLM)")

    # 測試期 BI 分佈
    test_vals = combined.loc[combined["date"] >= test_start, "bi_raw"]
    print(f"  測試期 BI 分佈: mean={test_vals.mean():.3f}, std={test_vals.std():.3f}")
    print(f"    bullish(+1): {(test_vals == 1).sum()}, neutral(0): {(test_vals == 0).sum()}, bearish(-1): {(test_vals == -1).sum()}")

    return combined


def load_and_merge_llm(test_start: str = "2025-01-01") -> pd.DataFrame:
    """載入並對齊所有指標（使用 LLM-BI）。"""
    processed_dir = get_data_dir("processed")
    raw_dir = get_data_dir("raw/taiex")

    ai = pd.read_csv(processed_dir / "attention_index.csv", encoding="utf-8-sig")
    pi = pd.read_csv(processed_dir / "propagation_index.csv", encoding="utf-8-sig")
    taiex = pd.read_csv(raw_dir / "taiex_daily.csv", encoding="utf-8-sig")

    for df in [ai, pi, taiex]:
        df["date"] = pd.to_datetime(df["date"])

    # 建構 LLM-BI
    bi_llm = build_llm_bi(test_start)

    # 合併
    merged = ai[["date", "ai_raw"]].merge(
        bi_llm[["date", "bi_raw"]], on="date"
    ).merge(
        pi[["date", "pi_raw"]], on="date"
    ).merge(
        taiex[["date", "close", "daily_return", "trend_label"]], on="date"
    ).sort_values("date").reset_index(drop=True)

    # Z-score（只用訓練集統計量）+ 預測目標 shift(-1)，共用 src.utils.preprocessing
    merged = train_only_zscore(merged, test_start)
    merged = add_next_day_target(merged)

    return merged


def run_single_config(
    merged: pd.DataFrame,
    window_size: int,
    alpha: float,
    theta: float,
    test_start: str = "2025-01-01",
) -> pd.DataFrame:
    """對單一參數組合執行預測。"""
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
            "t": t,
            "forecast": pred["forecast"],
            "actual": int(labels[t]),
            "support": pred["support"],
            "mean_gini": pred["mean_gini"],
            "reason": pred["reason"],
            "close": merged["close"].iloc[t],
        })

        if len(predictions) % 50 == 0:
            logger.info(
                f"  進度: {len(predictions)}/{len(merged) - start_t} "
                f"({len(predictions)/(len(merged)-start_t)*100:.0f}%)"
            )

    return pd.DataFrame(predictions)


def main() -> None:
    """主流程：LLM-BI vs Baseline 對比。"""
    print("=" * 60)
    print("  Phase 2b: LLM 強化 BI — A/B 對比")
    print("=" * 60)

    config = get_config()
    model_config = config["model"]

    # 載入 LLM-BI 版本資料
    print("\n[1] 載入 LLM-BI 版本資料...")
    merged = load_and_merge_llm()

    train = merged[merged["date"] < "2025-01-01"]
    test = merged[merged["date"] >= "2025-01-01"]
    print(f"  合併後: {len(merged)} 筆")
    print(f"  訓練期: {len(train)} 筆")
    print(f"  測試期: {len(test)} 筆")

    # 參數
    window_sizes = [m * 20 for m in model_config["sliding_window_months"]]
    alpha = model_config["sds_detection"]["alpha_values"][1]
    theta = model_config["sds_detection"]["theta_values"][1]

    print(f"\n[2] 開始 SDS 偵測 (LLM-BI)...")
    print(f"  滑動視窗: {window_sizes} 天")

    output_dir = get_data_dir("processed")
    all_results = []

    for w in window_sizes:
        print(f"\n  --- 視窗 {w} 天 ({w//20} 個月) ---")
        t0 = time.time()

        pred_df = run_single_config(merged, window_size=w, alpha=alpha, theta=theta)
        elapsed = time.time() - t0
        print(f"  完成: {len(pred_df)} 筆, {elapsed:.1f}s")

        if len(pred_df) > 0:
            evaluator = Evaluator(pred_df)
            metrics = evaluator.evaluate()
            print(f"  覆蓋率: {metrics['coverage']:.1%}")
            print(f"  準確率: {metrics['accuracy']:.1%}")
            print(f"  看漲準確率: {metrics['up_accuracy']:.1%} ({metrics['up_predictions']} 筆)")
            print(f"  看跌準確率: {metrics['down_accuracy']:.1%} ({metrics['down_predictions']} 筆)")

            pred_df["window_size"] = w
            all_results.append(pred_df)

            pred_df.to_csv(
                output_dir / f"predictions_llm_w{w}.csv",
                index=False, encoding="utf-8-sig",
            )

    # 載入 baseline 結果做對比
    print("\n" + "=" * 60)
    print("  A/B 對比: Baseline vs LLM-BI")
    print("=" * 60)

    for w in window_sizes:
        baseline_path = output_dir / f"predictions_w{w}.csv"
        llm_path = output_dir / f"predictions_llm_w{w}.csv"

        if baseline_path.exists() and llm_path.exists():
            bl = pd.read_csv(baseline_path)
            lm = pd.read_csv(llm_path)

            bl_eval = Evaluator(bl).evaluate()
            lm_eval = Evaluator(lm).evaluate()

            print(f"\n  w={w} ({w//20}M):")
            print(f"    {'':>20s} | {'Baseline':>10s} | {'LLM-BI':>10s} | {'Delta':>10s}")
            print(f"    {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

            cov_delta = lm_eval['coverage'] - bl_eval['coverage']
            acc_delta = lm_eval['accuracy'] - bl_eval['accuracy']

            print(f"    {'覆蓋率':>16s} | {bl_eval['coverage']:>9.1%} | {lm_eval['coverage']:>9.1%} | {cov_delta:>+9.1%}")
            print(f"    {'準確率':>16s} | {bl_eval['accuracy']:>9.1%} | {lm_eval['accuracy']:>9.1%} | {acc_delta:>+9.1%}")
            print(f"    {'看漲準確率':>14s} | {bl_eval['up_accuracy']:>9.1%} | {lm_eval['up_accuracy']:>9.1%} |")
            print(f"    {'看跌準確率':>14s} | {bl_eval['down_accuracy']:>9.1%} | {lm_eval['down_accuracy']:>9.1%} |")
            print(f"    {'預測筆數':>15s} | {bl_eval['valid_predictions']:>10d} | {lm_eval['valid_predictions']:>10d} |")

    print("\n" + "=" * 60)
    print("  Phase 2b 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
