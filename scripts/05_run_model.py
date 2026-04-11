"""Phase 2: 情緒空間 SDS 模型建構與預測。

對齊 AI/BI/PI 三個指標與 TAIEX 趨勢標籤，
使用滑動視窗逐步進行 SDS 偵測與趨勢預測。

論文模型架構：
  Model A: AI × BI（關注度 × 看漲度）
  Model B: AI × PI（關注度 × 傳播度）
  Model C: BI × PI（看漲度 × 傳播度）
  Model D: AI × BI × PI（三維情緒空間）

時間分割：
  訓練期：2021-01-04 ~ 2024-12-31（~970 個交易日）
  測試期：2025-01-01 ~ 2026-01-30（~262 個交易日）
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

logger = setup_logger("run_model")


def load_and_merge(test_start: str = "2025-01-01") -> pd.DataFrame:
    """載入並對齊所有指標與 TAIEX 資料。

    重要的方法學修正：
    1. Z-score 只用訓練集的 mean/std 計算，避免資料洩漏
    2. 預測目標 shift(-1)，確保用 t 天情緒預測 t+1 天漲跌

    Args:
        test_start: 測試期起始日期，用於分割訓練集計算 z-score。

    Returns:
        合併後的 DataFrame，index 為日期順序。
    """
    processed_dir = get_data_dir("processed")
    raw_dir = get_data_dir("raw/taiex")

    ai = pd.read_csv(processed_dir / "attention_index.csv", encoding="utf-8-sig")
    bi = pd.read_csv(processed_dir / "bullish_index.csv", encoding="utf-8-sig")
    pi = pd.read_csv(processed_dir / "propagation_index.csv", encoding="utf-8-sig")
    taiex = pd.read_csv(raw_dir / "taiex_daily.csv", encoding="utf-8-sig")

    for df in [ai, bi, pi, taiex]:
        df["date"] = pd.to_datetime(df["date"])

    # 使用 raw 值合併（不使用 builder 產出的 zscore）
    merged = ai[["date", "ai_raw"]].merge(
        bi[["date", "bi_raw"]], on="date"
    ).merge(
        pi[["date", "pi_raw"]], on="date"
    ).merge(
        taiex[["date", "close", "daily_return", "trend_label"]], on="date"
    ).sort_values("date").reset_index(drop=True)

    # 修正 1: Z-score 只用訓練集的 mean/std（防止資料洩漏）
    train_mask = merged["date"] < pd.Timestamp(test_start)
    for raw_col, z_col in [("ai_raw", "ai_zscore"), ("bi_raw", "bi_zscore"), ("pi_raw", "pi_zscore")]:
        train_mean = merged.loc[train_mask, raw_col].mean()
        train_std = merged.loc[train_mask, raw_col].std()
        merged[z_col] = (merged[raw_col] - train_mean) / train_std

    logger.info(
        f"Z-score 使用訓練集統計量（< {test_start}，{train_mask.sum()} 筆）"
    )

    # 修正 2: 預測目標的時間對齊（論文 Section 4.2.4）
    # 原論文定義 m_{t+1} = sign(CP_{t+1} - CP_t)
    # 即：用 t 天的情緒指標預測 t+1 天的漲跌
    merged["target"] = merged["trend_label"].shift(-1)
    merged = merged.dropna(subset=["target"]).reset_index(drop=True)
    merged["target"] = merged["target"].astype(int)

    return merged


def run_single_config(
    merged: pd.DataFrame,
    window_size: int,
    alpha: float,
    theta: float,
    test_start: str = "2025-01-01",
) -> pd.DataFrame:
    """對單一參數組合執行預測。

    Args:
        merged: 合併後的資料。
        window_size: 滑動視窗大小（交易日數）。
        alpha: 樣本比例門檻。
        theta: 機率門檻。
        test_start: 測試期起始日期。

    Returns:
        預測結果 DataFrame。
    """
    ai = merged["ai_zscore"].values
    bi = merged["bi_zscore"].values
    pi = merged["pi_zscore"].values
    # target = 明天的漲跌（已在 load_and_merge 中 shift(-1)）
    # 視窗內的 labels 也用 target，確保：
    #   視窗 [t-w, t-1] 的 label 是「各自隔天的漲跌」
    #   這些 label 在時間 t 已經是已知資訊（歷史事實）
    labels = merged["target"].values
    dates = merged["date"].values

    # 找到測試期起始索引
    test_mask = merged["date"] >= pd.Timestamp(test_start)
    test_start_idx = test_mask.idxmax() if test_mask.any() else len(merged)

    detector = SDSDetector()
    predictions = []

    # 確保測試期每個點都有足夠的視窗
    start_t = max(window_size + 1, test_start_idx)

    for t in range(start_t, len(merged)):
        # 使用 [t-w, t-1] 的情緒指標和對應的「隔天漲跌」標籤
        # 注意：labels[t-w:t] 中的每個 label[i] = 第 i+1 天的漲跌
        # 這些在時間 t 都是已知的歷史事實，無資料洩漏
        model_results = detector.detect_all_models(
            ai[:t], bi[:t], pi[:t], labels[:t],
            window_size=window_size,
            alpha=alpha,
            theta=theta,
        )

        # 用 t 天的情緒指標預測 t+1 天的趨勢
        sample = np.array([ai[t], bi[t], pi[t]])
        pred = detector.predict_single(sample, model_results)

        predictions.append({
            "date": dates[t],
            "t": t,
            "forecast": pred["forecast"],
            "actual": int(labels[t]),  # target[t] = 明天的實際漲跌
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
    """主流程：執行多組參數的 SDS 模型預測。"""
    print("=" * 60)
    print("  Phase 2: 情緒空間 SDS 模型")
    print("=" * 60)

    config = get_config()
    model_config = config["model"]

    # 載入資料
    print("\n[1] 載入並對齊資料...")
    merged = load_and_merge()

    train = merged[merged["date"] < "2025-01-01"]
    test = merged[merged["date"] >= "2025-01-01"]
    print(f"  合併後: {len(merged)} 筆")
    print(f"  訓練期: {len(train)} 筆 ({train['date'].min().date()} ~ {train['date'].max().date()})")
    print(f"  測試期: {len(test)} 筆 ({test['date'].min().date()} ~ {test['date'].max().date()})")

    # 參數組合
    window_months = model_config["sliding_window_months"]
    # 月份轉交易日數（每月約 20-22 個交易日）
    window_sizes = [m * 20 for m in window_months]

    # 先用預設 alpha/theta 跑，後續可做 grid search
    alpha = model_config["sds_detection"]["alpha_values"][1]  # 0.20
    theta = model_config["sds_detection"]["theta_values"][1]  # 0.65

    print(f"\n[2] 開始 SDS 偵測與預測...")
    print(f"  滑動視窗: {window_sizes} 天")
    print(f"  alpha={alpha}, theta={theta}")

    output_dir = get_data_dir("processed")
    all_results = []

    for w in window_sizes:
        print(f"\n  --- 視窗大小: {w} 天 ({w//20} 個月) ---")
        t0 = time.time()

        pred_df = run_single_config(
            merged, window_size=w, alpha=alpha, theta=theta,
        )

        elapsed = time.time() - t0
        print(f"  完成: {len(pred_df)} 筆預測, 耗時 {elapsed:.1f} 秒")

        # 基本評估
        if len(pred_df) > 0:
            evaluator = Evaluator(pred_df)
            metrics = evaluator.evaluate()
            print(f"  覆蓋率: {metrics['coverage']:.1%}")
            print(f"  準確率: {metrics['accuracy']:.1%}")
            print(f"  看漲準確率: {metrics['up_accuracy']:.1%} ({metrics['up_predictions']} 筆)")
            print(f"  看跌準確率: {metrics['down_accuracy']:.1%} ({metrics['down_predictions']} 筆)")

            pred_df["window_size"] = w
            pred_df["alpha"] = alpha
            pred_df["theta"] = theta
            all_results.append(pred_df)

            # 儲存個別結果
            pred_df.to_csv(
                output_dir / f"predictions_w{w}.csv",
                index=False, encoding="utf-8-sig",
            )

    # 合併所有結果
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(
            output_dir / "predictions_all.csv",
            index=False, encoding="utf-8-sig",
        )
        print(f"\n  所有預測結果已儲存至 data/processed/predictions_*.csv")

    print("\n" + "=" * 60)
    print("  Phase 2 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
