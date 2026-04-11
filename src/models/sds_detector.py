"""情緒驅動子空間（SDS）偵測模組。

整合情緒空間模型與 PSO 搜尋，辨識情緒驅動子空間並進行預測。
實作論文 Section 4.3 的完整流程。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.sentiment_space import SentimentSpace
from src.models.pso_search import PSOSearcher
from src.utils.config_loader import get_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("sds_detector")


class SDSDetector:
    """情緒驅動子空間偵測器。

    對每個時間點，在滑動視窗內搜尋 SDS，
    並基於 SDS 的機率分佈進行趨勢預測。

    Attributes:
        space_model: 情緒空間模型。
        pso: PSO 搜尋器。
        alpha_values: 樣本比例門檻候選值。
        theta_values: 機率門檻候選值。
    """

    def __init__(self) -> None:
        config = get_config()
        model_config = config["model"]

        self.alpha_values: List[float] = model_config["sds_detection"]["alpha_values"]
        self.theta_values: List[float] = model_config["sds_detection"]["theta_values"]

        bt_config = model_config.get("boundary_tolerance", {})
        d_inner = bt_config.get("d_inner", 0.7)
        d_outer = bt_config.get("d_outer", 2.5)

        self.pso = PSOSearcher(
            n_particles=model_config["pso"]["n_particles"],
            max_iterations=model_config["pso"]["max_iterations"],
            w=model_config["pso"]["w_inertia"],
            c1=model_config["pso"]["c1_cognitive"],
            c2=model_config["pso"]["c2_social"],
        )

        self.filtering = model_config.get("filtering", {})

    def detect_all_models(
        self,
        ai: np.ndarray,
        bi: np.ndarray,
        pi: np.ndarray,
        labels: np.ndarray,
        window_size: int = 20,
        alpha: float = 0.20,
        theta: float = 0.65,
    ) -> List[Dict]:
        """對四個子模型進行 SDS 偵測。

        Model A: AI x BI
        Model B: AI x PI
        Model C: BI x PI
        Model D: AI x BI x PI

        Args:
            ai: 關注度指數序列。
            bi: 看漲指數序列。
            pi: 傳播指數序列。
            labels: 趨勢標籤序列。
            window_size: 滑動視窗大小。
            alpha: 樣本比例門檻。
            theta: 機率門檻。

        Returns:
            四個模型的 SDS 偵測結果列表。
        """
        models = {
            "Model_A": np.column_stack([ai, bi]),
            "Model_B": np.column_stack([ai, pi]),
            "Model_C": np.column_stack([bi, pi]),
            "Model_D": np.column_stack([ai, bi, pi]),
        }

        results = []
        for name, data in models.items():
            logger.info(f"偵測 {name} 的 SDS")
            space = SentimentSpace(window_size=window_size)
            space_info = space.build_space(data, labels, len(data))

            if not space_info:
                results.append({"model": name, "sds_found": False})
                continue

            pso_result = self.pso.search(
                space_info["window_data"],
                space_info["window_labels"],
                space_info["center"],
                alpha=alpha,
                theta=theta,
            )

            results.append({
                "model": name,
                "sds_found": pso_result["sds_info"] is not None,
                "pso_result": pso_result,
                "space_info": space_info,
            })

        return results

    def predict_single(
        self,
        sample: np.ndarray,
        model_results: List[Dict],
    ) -> Dict:
        """基於四個子模型的結果對單一樣本進行預測。

        Args:
            sample: 待預測樣本 [ai, bi, pi]。
            model_results: detect_all_models 的結果。

        Returns:
            預測結果字典。
        """
        support_threshold = self.filtering.get("support_threshold", 2)
        unreliable_threshold = self.filtering.get("unreliable_support_threshold", 1)
        gini_threshold = self.filtering.get("gini_avg_threshold", 0.45)

        model_evals = []
        for result in model_results:
            if not result.get("sds_found"):
                model_evals.append({
                    "model": result["model"],
                    "support": 0,
                    "forecast": None,
                    "gini": 0.5,
                })
                continue

            pso_result = result["pso_result"]
            space_info = result["space_info"]

            # 取出對應維度的樣本
            model_name = result["model"]
            if model_name == "Model_A":
                s = sample[:2]
            elif model_name == "Model_B":
                s = sample[[0, 2]]
            elif model_name == "Model_C":
                s = sample[1:3]
            else:
                s = sample

            space = SentimentSpace()
            eval_result = space.evaluate_sample(
                s,
                space_info,
                pso_result["best_radii"],
            )

            # 判斷 Support
            sds_info = pso_result["sds_info"]
            if sds_info and eval_result["location"] == sds_info["space"]:
                support = 1
                if sds_info["p_up"] > sds_info["p_down"]:
                    forecast = 1.0
                else:
                    forecast = -1.0
            else:
                support = 0
                forecast = None

            model_evals.append({
                "model": model_name,
                "support": support,
                "forecast": forecast,
                "gini": eval_result["gini_index"],
                "neighbors": eval_result["neighbors"],
                "location": eval_result["location"],
            })

        # 彙總四個模型
        total_support = sum(e["support"] for e in model_evals)
        unreliable = sum(
            1 for e in model_evals
            if e["gini"] >= 0.5 and e["support"] == 1
        )
        mean_gini = np.mean([e["gini"] for e in model_evals])

        forecasts = [e["forecast"] for e in model_evals if e["forecast"] is not None]

        # 過濾條件
        if total_support < support_threshold:
            final_forecast = 0
            reason = "support 不足"
        elif unreliable >= unreliable_threshold:
            final_forecast = 0
            reason = "unreliable support 過多"
        elif mean_gini > gini_threshold:
            final_forecast = 0
            reason = "Gini 指數過高（隨機性高）"
        elif forecasts:
            final_forecast = int(np.sign(sum(forecasts)))
            reason = "sentiment-driven"
        else:
            final_forecast = 0
            reason = "無有效預測"

        return {
            "forecast": final_forecast,
            "reason": reason,
            "support": total_support,
            "unreliable_support": unreliable,
            "mean_gini": mean_gini,
            "model_details": model_evals,
        }

    def run_prediction(
        self,
        ai_series: np.ndarray,
        bi_series: np.ndarray,
        pi_series: np.ndarray,
        labels: np.ndarray,
        window_size: int = 20,
        alpha: float = 0.20,
        theta: float = 0.65,
    ) -> pd.DataFrame:
        """對整個時間序列進行逐步預測。

        Args:
            ai_series: 關注度指數序列。
            bi_series: 看漲指數序列。
            pi_series: 傳播指數序列。
            labels: 趨勢標籤序列。
            window_size: 滑動視窗大小。
            alpha: 樣本比例門檻。
            theta: 機率門檻。

        Returns:
            預測結果 DataFrame。
        """
        n = len(ai_series)
        predictions = []

        for t in range(window_size + 1, n):
            if t % 50 == 0:
                logger.info(f"預測進度：{t}/{n}")

            # 使用 [t-w, t-1] 的視窗建構模型
            model_results = self.detect_all_models(
                ai_series[:t],
                bi_series[:t],
                pi_series[:t],
                labels[:t],
                window_size=window_size,
                alpha=alpha,
                theta=theta,
            )

            # 預測 t+1
            sample = np.array([ai_series[t], bi_series[t], pi_series[t]])
            pred = self.predict_single(sample, model_results)

            predictions.append({
                "t": t,
                "forecast": pred["forecast"],
                "actual": labels[t] if t < len(labels) else None,
                "support": pred["support"],
                "mean_gini": pred["mean_gini"],
                "reason": pred["reason"],
            })

        return pd.DataFrame(predictions)
