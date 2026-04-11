"""多重過濾模組。

實作論文中的 Support、Unreliable Support、Gini Avg、Neighbors
等過濾參數，提高預測的可靠性。
"""

from typing import Dict, List, Optional

import numpy as np

from src.utils.logging_utils import setup_logger

logger = setup_logger("multi_filter")


class MultiFilter:
    """多重過濾器。

    結合多個過濾指標來篩選可靠的預測結果。

    Attributes:
        support_threshold: Support 門檻。
        unreliable_threshold: Unreliable support 門檻。
        gini_threshold: 平均 Gini 指數門檻。
        neighbors_align: 是否要求近鄰方向一致。
    """

    def __init__(
        self,
        support_threshold: int = 2,
        unreliable_threshold: int = 1,
        gini_threshold: float = 0.45,
        neighbors_align: bool = True,
    ) -> None:
        self.support_threshold = support_threshold
        self.unreliable_threshold = unreliable_threshold
        self.gini_threshold = gini_threshold
        self.neighbors_align = neighbors_align

    def filter(self, prediction: Dict) -> Dict:
        """對預測結果進行過濾。

        Args:
            prediction: 包含 model_details 的預測結果。

        Returns:
            過濾後的預測結果，加入 filtered_forecast 欄位。
        """
        model_details = prediction.get("model_details", [])

        # 計算 Support
        support = sum(1 for m in model_details if m.get("support", 0) == 1)

        # 計算 Unreliable Support
        unreliable = sum(
            1 for m in model_details
            if m.get("support", 0) == 1 and m.get("gini", 0.5) >= 0.5
        )

        # 計算 Mean Gini
        gini_values = [m.get("gini", 0.5) for m in model_details]
        mean_gini = np.mean(gini_values) if gini_values else 0.5

        # 計算 Neighbors 方向
        supported_forecasts = [
            m.get("forecast")
            for m in model_details
            if m.get("support", 0) == 1 and m.get("forecast") is not None
        ]

        # 過濾決策
        if support < self.support_threshold:
            filtered_forecast = 0
            filter_reason = f"support ({support}) < threshold ({self.support_threshold})"
        elif unreliable >= self.unreliable_threshold:
            filtered_forecast = 0
            filter_reason = f"unreliable support ({unreliable}) >= threshold"
        elif mean_gini > self.gini_threshold:
            filtered_forecast = 0
            filter_reason = f"mean gini ({mean_gini:.3f}) > threshold ({self.gini_threshold})"
        elif self.neighbors_align and supported_forecasts:
            neighbors_dirs = [
                m.get("neighbors", 0)
                for m in model_details
                if m.get("support", 0) == 1
            ]
            forecast_dir = np.sign(sum(supported_forecasts))
            neighbor_dir = np.sign(sum(neighbors_dirs))

            if forecast_dir != 0 and neighbor_dir != 0 and forecast_dir != neighbor_dir:
                filtered_forecast = 0
                filter_reason = "neighbors direction contradicts forecast"
            else:
                filtered_forecast = int(forecast_dir) if forecast_dir != 0 else 0
                filter_reason = "passed all filters"
        elif supported_forecasts:
            filtered_forecast = int(np.sign(sum(supported_forecasts)))
            filter_reason = "passed all filters"
        else:
            filtered_forecast = 0
            filter_reason = "no supported forecasts"

        prediction["filtered_forecast"] = filtered_forecast
        prediction["filter_reason"] = filter_reason
        prediction["filter_details"] = {
            "support": support,
            "unreliable_support": unreliable,
            "mean_gini": mean_gini,
        }

        return prediction
