"""模型評估模組。

計算預測準確度、覆蓋率、SDS 偵測率等指標。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logger

logger = setup_logger("evaluator")


class Evaluator:
    """模型評估器。

    計算各項預測績效指標。

    Attributes:
        predictions: 預測結果 DataFrame。
    """

    def __init__(self, predictions: pd.DataFrame) -> None:
        self.predictions = predictions.copy()

    def evaluate(self) -> Dict:
        """計算完整評估指標。

        Returns:
            包含各項指標的字典。
        """
        df = self.predictions.dropna(subset=["actual"])

        # 所有預測的準確率（包含 0 預測）
        total = len(df)
        if total == 0:
            return {"error": "無有效預測"}

        # 有效預測（forecast != 0）
        valid = df[df["forecast"] != 0]
        n_valid = len(valid)

        # 覆蓋率
        coverage = n_valid / total if total > 0 else 0

        # 準確率（僅計算有效預測）
        if n_valid > 0:
            correct = (valid["forecast"] == valid["actual"]).sum()
            accuracy = correct / n_valid
        else:
            accuracy = 0.0

        # 分方向準確率
        up_preds = valid[valid["forecast"] == 1]
        down_preds = valid[valid["forecast"] == -1]

        up_accuracy = (
            (up_preds["actual"] == 1).sum() / len(up_preds)
            if len(up_preds) > 0
            else 0.0
        )
        down_accuracy = (
            (down_preds["actual"] == -1).sum() / len(down_preds)
            if len(down_preds) > 0
            else 0.0
        )

        # SDS 偵測率
        sds_rate = coverage  # 等同於覆蓋率

        result = {
            "total_samples": total,
            "valid_predictions": n_valid,
            "coverage": coverage,
            "accuracy": accuracy,
            "up_predictions": len(up_preds),
            "up_accuracy": up_accuracy,
            "down_predictions": len(down_preds),
            "down_accuracy": down_accuracy,
            "sds_detection_rate": sds_rate,
        }

        # 如果有 filtered_forecast
        if "filtered_forecast" in df.columns:
            filtered_valid = df[df["filtered_forecast"] != 0]
            n_filtered = len(filtered_valid)

            if n_filtered > 0:
                filtered_correct = (
                    filtered_valid["filtered_forecast"]
                    == filtered_valid["actual"]
                ).sum()
                result["filtered_accuracy"] = filtered_correct / n_filtered
                result["filtered_coverage"] = n_filtered / total
                result["filtered_valid_predictions"] = n_filtered
            else:
                result["filtered_accuracy"] = 0.0
                result["filtered_coverage"] = 0.0

        logger.info(
            f"評估結果 — 覆蓋率: {coverage:.2%}, "
            f"準確率: {accuracy:.2%}, "
            f"有效預測: {n_valid}/{total}"
        )

        return result

    def generate_report(self) -> str:
        """產生評估報告文字。

        Returns:
            格式化的報告字串。
        """
        metrics = self.evaluate()

        lines = [
            "=" * 60,
            "情緒空間模型預測評估報告",
            "=" * 60,
            f"總樣本數：{metrics['total_samples']}",
            f"有效預測數：{metrics['valid_predictions']}",
            f"覆蓋率：{metrics['coverage']:.2%}",
            f"預測準確率：{metrics['accuracy']:.2%}",
            "",
            f"看漲預測：{metrics['up_predictions']} 筆，"
            f"準確率 {metrics['up_accuracy']:.2%}",
            f"看跌預測：{metrics['down_predictions']} 筆，"
            f"準確率 {metrics['down_accuracy']:.2%}",
            "",
            f"SDS 偵測率：{metrics['sds_detection_rate']:.2%}",
        ]

        if "filtered_accuracy" in metrics:
            lines.extend([
                "",
                "--- 過濾後 ---",
                f"過濾後有效預測：{metrics.get('filtered_valid_predictions', 0)}",
                f"過濾後覆蓋率：{metrics.get('filtered_coverage', 0):.2%}",
                f"過濾後準確率：{metrics.get('filtered_accuracy', 0):.2%}",
            ])

        lines.append("=" * 60)

        report = "\n".join(lines)
        logger.info("\n" + report)
        return report
