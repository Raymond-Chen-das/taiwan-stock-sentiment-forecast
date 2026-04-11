"""情緒空間模型（Sentiment Space Model）。

實作論文中的核心概念：將情緒指標映射到多維幾何空間，
透過橢球體分割內外空間，辨識情緒驅動子空間。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logger

logger = setup_logger("sentiment_space")


class SentimentSpace:
    """情緒空間模型。

    將情緒樣本映射到 n 維空間，計算幾何中心與距離，
    並透過高維球面（HDS）分割內外空間。

    Attributes:
        window_size: 滑動視窗大小（交易日數）。
        d_inner: 內部邊界容忍度。
        d_outer: 外部邊界容忍度。
    """

    def __init__(
        self,
        window_size: int = 20,
        d_inner: float = 0.7,
        d_outer: float = 2.5,
    ) -> None:
        self.window_size = window_size
        self.d_inner = d_inner
        self.d_outer = d_outer

    def build_space(
        self,
        sentiment_data: np.ndarray,
        labels: np.ndarray,
        t: int,
    ) -> Dict:
        """建構時間 t 的情緒空間。

        Args:
            sentiment_data: 情緒特徵矩陣 (T x n)。
            labels: 趨勢標籤向量 (T,)。
            t: 當前時間點索引。

        Returns:
            情緒空間資訊字典。
        """
        w = self.window_size

        if t < w:
            return {}

        # 取得滑動視窗內的資料 [t-w, t-1]
        window_data = sentiment_data[t - w:t]
        window_labels = labels[t - w:t]

        # 幾何中心
        center = self._compute_center(window_data)

        # 計算每個樣本到中心的距離
        distances = self._compute_distances(window_data, center)

        return {
            "window_data": window_data,
            "window_labels": window_labels,
            "center": center,
            "distances": distances,
            "t": t,
        }

    def _compute_center(self, data: np.ndarray) -> np.ndarray:
        """計算幾何中心（最小化所有樣本到中心的距離和）。

        使用均值作為近似解。

        Args:
            data: 樣本矩陣。

        Returns:
            幾何中心向量。
        """
        return np.mean(data, axis=0)

    def _compute_distances(
        self,
        data: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """計算樣本到中心的距離。

        Args:
            data: 樣本矩陣。
            center: 中心向量。

        Returns:
            距離向量。
        """
        return np.sqrt(np.sum((data - center) ** 2, axis=1))

    def compute_ellipsoid_distance(
        self,
        sample: np.ndarray,
        center: np.ndarray,
        radii: np.ndarray,
    ) -> float:
        """計算樣本到橢球中心的標準化距離。

        D(S, R, C) = sum((s_i - c_i)^2 / r_i^2)

        Args:
            sample: 樣本向量。
            center: 中心向量。
            radii: 半徑向量。

        Returns:
            標準化距離。
        """
        return float(np.sum((sample - center) ** 2 / (radii ** 2)))

    def partition_space(
        self,
        data: np.ndarray,
        center: np.ndarray,
        radii: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """使用 HDS 將空間分割為內外區域。

        Args:
            data: 樣本矩陣。
            center: 中心向量。
            radii: 半徑向量。

        Returns:
            (inner_mask, outer_mask) 布林陣列。
        """
        distances = np.array([
            self.compute_ellipsoid_distance(s, center, radii)
            for s in data
        ])

        inner_mask = distances <= 1.0
        outer_mask = distances > 1.0

        return inner_mask, outer_mask

    def compute_subspace_probability(
        self,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """計算子空間內的趨勢機率分佈。

        Args:
            labels: 子空間內的趨勢標籤。

        Returns:
            包含上漲/下跌機率的字典。
        """
        if len(labels) == 0:
            return {"p_up": 0.0, "p_down": 0.0, "p_stable": 0.0}

        n = len(labels)
        return {
            "p_up": float(np.sum(labels == 1)) / n,
            "p_down": float(np.sum(labels == -1)) / n,
            "p_stable": float(np.sum(labels == 0)) / n,
        }

    def compute_gini_index(
        self,
        labels: np.ndarray,
        k: int = 2,
    ) -> float:
        """計算 Gini 指數（衡量隨機性）。

        G = 1 - sum(p_i^2)

        Args:
            labels: 標籤陣列。
            k: 近鄰數量。

        Returns:
            Gini 指數（0 = 純淨，0.5 = 最大隨機性）。
        """
        if len(labels) == 0:
            return 0.5

        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return float(1 - np.sum(probs ** 2))

    def evaluate_sample(
        self,
        sample: np.ndarray,
        space_info: Dict,
        radii: np.ndarray,
    ) -> Dict:
        """評估單一樣本的情緒空間位置與預測。

        Args:
            sample: 待評估的樣本向量。
            space_info: 情緒空間資訊。
            radii: HDS 半徑向量。

        Returns:
            評估結果字典。
        """
        center = space_info["center"]
        window_data = space_info["window_data"]
        window_labels = space_info["window_labels"]

        # 計算樣本到中心的標準化距離
        dist = self.compute_ellipsoid_distance(sample, center, radii)

        # 判斷內外空間
        location = "inner" if dist <= 1.0 else "outer"

        # 空間分割
        inner_mask, outer_mask = self.partition_space(
            window_data, center, radii
        )

        # 計算內外空間的機率分佈
        inner_probs = self.compute_subspace_probability(
            window_labels[inner_mask]
        )
        outer_probs = self.compute_subspace_probability(
            window_labels[outer_mask]
        )

        # 計算 Gini 指數
        gini = self.compute_gini_index(window_labels)

        # 找 K 近鄰
        dists_to_data = np.sqrt(
            np.sum((window_data - sample) ** 2, axis=1)
        )
        k_nearest_idx = np.argsort(dists_to_data)[:4]
        neighbor_labels = window_labels[k_nearest_idx]
        neighbors_direction = float(np.sign(np.sum(neighbor_labels)))

        return {
            "distance": dist,
            "location": location,
            "inner_probs": inner_probs,
            "outer_probs": outer_probs,
            "gini_index": gini,
            "neighbors": neighbors_direction,
        }
