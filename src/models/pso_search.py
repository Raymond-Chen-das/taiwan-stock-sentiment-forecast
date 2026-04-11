"""粒子群最佳化（PSO）搜尋模組。

使用 PSO 演算法在情緒空間中搜尋情緒驅動子空間（SDS）。
基於論文 Algorithm 1 的實作。
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logging_utils import setup_logger

logger = setup_logger("pso_search")


class PSOSearcher:
    """粒子群最佳化搜尋器。

    搜尋最優的 HDS 半徑向量，使得子空間內的
    機率偏斜度最大化。

    Attributes:
        n_particles: 粒子數量。
        max_iterations: 最大迭代次數。
        w: 慣性權重。
        c1: 認知學習因子。
        c2: 社會學習因子。
    """

    def __init__(
        self,
        n_particles: int = 30,
        max_iterations: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ) -> None:
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def search(
        self,
        sentiment_data: np.ndarray,
        labels: np.ndarray,
        center: np.ndarray,
        alpha: float = 0.20,
        theta: float = 0.65,
    ) -> Dict:
        """搜尋最優的 SDS 半徑。

        Args:
            sentiment_data: 情緒空間樣本 (w x n)。
            labels: 趨勢標籤 (w,)。
            center: 幾何中心。
            alpha: 最小樣本比例門檻。
            theta: 機率分佈顯著性門檻。

        Returns:
            搜尋結果字典，包含最佳半徑與適應度。
        """
        n_dims = sentiment_data.shape[1]
        n_total = len(sentiment_data)

        # 定義搜尋邊界
        r_min, r_max = self._define_boundaries(sentiment_data)

        # 初始化粒子
        positions = self._initialize_positions(r_min, r_max)
        velocities = self._initialize_velocities(r_min, r_max)
        v_max = (r_max - r_min) * 0.5

        # 記錄最佳位置
        personal_best_pos = positions.copy()
        personal_best_fit = np.full(self.n_particles, -np.inf)
        global_best_pos = positions[0].copy()
        global_best_fit = -np.inf

        # 記錄最佳 SDS 資訊
        best_sds_info = None

        for iteration in range(self.max_iterations):
            for k in range(self.n_particles):
                # 計算適應度
                fitness, sds_info = self._evaluate_fitness(
                    positions[k],
                    sentiment_data,
                    labels,
                    center,
                    alpha,
                    theta,
                    n_total,
                )

                # 更新個體最佳
                if fitness > personal_best_fit[k]:
                    personal_best_fit[k] = fitness
                    personal_best_pos[k] = positions[k].copy()

                # 更新全域最佳
                if fitness > global_best_fit:
                    global_best_fit = fitness
                    global_best_pos = positions[k].copy()
                    best_sds_info = sds_info

            # 更新速度和位置
            for k in range(self.n_particles):
                r1 = np.random.random(n_dims)
                r2 = np.random.random(n_dims)

                velocities[k] = (
                    self.w * velocities[k]
                    + self.c1 * r1 * (personal_best_pos[k] - positions[k])
                    + self.c2 * r2 * (global_best_pos - positions[k])
                )

                # 限制速度
                velocities[k] = np.clip(velocities[k], -v_max, v_max)

                # 更新位置
                positions[k] = positions[k] + velocities[k]

                # 限制在邊界內
                positions[k] = np.clip(positions[k], r_min, r_max)

            # 收斂檢查
            if iteration > 10:
                if abs(global_best_fit - personal_best_fit.mean()) < 1e-6:
                    logger.debug(f"PSO 在第 {iteration} 次迭代收斂")
                    break

        return {
            "best_radii": global_best_pos,
            "best_fitness": global_best_fit,
            "sds_info": best_sds_info,
            "iterations": iteration + 1,
        }

    def _define_boundaries(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """定義搜尋邊界。

        r_min = 0.1
        r_max = max(|Q_0.05|, |Q_0.95|) + 0.1

        Args:
            data: 情緒空間資料。

        Returns:
            (r_min, r_max) 陣列。
        """
        n_dims = data.shape[1]
        r_min = np.full(n_dims, 0.1)

        r_max = np.zeros(n_dims)
        for i in range(n_dims):
            q05 = np.percentile(data[:, i], 5)
            q95 = np.percentile(data[:, i], 95)
            r_max[i] = max(abs(q05), abs(q95)) + 0.1

        return r_min, r_max

    def _initialize_positions(
        self,
        r_min: np.ndarray,
        r_max: np.ndarray,
    ) -> np.ndarray:
        """隨機初始化粒子位置。

        Args:
            r_min: 最小邊界。
            r_max: 最大邊界。

        Returns:
            粒子位置矩陣 (n_particles x n_dims)。
        """
        n_dims = len(r_min)
        positions = np.zeros((self.n_particles, n_dims))
        for i in range(self.n_particles):
            positions[i] = r_min + np.random.random(n_dims) * (r_max - r_min)
        return positions

    def _initialize_velocities(
        self,
        r_min: np.ndarray,
        r_max: np.ndarray,
    ) -> np.ndarray:
        """初始化粒子速度。

        Args:
            r_min: 最小邊界。
            r_max: 最大邊界。

        Returns:
            速度矩陣。
        """
        n_dims = len(r_min)
        v_max = (r_max - r_min) * 0.5
        return np.random.uniform(-v_max, v_max, (self.n_particles, n_dims))

    def _evaluate_fitness(
        self,
        radii: np.ndarray,
        data: np.ndarray,
        labels: np.ndarray,
        center: np.ndarray,
        alpha: float,
        theta: float,
        n_total: int,
    ) -> Tuple[float, Optional[Dict]]:
        """計算粒子的適應度。

        適應度 = max(N_up/N_sub, N_down/N_sub)（偏斜度）
        需滿足：
        1. 樣本比例 > alpha
        2. 機率偏斜 > theta

        Args:
            radii: 半徑向量。
            data: 樣本矩陣。
            labels: 標籤向量。
            center: 中心向量。
            alpha: 樣本比例門檻。
            theta: 機率門檻。
            n_total: 總樣本數。

        Returns:
            (適應度, SDS 資訊) 元組。
        """
        # 計算每個樣本的標準化距離
        distances = np.sum((data - center) ** 2 / (radii ** 2), axis=1)

        inner_mask = distances <= 1.0
        outer_mask = distances > 1.0

        # 檢查兩個子空間
        best_fitness = -np.inf
        best_info = None

        for mask, space_name in [(inner_mask, "inner"), (outer_mask, "outer")]:
            n_sub = np.sum(mask)
            if n_sub == 0:
                continue

            # Step 1: 樣本比例檢查
            if n_sub / n_total <= alpha:
                continue

            sub_labels = labels[mask]
            n_up = np.sum(sub_labels == 1)
            n_down = np.sum(sub_labels == -1)

            # Step 2: 機率分佈顯著性
            p_up = n_up / n_sub
            p_down = n_down / n_sub

            if max(p_up, p_down) <= theta:
                continue

            # 適應度 = 偏斜度
            skewness = max(n_up / max(n_sub, 1), n_down / max(n_sub, 1))

            if skewness > best_fitness:
                best_fitness = skewness
                best_info = {
                    "space": space_name,
                    "n_sub": int(n_sub),
                    "n_up": int(n_up),
                    "n_down": int(n_down),
                    "p_up": p_up,
                    "p_down": p_down,
                    "skewness": skewness,
                    "radii": radii.copy(),
                }

        if best_fitness == -np.inf:
            return 0.0, None

        return best_fitness, best_info
