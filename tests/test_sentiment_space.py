"""情緒空間模型（SentimentSpace）核心幾何邏輯的單元測試。

涵蓋論文方法的數學核心：橢球標準化距離、內外空間分割、
Gini 不純度、子空間機率分佈、幾何中心與滑動視窗。
"""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.sentiment_space import SentimentSpace


def test_ellipsoid_distance_formula():
    """D = sum((s_i - c_i)^2 / r_i^2)。"""
    space = SentimentSpace()
    # 落在橢球邊界上：(2/2)^2 + (0/1)^2 = 1.0
    d = space.compute_ellipsoid_distance(
        np.array([2.0, 0.0]), np.array([0.0, 0.0]), np.array([2.0, 1.0])
    )
    assert abs(d - 1.0) < 1e-12

    # (1/1)^2 + (1/1)^2 = 2.0
    d2 = space.compute_ellipsoid_distance(
        np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0])
    )
    assert abs(d2 - 2.0) < 1e-12


def test_partition_inner_outer():
    """距離 <= 1 為內部，> 1 為外部。"""
    space = SentimentSpace()
    data = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    center = np.array([0.0, 0.0])
    radii = np.array([1.0, 1.0])

    inner, outer = space.partition_space(data, center, radii)
    np.testing.assert_array_equal(inner, [True, False, False])
    np.testing.assert_array_equal(outer, [False, True, True])
    # 內外互補
    np.testing.assert_array_equal(inner, ~outer)


def test_gini_index():
    """純淨 -> 0；五五波 -> 0.5；空集合 -> 0.5（依實作約定）。"""
    space = SentimentSpace()
    assert abs(space.compute_gini_index(np.array([1, 1, 1, 1]))) < 1e-12
    assert abs(space.compute_gini_index(np.array([1, -1])) - 0.5) < 1e-12
    assert abs(space.compute_gini_index(np.array([])) - 0.5) < 1e-12


def test_subspace_probability():
    space = SentimentSpace()
    probs = space.compute_subspace_probability(np.array([1, 1, -1, 0]))
    assert abs(probs["p_up"] - 0.5) < 1e-12
    assert abs(probs["p_down"] - 0.25) < 1e-12
    assert abs(probs["p_stable"] - 0.25) < 1e-12

    empty = space.compute_subspace_probability(np.array([]))
    assert empty == {"p_up": 0.0, "p_down": 0.0, "p_stable": 0.0}


def test_geometric_center_is_mean():
    space = SentimentSpace()
    data = np.array([[0.0, 0.0], [2.0, 2.0]])
    center = space._compute_center(data)
    np.testing.assert_allclose(center, [1.0, 1.0])


def test_build_space_requires_full_window():
    """t < window_size 時回傳空字典；t >= window_size 時取得長度 w 的視窗。"""
    space = SentimentSpace(window_size=3)
    data = np.array([[float(i), float(i)] for i in range(6)])
    labels = np.array([1, -1, 0, 1, -1, 1])

    assert space.build_space(data, labels, t=2) == {}

    info = space.build_space(data, labels, t=5)
    assert info["window_data"].shape == (3, 2)
    np.testing.assert_array_equal(info["window_data"], data[2:5])
    np.testing.assert_array_equal(info["window_labels"], labels[2:5])
