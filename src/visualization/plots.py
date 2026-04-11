"""視覺化模組。

使用 Plotly 繪製各類圖表（不使用 matplotlib）。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.logging_utils import setup_logger

logger = setup_logger("plots")


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "相關係數矩陣",
    output_path: Optional[str] = None,
) -> go.Figure:
    """繪製相關矩陣熱力圖。

    Args:
        corr_matrix: 相關矩陣 DataFrame。
        title: 圖表標題。
        output_path: 輸出檔案路徑（可選）。

    Returns:
        Plotly Figure 物件。
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 12},
    ))

    fig.update_layout(
        title=title,
        width=700,
        height=600,
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def plot_time_series_comparison(
    df: pd.DataFrame,
    date_col: str = "date",
    columns: Optional[List[str]] = None,
    title: str = "時間序列對比",
    output_path: Optional[str] = None,
) -> go.Figure:
    """繪製多條時間序列對比圖。

    Args:
        df: 資料 DataFrame。
        date_col: 日期欄位名稱。
        columns: 要繪製的欄位列表。
        title: 圖表標題。
        output_path: 輸出路徑。

    Returns:
        Plotly Figure 物件。
    """
    if columns is None:
        columns = [c for c in df.columns if c != date_col]

    n_cols = len(columns)
    fig = make_subplots(
        rows=n_cols,
        cols=1,
        shared_xaxes=True,
        subplot_titles=columns,
        vertical_spacing=0.05,
    )

    colors = px.colors.qualitative.Set2

    for i, col in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[col],
                name=col,
                line=dict(color=colors[i % len(colors)]),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=300 * n_cols,
        showlegend=True,
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def plot_sentiment_space_2d(
    data: np.ndarray,
    labels: np.ndarray,
    center: np.ndarray,
    radii: Optional[np.ndarray] = None,
    sample: Optional[np.ndarray] = None,
    axis_names: List[str] = ["指標 1", "指標 2"],
    title: str = "情緒空間（2D）",
    output_path: Optional[str] = None,
) -> go.Figure:
    """繪製 2D 情緒空間圖。

    Args:
        data: 樣本資料 (n x 2)。
        labels: 趨勢標籤。
        center: 幾何中心。
        radii: HDS 半徑（可選）。
        sample: 待評估樣本（可選）。
        axis_names: 軸名稱。
        title: 圖表標題。
        output_path: 輸出路徑。

    Returns:
        Plotly Figure 物件。
    """
    fig = go.Figure()

    # 上漲樣本（橘色）
    up_mask = labels == 1
    fig.add_trace(go.Scatter(
        x=data[up_mask, 0],
        y=data[up_mask, 1],
        mode="markers",
        name="上漲",
        marker=dict(color="orange", size=8),
    ))

    # 下跌樣本（藍色）
    down_mask = labels == -1
    fig.add_trace(go.Scatter(
        x=data[down_mask, 0],
        y=data[down_mask, 1],
        mode="markers",
        name="下跌",
        marker=dict(color="steelblue", size=8),
    ))

    # 幾何中心
    fig.add_trace(go.Scatter(
        x=[center[0]],
        y=[center[1]],
        mode="markers",
        name="幾何中心",
        marker=dict(
            color="black",
            size=12,
            symbol="triangle-up",
        ),
    ))

    # 橢圓（HDS 邊界）
    if radii is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = center[0] + radii[0] * np.cos(theta)
        ellipse_y = center[1] + radii[1] * np.sin(theta)

        fig.add_trace(go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode="lines",
            name="SDS 邊界",
            line=dict(color="gold", width=2),
        ))

    # 待評估樣本
    if sample is not None:
        fig.add_trace(go.Scatter(
            x=[sample[0]],
            y=[sample[1]],
            mode="markers",
            name="評估樣本",
            marker=dict(
                color="red",
                size=15,
                symbol="circle-open",
                line=dict(width=3),
            ),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=axis_names[0],
        yaxis_title=axis_names[1],
        width=700,
        height=600,
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def plot_sentiment_space_3d(
    data: np.ndarray,
    labels: np.ndarray,
    center: np.ndarray,
    radii: Optional[np.ndarray] = None,
    sample: Optional[np.ndarray] = None,
    axis_names: List[str] = ["AI", "BI", "PI"],
    title: str = "情緒空間（3D）",
    output_path: Optional[str] = None,
) -> go.Figure:
    """繪製 3D 情緒空間圖。

    Args:
        data: 樣本資料 (n x 3)。
        labels: 趨勢標籤。
        center: 幾何中心。
        radii: HDS 半徑（可選）。
        sample: 待評估樣本（可選）。
        axis_names: 軸名稱。
        title: 圖表標題。
        output_path: 輸出路徑。

    Returns:
        Plotly Figure 物件。
    """
    fig = go.Figure()

    # 上漲樣本
    up_mask = labels == 1
    fig.add_trace(go.Scatter3d(
        x=data[up_mask, 0],
        y=data[up_mask, 1],
        z=data[up_mask, 2],
        mode="markers",
        name="上漲",
        marker=dict(color="orange", size=4),
    ))

    # 下跌樣本
    down_mask = labels == -1
    fig.add_trace(go.Scatter3d(
        x=data[down_mask, 0],
        y=data[down_mask, 1],
        z=data[down_mask, 2],
        mode="markers",
        name="下跌",
        marker=dict(color="steelblue", size=4),
    ))

    # 幾何中心
    fig.add_trace(go.Scatter3d(
        x=[center[0]],
        y=[center[1]],
        z=[center[2]],
        mode="markers",
        name="幾何中心",
        marker=dict(color="black", size=8, symbol="diamond"),
    ))

    # 待評估樣本
    if sample is not None:
        fig.add_trace(go.Scatter3d(
            x=[sample[0]],
            y=[sample[1]],
            z=[sample[2]],
            mode="markers",
            name="評估樣本",
            marker=dict(
                color="red",
                size=10,
                symbol="circle-open",
            ),
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=axis_names[0],
            yaxis_title=axis_names[1],
            zaxis_title=axis_names[2],
        ),
        width=800,
        height=700,
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def plot_prediction_results(
    predictions: pd.DataFrame,
    title: str = "預測結果",
    output_path: Optional[str] = None,
) -> go.Figure:
    """繪製預測結果時間序列圖。

    Args:
        predictions: 預測結果 DataFrame。
        title: 圖表標題。
        output_path: 輸出路徑。

    Returns:
        Plotly Figure 物件。
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["實際 vs 預測", "Support 與 Gini 指數"],
        vertical_spacing=0.1,
    )

    # 實際趨勢
    fig.add_trace(
        go.Scatter(
            x=predictions["t"],
            y=predictions["actual"],
            name="實際",
            line=dict(color="steelblue"),
        ),
        row=1,
        col=1,
    )

    # 預測趨勢
    fig.add_trace(
        go.Scatter(
            x=predictions["t"],
            y=predictions["forecast"],
            name="預測",
            line=dict(color="orange", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Support
    if "support" in predictions.columns:
        fig.add_trace(
            go.Bar(
                x=predictions["t"],
                y=predictions["support"],
                name="Support",
                marker_color="lightgreen",
            ),
            row=2,
            col=1,
        )

    # Gini
    if "mean_gini" in predictions.columns:
        fig.add_trace(
            go.Scatter(
                x=predictions["t"],
                y=predictions["mean_gini"],
                name="Mean Gini",
                line=dict(color="red"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=600,
    )

    if output_path:
        fig.write_html(output_path)

    return fig
