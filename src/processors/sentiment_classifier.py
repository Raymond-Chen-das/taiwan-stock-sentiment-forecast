"""情緒分類模組（進階版）。

使用基於規則或預訓練模型對中文文本進行情緒分類。
支援正面、中性、負面三分類。
"""

import re
from typing import List, Optional, Tuple

import pandas as pd

from src.utils.logging_utils import setup_logger

logger = setup_logger("sentiment_classifier")

# 基礎情緒詞典（台股常用語）
_POSITIVE_WORDS = [
    "看多", "做多", "多方", "上漲", "大漲", "噴出", "起飛", "利多",
    "突破", "創高", "買進", "加碼", "進場", "紅盤", "強勢", "飆漲",
    "反彈", "回升", "樂觀", "看好", "抄底", "翻多", "底部",
]

_NEGATIVE_WORDS = [
    "看空", "做空", "空方", "下跌", "大跌", "崩盤", "利空",
    "跌破", "創低", "賣出", "減碼", "出場", "綠盤", "弱勢", "暴跌",
    "回檔", "下挫", "悲觀", "看壞", "停損", "翻空", "套牢", "割肉",
]


class SentimentClassifier:
    """中文情緒分類器。

    提供基於詞典的情緒分類功能。

    Attributes:
        positive_words: 正面情緒詞列表。
        negative_words: 負面情緒詞列表。
    """

    def __init__(self) -> None:
        self.positive_words = _POSITIVE_WORDS
        self.negative_words = _NEGATIVE_WORDS

    def classify(self, text: str) -> Tuple[str, float]:
        """對單一文本進行情緒分類。

        Args:
            text: 輸入文本。

        Returns:
            (標籤, 分數) 元組。標籤為 'positive', 'neutral', 'negative'。
        """
        if not text or not isinstance(text, str):
            return "neutral", 0.0

        pos_count = sum(1 for w in self.positive_words if w in text)
        neg_count = sum(1 for w in self.negative_words if w in text)

        total = pos_count + neg_count
        if total == 0:
            return "neutral", 0.0

        score = (pos_count - neg_count) / total

        if score > 0.1:
            return "positive", score
        elif score < -0.1:
            return "negative", score
        else:
            return "neutral", score

    def classify_batch(self, texts: List[str]) -> pd.DataFrame:
        """批次分類多個文本。

        Args:
            texts: 文本列表。

        Returns:
            包含 text, label, score 的 DataFrame。
        """
        results = [self.classify(t) for t in texts]
        return pd.DataFrame({
            "text": texts,
            "label": [r[0] for r in results],
            "score": [r[1] for r in results],
        })

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "title",
    ) -> pd.DataFrame:
        """對 DataFrame 的指定欄位進行情緒分類。

        Args:
            df: 輸入 DataFrame。
            text_column: 文本欄位名稱。

        Returns:
            加入 sentiment_label 和 sentiment_score 的 DataFrame。
        """
        df = df.copy()
        results = [self.classify(str(t)) for t in df[text_column]]
        df["sentiment_label"] = [r[0] for r in results]
        df["sentiment_score"] = [r[1] for r in results]

        # 統計
        counts = df["sentiment_label"].value_counts()
        logger.info(f"情緒分類結果：{counts.to_dict()}")

        return df
