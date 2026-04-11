"""建構三個情緒指標腳本。

載入原始資料，建構 AI、BI、PI 並對齊到交易日。
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import get_config, get_data_dir
from src.utils.logging_utils import setup_logger

logger = setup_logger("build_indices")


def main() -> None:
    """建構所有指標。"""
    config = get_config()

    print("=" * 60)
    print("  建構情緒指標")
    print("=" * 60)

    # 1. 建構關注度指數（AI）
    print("\n[1/3] 建構關注度指數（Attention Index）...")
    try:
        from src.processors.attention_builder import AttentionBuilder

        ai_builder = AttentionBuilder()
        ai_df = ai_builder.build()
        print(f"  完成：{len(ai_df)} 筆")
    except Exception as e:
        print(f"  失敗：{e}")
        ai_df = None

    # 2. 建構看漲指數（BI）
    print("\n[2/3] 建構看漲指數（Bullish Index）...")
    try:
        from src.processors.bullish_builder import BullishBuilder

        bi_builder = BullishBuilder()
        bi_df = bi_builder.build()
        print(f"  完成：{len(bi_df)} 筆")
    except Exception as e:
        print(f"  失敗：{e}")
        bi_df = None

    # 3. 建構傳播指數（PI）
    print("\n[3/3] 建構傳播指數（Propagation Index）...")
    try:
        from src.processors.propagation_builder import PropagationBuilder

        pi_builder = PropagationBuilder()
        pi_df = pi_builder.build()
        print(f"  完成：{len(pi_df)} 筆")
    except Exception as e:
        print(f"  失敗：{e}")
        pi_df = None

    # 統計
    print("\n" + "=" * 60)
    print("  建構結果統計")
    print("=" * 60)

    for name, df in [("AI", ai_df), ("BI", bi_df), ("PI", pi_df)]:
        if df is not None and len(df) > 0:
            print(f"  {name}: {len(df)} 筆")
            if f"{name.lower()}_zscore" in df.columns:
                col = f"{name.lower()}_zscore"
            elif "ai_zscore" in df.columns:
                col = "ai_zscore"
            elif "bi_zscore" in df.columns:
                col = "bi_zscore"
            elif "pi_zscore" in df.columns:
                col = "pi_zscore"
            else:
                continue
            print(f"    均值: {df[col].mean():.4f}, 標準差: {df[col].std():.4f}")
        else:
            print(f"  {name}: 無資料")

    print("\n指標已儲存至 data/processed/")


if __name__ == "__main__":
    main()
