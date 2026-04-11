"""收集器基本測試。"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def test_config_loader():
    """測試設定檔載入。"""
    from src.utils.config_loader import get_config, get_project_root

    config = get_config()
    assert "project" in config
    assert config["project"]["name"] == "taiwan-sentiment-forecast"
    assert config["project"]["train_start_date"] == "2021-01-01"
    assert config["project"]["test_end_date"] == "2026-01-31"


def test_calendar():
    """測試交易日曆。"""
    import datetime
    from src.utils.calendar_utils import TaiwanTradingCalendar

    cal = TaiwanTradingCalendar()

    # 2024-01-01 元旦不交易
    assert not cal.is_trading_day(datetime.date(2024, 1, 1))

    # 2024-01-02 週二交易
    assert cal.is_trading_day(datetime.date(2024, 1, 2))

    # 週六不交易
    assert not cal.is_trading_day(datetime.date(2024, 1, 6))

    # 取得交易日
    days = cal.get_trading_days(
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 7),
    )
    assert len(days) > 0
    assert all(cal.is_trading_day(d) for d in days)


def test_sentiment_space():
    """測試情緒空間模型。"""
    import numpy as np
    from src.models.sentiment_space import SentimentSpace

    space = SentimentSpace(window_size=10)

    # 模擬資料
    np.random.seed(42)
    data = np.random.randn(20, 3)
    labels = np.random.choice([1, -1], size=20)

    # 建構空間
    info = space.build_space(data, labels, 15)
    assert "center" in info
    assert len(info["center"]) == 3


def test_pso_search():
    """測試 PSO 搜尋。"""
    import numpy as np
    from src.models.pso_search import PSOSearcher

    pso = PSOSearcher(n_particles=10, max_iterations=20)

    np.random.seed(42)
    data = np.random.randn(30, 2)
    labels = np.where(data[:, 0] > 0, 1, -1)
    center = np.mean(data, axis=0)

    result = pso.search(data, labels, center, alpha=0.15, theta=0.55)
    assert "best_radii" in result
    assert "best_fitness" in result


if __name__ == "__main__":
    test_config_loader()
    print("[OK] test_config_loader")

    test_calendar()
    print("[OK] test_calendar")

    test_sentiment_space()
    print("[OK] test_sentiment_space")

    test_pso_search()
    print("[OK] test_pso_search")

    print("\n所有測試通過！")
