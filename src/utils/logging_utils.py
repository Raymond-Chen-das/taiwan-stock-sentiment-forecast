"""日誌工具模組。

提供統一的日誌設定，同時輸出到 console 和檔案。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """設定並回傳 logger。

    同時輸出到 console（INFO 級別）和檔案（DEBUG 級別）。

    Args:
        name: Logger 名稱。
        log_file: 日誌檔案路徑，若為 None 則自動使用 logs/{name}.log。
        level: 日誌級別。

    Returns:
        設定完成的 Logger 物件。
    """
    logger = logging.getLogger(name)

    # 避免重複設定 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG)
    if log_file is None:
        from src.utils.config_loader import get_project_root
        try:
            log_dir = get_project_root() / "logs"
        except FileNotFoundError:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / f"{name}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
