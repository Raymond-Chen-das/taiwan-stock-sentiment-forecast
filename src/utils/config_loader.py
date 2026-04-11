"""設定檔載入模組。

提供統一的介面來載入 settings.yaml 和 credentials.yaml，
使用 pathlib.Path 確保 Windows 相容性。
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _find_project_root(start_path: Optional[Path] = None) -> Path:
    """從指定路徑往上搜尋，直到找到包含 config/ 資料夾的目錄。

    Args:
        start_path: 搜尋起點，預設為當前檔案所在目錄。

    Returns:
        專案根目錄的 Path 物件。

    Raises:
        FileNotFoundError: 找不到包含 config/ 的目錄。
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current = start_path
    for _ in range(10):  # 最多往上搜尋 10 層
        if (current / "config").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        f"無法從 {start_path} 找到包含 config/ 資料夾的專案根目錄"
    )


def get_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """載入 settings.yaml 設定檔。

    Args:
        config_path: 設定檔路徑，預設自動偵測。

    Returns:
        設定檔內容的字典。
    """
    if config_path is None:
        root = _find_project_root()
        config_path = root / "config" / "settings.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_credentials(cred_path: Optional[Path] = None) -> Dict[str, Any]:
    """載入 credentials.yaml 認證檔。

    Args:
        cred_path: 認證檔路徑，預設自動偵測。

    Returns:
        認證資訊的字典。
    """
    if cred_path is None:
        root = _find_project_root()
        cred_path = root / "config" / "credentials.yaml"

    if not cred_path.exists():
        raise FileNotFoundError(
            f"認證檔不存在：{cred_path}\n"
            "請複製 credentials.yaml.example 並填入 API 金鑰。"
        )

    with open(cred_path, "r", encoding="utf-8") as f:
        creds = yaml.safe_load(f)

    return creds


def get_project_root() -> Path:
    """取得專案根目錄路徑。

    Returns:
        專案根目錄的 Path 物件。
    """
    return _find_project_root()


def get_data_dir(sub: str = "") -> Path:
    """取得資料目錄路徑。

    Args:
        sub: 子目錄名稱（如 'raw/taiex'）。

    Returns:
        資料目錄的 Path 物件。
    """
    data_dir = _find_project_root() / "data"
    if sub:
        data_dir = data_dir / sub
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
