"""Configuration loader for the Multimodal Evidence Fusion QA System."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file if it exists
load_dotenv(PROJECT_ROOT / ".env")


def load_yaml_config(config_path: Path = None) -> Dict[str, Any]:
    """Load configuration from config.yaml with fallback to defaults."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# Load static config
_RAW_CONFIG = load_yaml_config()


class AppConfig:
    """Convenience wrapper for application configuration and environment variables."""

    # Project paths
    ROOT_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / _RAW_CONFIG.get("storage", {}).get("base_dir", "data")
    UPLOADS_DIR: Path = PROJECT_ROOT / _RAW_CONFIG.get("storage", {}).get("uploads_dir", "data/uploads")
    EXTRACTED_DIR: Path = PROJECT_ROOT / _RAW_CONFIG.get("storage", {}).get("extracted_dir", "data/extracted")
    FIGURES_DIR: Path = PROJECT_ROOT / _RAW_CONFIG.get("storage", {}).get("extracted_dir", "data/extracted") / "figures"
    INDICES_DIR: Path = PROJECT_ROOT / _RAW_CONFIG.get("storage", {}).get("indices_dir", "data/indices")

    # App meta
    APP_NAME: str = _RAW_CONFIG.get("app", {}).get("name", "Query-Aware Multimodal Evidence Fusion")
    APP_TAGLINE: str = _RAW_CONFIG.get("app", {}).get("tagline", "Grounded Document QA with Citations")
    APP_VERSION: str = _RAW_CONFIG.get("app", {}).get("version", "0.5.0-phase5")
    DEBUG: bool = _RAW_CONFIG.get("app", {}).get("debug", False)

    # Retrieval & Indexing
    TEXT_EMBEDDING_MODEL: str = os.getenv("TEXT_EMBEDDING_MODEL", _RAW_CONFIG.get("indexing", {}).get("embedding_model", "all-MiniLM-L6-v2"))
    IMAGE_EMBEDDING_MODEL: str = os.getenv("IMAGE_EMBEDDING_MODEL", _RAW_CONFIG.get("indexing", {}).get("image_model", "openai/clip-vit-base-patch32"))
    DEVICE: str = os.getenv("DEVICE", _RAW_CONFIG.get("indexing", {}).get("device", "cpu"))

    # API Keys & Generation
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_PROVIDER: str = _RAW_CONFIG.get("generation", {}).get("provider", "gemini")
    LLM_MODEL: str = _RAW_CONFIG.get("generation", {}).get("model_name", "gemini-1.5-flash")

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required runtime data directories exist."""
        for directory in [cls.DATA_DIR, cls.UPLOADS_DIR, cls.EXTRACTED_DIR, cls.FIGURES_DIR, cls.INDICES_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# Initialize runtime directories
AppConfig.ensure_directories()
