"""
Configuration Loader

Purpose:
    Loads and validates YAML configuration files.
    Supports environment variable interpolation.
    Provides cached config access.

Dependencies:
    - PyYAML for YAML parsing
    - python-dotenv for .env loading

Logging:
    - Config load at INFO
    - Missing config at WARNING
    - Parse errors at ERROR

Fallbacks:
    - Uses default values if config missing
    - Falls back to environment variables
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# Load .env file
load_dotenv()

# Config directory
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

# Cache for loaded configs
_config_cache: Dict[str, Dict[str, Any]] = {}


def _interpolate_env_vars(value: Any) -> Any:
    """
    Replace ${VAR_NAME} patterns with environment variables.

    Args:
        value: Config value that may contain env var references.

    Returns:
        Value with env vars interpolated.
    """
    if isinstance(value, str):
        pattern = re.compile(r'\$\{([^}]+)\}')
        matches = pattern.findall(value)
        for var_name in matches:
            env_val = os.environ.get(var_name, "")
            value = value.replace(f"${{{var_name}}}", env_val)
        return value
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def load_config(
    config_name: str,
    config_dir: Optional[Path] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_name: Name of config file (without .yaml extension).
        config_dir: Optional custom config directory.
        use_cache: Whether to use cached config.

    Returns:
        Dictionary with parsed configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file has invalid YAML.
    """
    if use_cache and config_name in _config_cache:
        return _config_cache[config_name]

    search_dir = config_dir or CONFIG_DIR
    config_path = search_dir / f"{config_name}.yaml"

    if not config_path.exists():
        logger.warning(
            f"Config file not found: {config_path}, using defaults",
            extra={"config_name": config_name}
        )
        return {}

    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f) or {}

        config = _interpolate_env_vars(raw_config)

        if use_cache:
            _config_cache[config_name] = config

        logger.info(
            f"Loaded config: {config_name}",
            extra={"config_path": str(config_path)}
        )
        return config

    except yaml.YAMLError as e:
        logger.error(
            f"Failed to parse config: {config_name}",
            exc_info=True,
            extra={"config_path": str(config_path), "error": str(e)}
        )
        raise
    except Exception as e:
        logger.error(
            f"Failed to load config: {config_name}",
            exc_info=True,
            extra={"config_path": str(config_path), "error": str(e)}
        )
        raise


def load_strategy_config(strategy_name: str) -> Dict[str, Any]:
    """
    Load a strategy-specific configuration.

    Args:
        strategy_name: Name of strategy config file.

    Returns:
        Strategy configuration dictionary.
    """
    strategies_dir = CONFIG_DIR / "strategies"
    return load_config(strategy_name, config_dir=strategies_dir)


def reload_config(config_name: str) -> Dict[str, Any]:
    """
    Force reload a configuration file (bypasses cache).

    Args:
        config_name: Name of config file.

    Returns:
        Fresh configuration dictionary.
    """
    if config_name in _config_cache:
        del _config_cache[config_name]

    logger.info(f"Reloading config: {config_name}")
    return load_config(config_name, use_cache=False)


def get_nested(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get a nested config value using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated key path (e.g., 'database.pool_size').
        default: Default value if key not found.

    Returns:
        Config value or default.

    Example:
        >>> config = {'database': {'pool_size': 20}}
        >>> get_nested(config, 'database.pool_size')
        20
    """
    keys = key_path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def clear_cache() -> None:
    """Clear all cached configurations."""
    _config_cache.clear()
    logger.info("Config cache cleared")
