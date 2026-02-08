"""
Dynamic Strategy Loader

Purpose:
    Dynamically loads and manages trading strategies.
    Scans config/strategies/*.yaml for strategy definitions.
    Supports hot-reloading on config changes.

Dependencies:
    - Strategy implementations
    - YAML config files

Logging:
    - Strategy load at INFO
    - Load errors at ERROR

Fallbacks:
    Invalid strategies are skipped with warnings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Type

from src.monitoring.logger import get_logger
from src.strategies.base_strategy import BaseStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_breakout import (
    MomentumBreakoutStrategy,
)
from src.strategies.mother_candle import MotherCandleStrategy
from src.strategies.volume_surge import VolumeSurgeStrategy
from src.utils.config_loader import load_strategy_config
from src.utils.validators import validate_strategy_config

logger = get_logger(__name__)

# Registry of strategy name -> class
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "Momentum Breakout": MomentumBreakoutStrategy,
    "Mean Reversion": MeanReversionStrategy,
    "Volume Surge": VolumeSurgeStrategy,
    "Mother Candle": MotherCandleStrategy,
}

# Config directory
STRATEGIES_DIR = (
    Path(__file__).parent.parent.parent / "config" / "strategies"
)


class StrategyLoader:
    """Dynamically loads and manages strategies."""

    def __init__(self):
        """Initialize strategy loader."""
        self.strategies: Dict[str, BaseStrategy] = {}
        self._loaded = False

    def load_all(self) -> List[BaseStrategy]:
        """
        Load all enabled strategies from config files.

        Returns:
            List of initialized strategy instances.
        """
        self.strategies.clear()

        if not STRATEGIES_DIR.exists():
            logger.warning(
                f"Strategies directory not found: "
                f"{STRATEGIES_DIR}"
            )
            return []

        strategy_files = sorted(STRATEGIES_DIR.glob("*.yaml"))

        for filepath in strategy_files:
            # Skip template files
            if "template" in filepath.stem:
                continue

            try:
                self._load_strategy(filepath)
            except Exception as e:
                logger.error(
                    f"Failed to load strategy from "
                    f"{filepath.name}: {e}",
                    exc_info=True,
                )

        self._loaded = True
        loaded = list(self.strategies.values())

        # Sort by priority (lower number = higher priority)
        loaded.sort(
            key=lambda s: s.priority, reverse=True
        )

        logger.info(
            f"Loaded {len(loaded)} strategies: "
            f"{[s.name for s in loaded]}"
        )

        return loaded

    def _load_strategy(self, filepath: Path) -> None:
        """
        Load a single strategy from YAML config.

        Args:
            filepath: Path to strategy YAML file.
        """
        config = load_strategy_config(filepath.stem)

        if not config:
            logger.warning(
                f"Empty config for strategy: {filepath.name}"
            )
            return

        # Validate config
        validation = validate_strategy_config(config)
        if not validation["valid"]:
            logger.warning(
                f"Invalid strategy config {filepath.name}: "
                f"{validation['issues']}"
            )
            return

        strategy_name = (
            config.get("strategy", {}).get("name", "")
        )
        enabled = config.get("strategy", {}).get("enabled", False)

        if not enabled:
            logger.info(
                f"Strategy disabled: {strategy_name}"
            )
            return

        # Find matching implementation class
        strategy_class = STRATEGY_REGISTRY.get(strategy_name)

        if strategy_class is None:
            logger.warning(
                f"No implementation found for strategy: "
                f"{strategy_name}"
            )
            return

        # Instantiate strategy
        strategy = strategy_class(config)
        self.strategies[strategy_name] = strategy

        logger.info(
            f"Loaded strategy: {strategy_name} "
            f"(v{strategy.version})",
            extra={
                "strategy": strategy_name,
                "priority": strategy.priority,
            },
        )

    def get_strategy(
        self, name: str
    ) -> Optional[BaseStrategy]:
        """
        Get a loaded strategy by name.

        Args:
            name: Strategy name.

        Returns:
            Strategy instance or None.
        """
        return self.strategies.get(name)

    def get_enabled_strategies(self) -> List[BaseStrategy]:
        """
        Get all enabled strategies, sorted by priority.

        Returns:
            List of enabled strategy instances.
        """
        if not self._loaded:
            self.load_all()

        return sorted(
            self.strategies.values(),
            key=lambda s: s.priority,
            reverse=True,
        )

    def reload(self) -> List[BaseStrategy]:
        """
        Reload all strategies (hot-reload).

        Returns:
            List of reloaded strategy instances.
        """
        logger.info("Reloading all strategies...")
        return self.load_all()

    def register_strategy(
        self,
        name: str,
        strategy_class: Type[BaseStrategy],
    ) -> None:
        """
        Register a custom strategy class.

        Args:
            name: Strategy name.
            strategy_class: Strategy implementation class.
        """
        STRATEGY_REGISTRY[name] = strategy_class
        logger.info(
            f"Registered custom strategy: {name}"
        )
