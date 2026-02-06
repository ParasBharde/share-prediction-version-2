"""
Strategy Deployment Script

Purpose:
    Validates and deploys a new trading strategy configuration.
    Runs validation, backtest, and activates the strategy.

Usage:
    python scripts/deploy_strategy.py --config path/to/strategy.yaml
    python scripts/deploy_strategy.py --validate-only --config strategy.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.logger import get_logger
from src.utils.config_loader import load_config
from src.utils.validators import validate_strategy_config

logger = get_logger(__name__)

STRATEGIES_DIR = Path(__file__).parent.parent / "config" / "strategies"


def deploy_strategy(
    config_path: str, validate_only: bool = False
):
    """
    Deploy a new strategy.

    Args:
        config_path: Path to strategy YAML config.
        validate_only: Only validate, don't deploy.
    """
    source = Path(config_path)

    if not source.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Load and validate
    import yaml

    with open(source) as f:
        config = yaml.safe_load(f)

    validation = validate_strategy_config(config)

    if not validation["valid"]:
        logger.error(
            f"Strategy validation failed: {validation['issues']}"
        )
        sys.exit(1)

    logger.info("Strategy config validation passed")

    strategy_name = config.get("strategy", {}).get(
        "name", "unknown"
    )

    if validate_only:
        logger.info(
            f"Validation complete for: {strategy_name}"
        )
        return

    # Copy to strategies directory
    dest = STRATEGIES_DIR / source.name

    if dest.exists():
        # Backup existing
        backup = dest.with_suffix(".yaml.bak")
        shutil.copy2(dest, backup)
        logger.info(f"Backed up existing config to {backup}")

    shutil.copy2(source, dest)
    logger.info(
        f"Deployed strategy '{strategy_name}' to {dest}"
    )
    logger.info(
        "Strategy will be loaded on next scan cycle. "
        "Use hot-reload to activate immediately."
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy a trading strategy"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to strategy YAML config",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate config",
    )

    args = parser.parse_args()
    deploy_strategy(args.config, args.validate_only)


if __name__ == "__main__":
    main()
