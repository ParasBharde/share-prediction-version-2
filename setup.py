"""Package installation configuration."""

from setuptools import find_packages, setup

setup(
    name="algotrade-scanner",
    version="1.0.0",
    description="NSE Stock Scanning & Alert System",
    author="AlgoTrade",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "aiohttp>=3.9.0",
        "APScheduler>=3.10.0",
        "SQLAlchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "redis>=5.0.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "yfinance>=0.2.30",
        "python-telegram-bot>=20.0",
        "PyYAML>=6.0",
        "python-dotenv>=1.0.0",
        "prometheus-client>=0.19.0",
        "Jinja2>=3.1.0",
        "pytz>=2023.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "algotrade-scan=scripts.daily_scan:main",
        ],
    },
)
