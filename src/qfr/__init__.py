"""quant-factor-ranking.

A cross-sectional, machine-learning multi-factor equity ranking model for the
S&P 500. The package is organised into focused sub-modules that mirror the
research workflow:

    data        Point-in-time universe, prices and fundamentals from FMP.
    factors     Value / Quality / Momentum / Growth / Risk factor construction.
    validation  Information-coefficient and factor-significance analysis.
    models      Learning-to-rank models (LightGBM, XGBoost) + baselines.
    portfolio   Decile / long-short portfolio formation.
    backtest    Walk-forward, transaction-cost-aware backtest engine.
    utils       Configuration, logging and IO helpers.
"""

__version__ = "0.1.0"
