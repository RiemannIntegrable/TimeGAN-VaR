"""
Utilidades para evaluación y visualización de modelos y resultados.
"""

from .evaluation import (
    calculate_kl_divergence,
    discriminative_score,
    predictive_score,
    visualize_tsne
)

from .visualization import (
    plot_stock_prices,
    plot_returns_distribution,
    plot_acf_comparison,
    plot_var_histogram,
    plot_multiple_var_methods,
    plot_synthetic_returns
)

__all__ = [
    'calculate_kl_divergence',
    'discriminative_score',
    'predictive_score',
    'visualize_tsne',
    'plot_stock_prices',
    'plot_returns_distribution',
    'plot_acf_comparison',
    'plot_var_histogram',
    'plot_multiple_var_methods',
    'plot_synthetic_returns'
]