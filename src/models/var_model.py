import numpy as np
import pandas as pd
from scipy import stats

def calculate_daily_returns(prices):
    """
    Calcula los rendimientos diarios a partir de una serie de precios.
    
    Args:
        prices: Array de numpy o Serie de pandas con precios
        
    Returns:
        Array con rendimientos diarios logarítmicos
    """
    if isinstance(prices, pd.Series) or isinstance(prices, pd.DataFrame):
        prices = prices.values.flatten()
    
    log_returns = np.diff(np.log(prices))
    return log_returns

def monte_carlo_var(synthetic_prices, initial_price, horizon=1, confidence_level=0.95, n_simulations=10000):
    """
    Calcula el VaR mediante simulación de Monte Carlo usando datos sintéticos.
    
    Args:
        synthetic_prices: Lista de arrays de precios sintéticos generados por TimeGAN
        initial_price: Precio inicial para las simulaciones
        horizon: Horizonte temporal para el cálculo del VaR (en días)
        confidence_level: Nivel de confianza para el VaR (por defecto 0.95 para VaR al 95%)
        n_simulations: Número de simulaciones de Monte Carlo
        
    Returns:
        Valor en Riesgo (VaR) y Expected Shortfall (ES)
    """
    # Convertir múltiples trayectorias sintéticas en una única colección de retornos diarios
    all_returns = []
    
    for price_path in synthetic_prices:
        returns = calculate_daily_returns(price_path)
        all_returns.extend(returns)
    
    all_returns = np.array(all_returns)
    
    # Realizar simulaciones de Monte Carlo
    simulated_returns = np.random.choice(all_returns, size=(n_simulations, horizon))
    simulated_cumulative_returns = np.sum(simulated_returns, axis=1)
    
    # Calcular precios finales simulados
    simulated_prices = initial_price * np.exp(simulated_cumulative_returns)
    
    # Calcular pérdidas y ganancias (P&L)
    pnl = simulated_prices - initial_price
    
    # Calcular VaR como percentil de las pérdidas
    var = -np.percentile(pnl, 100 * (1 - confidence_level))
    
    # Calcular Expected Shortfall (Conditional VaR)
    es = -np.mean(pnl[pnl <= -var])
    
    return var, es

def parametric_var(returns, initial_investment, horizon=1, confidence_level=0.95, distribution='normal'):
    """
    Calcula el VaR paramétrico asumiendo una distribución específica.
    
    Args:
        returns: Array de rendimientos históricos o sintéticos
        initial_investment: Inversión inicial
        horizon: Horizonte temporal (en días)
        confidence_level: Nivel de confianza para el VaR
        distribution: 'normal' o 't-student'
        
    Returns:
        VaR paramétrico
    """
    # Calcular media y desviación estándar de los rendimientos
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    # Factor de escala para el horizonte temporal (asumiendo independencia)
    scale_factor = np.sqrt(horizon)
    
    if distribution == 'normal':
        # Calcular el factor Z para la distribución normal
        z_score = stats.norm.ppf(1 - confidence_level)
        var = initial_investment * (-(mu * horizon + z_score * sigma * scale_factor))
    
    elif distribution == 't-student':
        # Estimar los grados de libertad para la distribución t
        params = stats.t.fit(returns)
        df = params[0]  # grados de libertad
        t_score = stats.t.ppf(1 - confidence_level, df)
        var = initial_investment * (-(mu * horizon + t_score * sigma * scale_factor))
    
    return var

def historical_var(returns, initial_investment, confidence_level=0.95):
    """
    Calcula el VaR histórico basado en la distribución empírica de rendimientos.
    
    Args:
        returns: Array de rendimientos históricos o sintéticos
        initial_investment: Inversión inicial
        confidence_level: Nivel de confianza para el VaR
        
    Returns:
        VaR histórico
    """
    # Ordenar los rendimientos de forma ascendente
    sorted_returns = np.sort(returns)
    
    # Encontrar el índice correspondiente al percentil
    index = int(np.floor((1 - confidence_level) * len(sorted_returns)))
    
    # Calcular VaR
    var = -initial_investment * sorted_returns[index]
    
    return var