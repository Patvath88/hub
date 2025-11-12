"""
Simulation and machine‑learning logic for the NBA player prop simulator.

This module defines helper functions to load sample data, train simple linear regression
models for each box‑score stat, compute recent form and head‑to‑head adjustments,
and run a Monte‑Carlo simulation to produce a projected stat line.  It
serves as the core of the backend API in ``app.py``.

In a production system you would replace the ``load_game_logs`` function
with a call to an external API (NBA Stats API, BallDontLie, etc.), and you
would likely implement more sophisticated feature engineering and models.

The simulator returns the median of simulated outcomes for each stat and
includes the machine learning prediction as a baseline estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.linear_model import LinearRegression


STATS = [
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
]


def load_game_logs() -> pd.DataFrame:
    """Load sample game logs from the CSV in the data directory.

    Returns:
        pandas.DataFrame: DataFrame containing historical game logs.
    """
    data_path = Path(__file__).resolve().parent / "data" / "sample_games.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])
    return df


def compute_recent_form(df: pd.DataFrame, player: str, n: int = 5) -> Dict[str, Tuple[float, float]]:
    """Compute the mean and standard deviation of the last ``n`` games for each stat.

    Args:
        df: DataFrame containing game logs.
        player: Name of the player.
        n: Number of recent games to consider.

    Returns:
        A dictionary mapping each stat to a tuple (mean, std).  If the player
        has fewer than ``n`` games, uses all available games.
    """
    player_games = df[df["player"] == player].sort_values("date", ascending=False).head(n)
    means_stds: Dict[str, Tuple[float, float]] = {}
    for stat in STATS:
        series = player_games[stat]
        means_stds[stat] = (series.mean(), series.std(ddof=0) if len(series) > 1 else 0.1)
    return means_stds


def compute_head_to_head(df: pd.DataFrame, player: str, opponent: str) -> Dict[str, float]:
    """Compute the mean of each stat for the player against a specific opponent.

    Args:
        df: DataFrame containing game logs.
        player: Name of the player.
        opponent: Name of the opponent team.

    Returns:
        A dictionary mapping each stat to the mean value against that opponent.
        If the player has no recorded games against the opponent, returns an
        empty dict.
    """
    mask = (df["player"] == player) & (df["opponent"] == opponent)
    h2h_games = df[mask]
    if h2h_games.empty:
        return {}
    return {stat: h2h_games[stat].mean() for stat in STATS}


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare features and targets for training ML models.

    For this simple prototype we use the following features:
        - player's last 3 game average for each stat
        - whether the game was at home (1) or away (0)
        - whether the player was flagged as injured (1) or not (0)

    Args:
        df: DataFrame containing game logs.

    Returns:
        X: Feature DataFrame
        y: Target DataFrame (one column per stat)
    """
    # Sort by date for computing rolling means
    df_sorted = df.sort_values(["player", "date"])

    # Compute rolling mean of the last 3 games for each stat
    rolling_means = (
        df_sorted.groupby("player")[STATS]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .add_suffix("_recent")
    )

    X = pd.DataFrame()
    for stat in STATS:
        X[f"{stat}_recent"] = rolling_means[f"{stat}_recent"]

    # Encode home/away as binary
    X["home"] = df_sorted["home"].astype(int)
    # Encode injury flag as binary
    X["injured"] = df_sorted["injured"].astype(int)

    # Target variables
    y = df_sorted[STATS].copy()

    return X, y


def train_models(df: pd.DataFrame) -> Dict[str, LinearRegression]:
    """Train a linear regression model for each stat.

    Args:
        df: DataFrame containing game logs.

    Returns:
        A dictionary mapping stat names to trained LinearRegression models.
    """
    X, y = prepare_training_data(df)

    models: Dict[str, LinearRegression] = {}
    for stat in STATS:
        model = LinearRegression()
        model.fit(X, y[stat])
        models[stat] = model
    return models


def predict_with_models(models: Dict[str, LinearRegression], df: pd.DataFrame, player: str, opponent: str) -> Dict[str, float]:
    """Use the trained models to predict the baseline stat line for a given matchup.

    Args:
        models: Dict of trained LinearRegression models keyed by stat name.
        df: DataFrame of game logs (used to compute features for the given player).
        player: Player name.
        opponent: Opponent team name (unused in the current features but included for consistency).

    Returns:
        A dict mapping each stat to the model prediction.
    """
    # Compute recent means for each stat to serve as features
    recent = compute_recent_form(df, player, n=3)
    features = []
    for stat in STATS:
        features.append(recent[stat][0])  # mean of last 3 games
    # Append home/away (we don't know for future games; assume neutral = 0)
    features.append(0)
    # Append injury flag (assume healthy = 0)
    features.append(0)

    feature_array = np.array(features).reshape(1, -1)
    preds: Dict[str, float] = {}
    for idx, stat in enumerate(STATS):
        model = models[stat]
        preds[stat] = float(model.predict(feature_array)[0])
    return preds


def run_monte_carlo_simulation(
    base_means: Dict[str, Tuple[float, float]],
    h2h_adjustments: Dict[str, float],
    n_simulations: int = 10000,
) -> Dict[str, float]:
    """Run Monte‑Carlo simulations for each stat.

    Args:
        base_means: Dict mapping stat to a (mean, std) tuple from recent form.
        h2h_adjustments: Dict mapping stat to the mean value against the opponent.
        n_simulations: Number of simulation iterations.

    Returns:
        Dict mapping each stat to the median of simulated outcomes.
    """
    results: Dict[str, float] = {}
    rng = np.random.default_rng()
    for stat in STATS:
        mean, std = base_means[stat]
        if std == 0:
            std = max(1.0, mean * 0.1)  # avoid zero variance
        # Sample from normal distribution for the stat
        samples = rng.normal(loc=mean, scale=std, size=n_simulations)
        # If we have head‑to‑head data, shift the samples towards that mean
        if stat in h2h_adjustments:
            # Blend 50/50 between recent mean and h2h mean
            h2h_mean = h2h_adjustments[stat]
            samples = (samples + h2h_mean) / 2.0
        # Clamp to non‑negative values (can't score negative points)
        samples = np.maximum(samples, 0)
        results[stat] = float(np.median(samples))
    return results


def simulate_player_stats(df: pd.DataFrame, models: Dict[str, LinearRegression], player: str, opponent: str) -> Dict[str, float]:
    """Combine ML predictions and Monte‑Carlo simulation to project a stat line.

    Args:
        df: DataFrame of game logs.
        models: Trained models for baseline predictions.
        player: Player name.
        opponent: Opponent team name.

    Returns:
        Dict mapping each stat to its projected value.
    """
    # Compute recent form (mean and std) for simulation
    recent_stats = compute_recent_form(df, player, n=5)
    # Compute head‑to‑head adjustments if available
    h2h = compute_head_to_head(df, player, opponent)
    # Run Monte‑Carlo simulation
    sim_results = run_monte_carlo_simulation(recent_stats, h2h, n_simulations=5000)
    # Get ML predictions
    ml_preds = predict_with_models(models, df, player, opponent)
    # Combine by averaging simulation median and ML prediction for each stat
    final_results: Dict[str, float] = {}
    for stat in STATS:
        final_results[stat] = float((sim_results[stat] + ml_preds[stat]) / 2.0)
    return final_results