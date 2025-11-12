"""
FastAPI application exposing the NBA player prop simulator service.

This module wires together the simulation logic from ``simulate.py`` and
serves a REST API for predicting a player's stat line in an upcoming
matchup.  It also hosts a minimal static front‑end (index.html) under
the ``/static`` route.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from typing import Dict

from simulate import (
    load_game_logs,
    train_models,
    simulate_player_stats,
)

app = FastAPI(title="NBA Player Prop Simulator", version="0.1.0")

# Load data and train models on startup
game_logs = load_game_logs()
models = train_models(game_logs)

# Mount the static directory to serve the front‑end
app.mount("/static", StaticFiles(directory=(__file__).rsplit("/", 1)[0] + "/static"), name="static")


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint providing a welcome message and usage hints."""
    return {
        "message": "Welcome to the NBA Player Prop Simulator API",
        "usage": "Call /simulate?player=LeBron%20James&opponent=Boston%20Celtics to get projections",
    }


@app.get("/simulate")
async def simulate(
    player: str = Query(..., description="Name of the NBA player"),
    opponent: str = Query(..., description="Name of the opposing team"),
) -> JSONResponse:
    """Return projected stat line for the given player and opponent.

    The result includes points, rebounds, assists, steals, blocks and turnovers.
    It combines a simple ML model with a Monte‑Carlo simulation based on
    recent form and head‑to‑head performance.
    """
    # Check if the player exists in our dataset
    if player not in game_logs["player"].unique():
        raise HTTPException(status_code=404, detail=f"Player '{player}' not found in dataset")
    # Check if the opponent exists in our dataset
    if opponent not in game_logs["opponent"].unique():
        raise HTTPException(status_code=404, detail=f"Opponent '{opponent}' not found in dataset")
    results = simulate_player_stats(game_logs, models, player, opponent)
    return JSONResponse(content={"player": player, "opponent": opponent, "projections": results})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)