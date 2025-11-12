"""
Unit tests for the NBA player prop simulator API.

This script uses FastAPI's TestClient to verify that the /simulate endpoint
returns a response for known players and opponents and that error handling
works for unknown inputs.
"""

"""
Simple integration tests for the NBA player prop simulator API.

This module can be run directly with ``python test_app.py``.  It uses
FastAPI's TestClient to send requests to the ``/simulate`` endpoint and
performs basic assertions.  If any assertion fails, an AssertionError
will be raised.  On success, a short message will be printed.
"""

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_simulate_valid_player_opponent() -> None:
    """Test that a valid player/opponent returns a 200 response with all stats."""
    response = client.get(
        "/simulate",
        params={"player": "LeBron James", "opponent": "Boston Celtics"},
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["player"] == "LeBron James"
    assert data["opponent"] == "Boston Celtics"
    for stat in ["points", "rebounds", "assists", "steals", "blocks", "turnovers"]:
        assert stat in data["projections"], f"Missing stat {stat}"
        assert isinstance(data["projections"][stat], (int, float)), f"Stat {stat} not numeric"


def test_simulate_unknown_player() -> None:
    """Test that an unknown player returns a 404 response."""
    response = client.get(
        "/simulate",
        params={"player": "Unknown Player", "opponent": "Boston Celtics"},
    )
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    data = response.json()
    assert "Player 'Unknown Player' not found" in data["detail"], data["detail"]


def test_simulate_unknown_opponent() -> None:
    """Test that an unknown opponent returns a 404 response."""
    response = client.get(
        "/simulate",
        params={"player": "LeBron James", "opponent": "Unknown Team"},
    )
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    data = response.json()
    assert "Opponent 'Unknown Team' not found" in data["detail"], data["detail"]


if __name__ == "__main__":
    # Run tests sequentially
    test_simulate_valid_player_opponent()
    test_simulate_unknown_player()
    test_simulate_unknown_opponent()
    print("All tests passed.")