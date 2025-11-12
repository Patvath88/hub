# NBA Player Prop Simulator

This is a simplified, self‑contained prototype of a deployable web application that predicts NBA player prop lines using a combination of Monte‑Carlo simulation and machine‑learning.  The project is meant as a working proof of concept that can be extended with real data sources (e.g. the NBA Stats API and BallDontLie) once an internet connection is available.

## Features

* **Data ingestion** – loads historical player game logs from a CSV (sample data included).  In a production setting you would replace this with API calls to fetch real data.
* **Monte‑Carlo simulation** – runs thousands of simulations of a player’s box score for a selected matchup, producing a median projection per stat.
* **Machine learning model** – trains a simple linear regression model for each stat using the sample data and uses it to estimate each stat’s expected value given context features (recent form, head‑to‑head results and simple injury flags).  The ML predictions are blended with the simulation results.
* **REST API** – built with FastAPI; exposes a `/simulate` endpoint that accepts a player name and opponent name and returns predicted stat lines in JSON.
* **Web front‑end** – a minimal HTML/JavaScript front‑end that calls the API and displays the projections to the user.  No external libraries are required for the UI.

## Project structure

```
nba_prop_simulator/
│   README.md          # this file
│   app.py             # FastAPI application
│   simulate.py        # simulation and ML logic
│   test_app.py        # automated test for the API
│
├── data/
│   └── sample_games.csv   # example game logs for a few players
│
├── static/
│   ├── index.html     # simple front‑end page
│   └── app.js         # JavaScript to call the API and update UI
```

## Running the application

1. Install dependencies using `pip` (if not already installed):

   ```bash
   pip install fastapi uvicorn pandas numpy scikit‑learn
   ```

2. Start the development server from the project root:

   ```bash
   uvicorn app:app --reload --port 8000
   ```

   The API will be available at `http://localhost:8000`.  You can also navigate to `http://localhost:8000/static/index.html` in your browser to use the built‑in UI.

3. Run the tests:

   ```bash
   python test_app.py
   ```

## Extending to real data

The included `sample_games.csv` file contains artificially generated statistics for a handful of players and opponents to demonstrate the end‑to‑end workflow.  To connect the simulator to real data:

1. Replace the `load_game_logs()` function in `simulate.py` with code that calls the NBA Stats API and BallDontLie API to fetch player game logs and returns a pandas DataFrame with at least these columns:
   * `player` – the player’s name or ID
   * `opponent` – the opposing team’s name or ID
   * `points`, `rebounds`, `assists`, `steals`, `blocks`, `turnovers`, `fg_attempts`, `ft_attempts`, `minutes` – basic box score stats
   * `date` – game date
   * optional contextual features like `home` (True/False), `injured` (True/False) etc.

2. Add additional features to the `prepare_training_data()` function to improve the ML model.  For example, incorporate pace, usage rate, player efficiency ratings, and more detailed injury context.  You can also experiment with more sophisticated models (random forests, gradient boosting, neural networks).

3. Increase the number of simulation samples in `simulate()` to 1,000,000 (the current default is lower for performance on small machines).  Ensure that your server has enough CPU to handle the larger simulations.

This prototype provides a starting point; with real data and tuning you can build a powerful and accurate player prop prediction service.