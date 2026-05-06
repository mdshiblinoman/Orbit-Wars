# Orbit Wars Starter (Baseline + Streamlit Website)

This workspace now contains:

- A simple, legal baseline bot in `agent.py`.
- A Streamlit website app in `app.py` that presents the game guide/specs and runs your bot against custom observation JSON.
- Two standalone tabular training scripts for the leaderboard CSV: `ml.py` for classical ML and `dl.py` for a small neural network.

## What it does

- Expands to neutral planets first.
- Attacks weak enemy planets when possible.
- Keeps a minimum defensive garrison on owned planets.
- Skips launches whose direct path crosses the sun.

## Action format reminder

Your bot must return:

```python
[[from_planet_id, direction_angle, num_ships], ...]
```

or `[]` for no action.

## Usage on Kaggle

1. Open the Orbit Wars notebook/starter kit.
2. Put the contents of `agent.py` into your agent cell/file.
3. Run local matches vs built-in bots.
4. Iterate on heuristics and submit.

## Run the Streamlit website locally

Use your active Python environment (in this workspace, conda was detected):

```bash
/home/noman/anaconda3/bin/python -m streamlit run app.py
```

Then open the local URL shown in terminal (usually http://localhost:8501).

Inside the app:

1. Read the Game Guide and Specs tabs.
2. Open Run Agent tab.
3. Paste or edit observation JSON.
4. Click Run Agent to execute `agent.py` and validate action format.

## Easy next upgrades

- Predict orbiting planet future positions before aiming.
- Route around the sun with two-hop launches.
- Estimate incoming enemy fleets and reserve defenders.
- Coordinate multi-planet attacks on high-production targets.

## Train on the CSV

The leaderboard CSV in this workspace is treated as a small tabular regression dataset by default, with `Score` as the target and `Rank`/`TeamId` dropped from features.

Run the classical ML script:

```bash
python ml.py --csv orbit-wars-publicleaderboard-2026-05-05T15:36:33.csv
```

Run the neural-network script:

```bash
python dl.py --csv orbit-wars-publicleaderboard-2026-05-05T15:36:33.csv
```

If you want a different target column, pass `--target <column_name>`.
