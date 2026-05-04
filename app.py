import importlib
import json
from typing import Any, Dict, List

import streamlit as st

st.set_page_config(page_title="Orbit Wars Starter Site", layout="wide")

DEFAULT_OBS = {
    "player": 0,
    "angular_velocity": 0.03,
    "remainingOverageTime": 60.0,
    "planets": [
        [0, 0, 10.0, 10.0, 2.5, 40, 3],
        [1, -1, 22.0, 13.0, 2.0, 12, 2],
        [2, -1, 35.0, 25.0, 2.0, 8, 1],
        [3, 1, 85.0, 85.0, 2.7, 35, 4],
    ],
    "fleets": [],
    "initial_planets": [],
    "comets": [],
    "comet_planet_ids": [],
}

CONFIG_ROWS = [
    ("episodeSteps", "500", "Maximum number of turns"),
    ("actTimeout", "1", "Seconds per turn"),
    ("shipSpeed", "6.0", "Maximum fleet speed"),
    ("sunRadius", "10.0", "Radius of the sun"),
    ("boardSize", "100.0", "Board dimensions"),
    ("cometSpeed", "4.0", "Comet speed (units/turn)"),
]


@st.cache_data
def _action_format_hint() -> str:
    return json.dumps([["from_planet_id", "direction_angle", "num_ships"]], indent=2)


def _render_header() -> None:
    st.title("Orbit Wars - Streamlit Starter Website")
    st.caption(
        "Interactive overview of rules plus a live runner for your current agent function."
    )



def _render_overview() -> None:
    st.subheader("How to Play Orbit Wars")
    st.markdown(
        """
Players begin with one home planet and compete to control the map over 500 turns.

- Board is a 100 x 100 continuous space.
- Sun is centered at (50, 50). Fleets crossing it are destroyed.
- Planets and comets are generated with 4-fold symmetry.
- Winner is the player with the highest final ship total:
  ships on owned planets plus ships in owned fleets.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Planets")
        st.markdown(
            """
- Planet structure: [id, owner, x, y, radius, ships, production].
- owner is -1 for neutral, otherwise player id.
- production range: 1 to 5 ships per turn.
- orbiting planets rotate if orbital radius plus planet radius is less than 50.
- static planets stay fixed.
"""
        )

        st.markdown("### Fleets")
        st.markdown(
            """
- Fleet structure: [id, owner, x, y, angle, from_planet_id, ships].
- Movement each turn is a straight segment at speed based on fleet size.
- Fleets are removed if they leave bounds, hit the sun, or collide with planets.
"""
        )

    with col2:
        st.markdown("### Comets")
        st.markdown(
            """
- Spawn in groups of four at turns 50, 150, 250, 350, 450.
- Radius is fixed at 1.0 and production is 1.
- Comets are planets for combat and launching.
- Departing comets are removed before launches that turn.
"""
        )

        st.markdown("### Turn Order")
        st.markdown(
            """
1. Comet expiration
2. Comet spawning
3. Fleet launch
4. Production
5. Fleet movement
6. Planet rotation and comet movement
7. Combat resolution
"""
        )


def _render_specs() -> None:
    st.subheader("Observation and Actions")
    st.markdown("Observation fields include planets, fleets, player, initial_planets, comets and comet_planet_ids.")

    st.markdown("Action format your agent must return:")
    st.code(_action_format_hint(), language="json")

    st.subheader("Configuration")
    st.table(
        {
            "Parameter": [x[0] for x in CONFIG_ROWS],
            "Default": [x[1] for x in CONFIG_ROWS],
            "Description": [x[2] for x in CONFIG_ROWS],
        }
    )


def _validate_actions(actions: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(actions, list):
        return ["Agent output must be a list."]

    for i, move in enumerate(actions):
        if not isinstance(move, list) or len(move) != 3:
            errors.append(f"Move {i} must be [from_planet_id, direction_angle, num_ships].")
            continue

        pid, ang, ships = move
        if not isinstance(pid, int):
            errors.append(f"Move {i}: from_planet_id must be int.")
        if not isinstance(ang, (int, float)):
            errors.append(f"Move {i}: direction_angle must be numeric.")
        if not isinstance(ships, int):
            errors.append(f"Move {i}: num_ships must be int.")
        if isinstance(ships, int) and ships <= 0:
            errors.append(f"Move {i}: num_ships must be > 0.")

    return errors


def _run_agent(obs: Dict[str, Any]) -> Any:
    mod = importlib.import_module("agent")
    mod = importlib.reload(mod)
    if not hasattr(mod, "agent"):
        raise AttributeError("No function named agent(obs) found in agent.py")
    return mod.agent(obs)


def _render_runner() -> None:
    st.subheader("Live Agent Runner")
    st.markdown("Edit the observation JSON and run your current agent function from agent.py.")

    obs_text = st.text_area(
        "Observation JSON",
        value=json.dumps(DEFAULT_OBS, indent=2),
        height=380,
    )

    left, right = st.columns([1, 3])
    with left:
        run_btn = st.button("Run Agent", type="primary")
    with right:
        st.caption("The app reloads agent.py on each run so your latest changes are used.")

    if not run_btn:
        return

    try:
        obs = json.loads(obs_text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return

    try:
        actions = _run_agent(obs)
    except Exception as exc:
        st.exception(exc)
        return

    errors = _validate_actions(actions)
    if errors:
        st.error("Returned action format is invalid.")
        for err in errors:
            st.write(f"- {err}")
    else:
        st.success("Agent executed successfully.")

    st.markdown("Returned actions")
    st.code(json.dumps(actions, indent=2), language="json")


def main() -> None:
    _render_header()

    tab_overview, tab_specs, tab_runner = st.tabs(
        ["Game Guide", "Specs", "Run Agent"]
    )

    with tab_overview:
        _render_overview()

    with tab_specs:
        _render_specs()

    with tab_runner:
        _render_runner()


if __name__ == "__main__":
    main()
