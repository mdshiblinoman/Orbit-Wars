"""
Orbit Wars - Nearest Planet Sniper Agent

A simple agent that captures the nearest unowned planet when it has
enough ships to guarantee the takeover.

Strategy:
  For each planet we own, find the closest planet we don't own.
  If we have more ships than the target's garrison, send exactly
  enough to capture it (garrison + 1). Otherwise, wait and accumulate.

Key concepts demonstrated:
  - Parsing the observation (planets, player ID)
  - Computing angles with atan2 for fleet direction
  - Sending moves as [from_planet_id, angle, num_ships]
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from typing import Iterable, List, Optional


def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    # Parse into named tuples for readable field access:
    #   Planet(id, owner, x, y, radius, ships, production)
    #   owner == -1 means neutral, 0-3 are player IDs
    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float("inf")
        for t in targets:
            dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # We need to send more ships than the target has to capture it.
        # Exactly target_ships + 1 guarantees the takeover.
        ships_needed = nearest.ships + 1

        # Only launch if we can afford it — otherwise keep accumulating
        if mine.ships >= ships_needed:
            # atan2(dy, dx) gives the angle from our planet to the target
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves


def draw_state(planets: Iterable[Planet], moves: Optional[List[list]] = None, figsize=(6, 6), title="Orbit Wars"):
    """Draw a simple visualization of the planets and optional moves.

    planets: iterable of Planet(id, owner, x, y, radius, ships, production)
    moves: list of [from_planet_id, angle (radians), ships]
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # color map for owners
    owner_colors = {
        -1: "#9aa0a6",  # neutral gray
        0: "#1f77b4",   # blue
        1: "#d62728",   # red
        2: "#2ca02c",   # green
        3: "#ff7f0e",   # orange
    }

    id_to_planet = {}
    xs = []
    ys = []

    for p in planets:
        id_to_planet[p.id] = p
        color = owner_colors.get(p.owner, "#7f7f7f")
        circle = Circle((p.x, p.y), p.radius, facecolor=color, edgecolor="k", alpha=0.8)
        ax.add_patch(circle)
        ax.text(p.x, p.y, f"{p.id}\n{p.ships}", ha="center", va="center", color="white", fontsize=8, weight="bold")
        xs.append(p.x)
        ys.append(p.y)

    # draw moves as arrows from source planet center in given angle
    if moves:
        for mv in moves:
            try:
                src_id, angle, ships = mv
            except ValueError:
                continue
            src = id_to_planet.get(src_id)
            if src is None:
                continue
            # arrow length scaled by planet radius and ships (for visibility)
            length = max(0.3, min(2.0, ships ** 0.5 * 0.1))
            dx = math.cos(angle) * length
            dy = math.sin(angle) * length
            arrow = FancyArrowPatch((src.x, src.y), (src.x + dx, src.y + dy), arrowstyle='->', mutation_scale=12, color='black')
            ax.add_patch(arrow)

    # set limits with padding
    if xs and ys:
        pad = 1.5
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    return fig, ax


def visualize_obs(obs, moves: Optional[List[list]] = None):
    """Create planets from an observation dict or object and visualize."""
    if isinstance(obs, dict):
        raw_planets = obs.get("planets", [])
        planets = [Planet(*p) for p in raw_planets]
    else:
        planets = obs.planets

    draw_state(planets, moves=moves)


if __name__ == "__main__":
    # Demo: create a small example and visualize the agent's planned moves.
    sample_raw = [
        # id, owner, x, y, radius, ships, production
        (0, 0, -2.5, 0.0, 0.8, 25, 1),
        (1, -1, 0.0, 0.0, 0.7, 12, 2),
        (2, 1, 2.5, 0.0, 0.9, 8, 1),
        (3, -1, 0.0, 2.5, 0.6, 5, 0),
    ]
    obs = {"player": 0, "planets": sample_raw}
    # compute agent moves for the demo observation
    moves = agent(obs)
    visualize_obs(obs, moves=moves)
    print("Agent moves:", moves)
    plt.show()
