"""Orbit Wars starter bot.

Baseline strategy:
- Expand to nearby high-value neutral planets first.
- Opportunistically attack weak enemy planets.
- Keep a minimum garrison on each owned planet.
- Avoid launches whose straight path would cross the sun.

Action format returned:
[[from_planet_id, direction_angle, num_ships], ...]
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

# Planet tuple layout: [id, owner, x, y, radius, ships, production]
PID, POWNER, PX, PY, PR, PSHIPS, PPROD = range(7)

CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MIN_GARRISON = 8
MAX_LAUNCH_FRACTION = 0.6


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _point_to_segment_distance(
    p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
) -> float:
    ax, ay = a
    bx, by = b
    px, py = p

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def _crosses_sun(src: Sequence[float], dst: Sequence[float]) -> bool:
    a = (float(src[PX]), float(src[PY]))
    b = (float(dst[PX]), float(dst[PY]))
    d = _point_to_segment_distance((CENTER_X, CENTER_Y), a, b)
    # Add a small safety margin to account for launch offset and movement discretization.
    return d <= (SUN_RADIUS + 0.5)


def _target_score(src: Sequence[float], dst: Sequence[float], me: int) -> float:
    distance = _dist((src[PX], src[PY]), (dst[PX], dst[PY]))
    ships = float(dst[PSHIPS])
    prod = float(dst[PPROD])

    # Prefer neutral expansion, then weak enemies, while discounting long travel.
    owner_bonus = 7.0 if dst[POWNER] == -1 else 3.0
    return prod * 12.0 + owner_bonus - ships * 1.6 - distance * 0.8


def _required_attack_ships(dst: Sequence[float]) -> int:
    # Small buffer helps absorb in-flight production/arrivals and ties.
    return int(math.ceil(float(dst[PSHIPS]) * 1.2 + 3.0))


def agent(obs: Dict) -> List[List[float]]:
    planets: List[Sequence[float]] = obs.get("planets", [])
    me = int(obs.get("player", 0))

    my_planets = [p for p in planets if int(p[POWNER]) == me]
    if not my_planets:
        return []

    candidate_targets = [p for p in planets if int(p[POWNER]) != me]
    if not candidate_targets:
        return []

    # Track planned sends to avoid overspending from a source planet.
    planned_outgoing: Dict[int, int] = {int(p[PID]): 0 for p in my_planets}
    actions: List[List[float]] = []

    # Stronger planets act first.
    my_planets.sort(key=lambda p: float(p[PSHIPS]), reverse=True)

    for src in my_planets:
        src_id = int(src[PID])
        src_ships = int(src[PSHIPS]) - planned_outgoing[src_id]

        available = src_ships - MIN_GARRISON
        if available <= 0:
            continue

        max_send = max(0, int(src_ships * MAX_LAUNCH_FRACTION))
        send_budget = min(available, max_send)
        if send_budget <= 0:
            continue

        ranked = sorted(
            candidate_targets,
            key=lambda t: _target_score(src, t, me),
            reverse=True,
        )

        for dst in ranked:
            if _crosses_sun(src, dst):
                continue

            needed = _required_attack_ships(dst)
            if needed > send_budget:
                continue

            theta = _angle((src[PX], src[PY]), (dst[PX], dst[PY]))
            actions.append([src_id, float(theta), int(needed)])
            planned_outgoing[src_id] += needed
            break

    return actions
