"""Render a simple Orbit Wars animation and save as MP4.

This script runs the `agent.agent(obs)` function from `agent.py` on the
DEFAULT_OBS from `app.py`, simulates fleets travelling at `shipSpeed`, and
produces a short MP4 animation `orbit_wars_demo.mp4`.

It's a lightweight visualizer aimed for demos, not a production-grade
simulator. It resolves fleet arrivals by simple compare-with-garrison rules.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import app

DEFAULT_OBS = app.DEFAULT_OBS


def _get_ship_speed() -> float:
    for k, v, _ in app.CONFIG_ROWS:
        if k == "shipSpeed":
            try:
                return float(v)
            except Exception:
                break
    return 6.0


def _get_angular_velocity() -> float:
    return float(getattr(app, 'DEFAULT_OBS', {}).get('angular_velocity', 0.03))


@dataclass
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int
    orbiting: bool = False
    orbit_radius: float = 0.0
    angle: float = 0.0


@dataclass
class Fleet:
    owner: int
    x: float
    y: float
    angle: float
    ships: int
    speed: float
    target_pid: int

    def step(self, dt: float) -> None:
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt


def build_planets(obs: Dict[str, Any]) -> List[Planet]:
    plist: List[Planet] = []
    for p in obs.get("planets", []):
        pid, owner, x, y, r, ships, prod = p
        pl = Planet(int(pid), int(owner), float(x), float(y), float(r), int(ships), int(prod))
        # determine orbital parameters relative to center (50,50)
        cx, cy = 50.0, 50.0
        dx = pl.x - cx
        dy = pl.y - cy
        radius_from_center = math.hypot(dx, dy)
        # orbit if inside the playable ring (heuristic from app.py)
        if radius_from_center + pl.radius < 50.0:
            pl.orbiting = True
            pl.orbit_radius = radius_from_center
            pl.angle = math.atan2(dy, dx)
        else:
            pl.orbiting = False
            pl.orbit_radius = 0.0
            pl.angle = 0.0
        plist.append(pl)
    return plist


def obs_from_state(planets: List[Planet], fleets: List[Fleet], player: int = 0) -> Dict[str, Any]:
    planets_list = [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production] for p in planets]
    fleets_list = []
    for i, f in enumerate(fleets):
        fleets_list.append([i, f.owner, f.x, f.y, f.angle, f.target_pid, f.ships])
    return {"player": player, "planets": planets_list, "fleets": fleets_list, "comets": []}


def find_target_by_pid(planets: List[Planet], pid: int) -> Planet:
    for p in planets:
        if p.id == pid:
            return p
    raise KeyError(pid)


def nearest_planet_in_direction(src: Planet, angle: float, planets: List[Planet]) -> Planet:
    best = None
    best_proj = -1e9
    dx = math.cos(angle)
    dy = math.sin(angle)
    for p in planets:
        if p.id == src.id:
            continue
        vx = p.x - src.x
        vy = p.y - src.y
        proj = vx * dx + vy * dy
        if proj <= 0:
            continue
        if proj > best_proj:
            best_proj = proj
            best = p
    return best if best is not None else min([p for p in planets if p.id != src.id], key=lambda q: math.hypot(q.x - src.x, q.y - src.y))


def simulate_and_render(duration_turns: int = 60, fps: int = 20, frames_per_turn: int = 4):
    # initial state
    planets = build_planets(DEFAULT_OBS)
    fleets: List[Fleet] = []
    ship_speed = _get_ship_speed()

    # prepare plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_title("Orbit Wars Demo")

    imgs = []

    # dynamic import of agent to pick up changes
    mod = importlib.import_module("agent")
    mod = importlib.reload(mod)

    for turn in range(duration_turns):
        # call agent at start of turn
        obs = obs_from_state(planets, fleets, player=0)
        try:
            actions = mod.agent(obs)
        except Exception:
            actions = []

        # convert actions to fleets
        for a in actions or []:
            try:
                from_pid, ang, num = a
                src = find_target_by_pid(planets, int(from_pid))
                target = nearest_planet_in_direction(src, float(ang), planets)
                fleets.append(Fleet(0, src.x, src.y, float(ang), int(num), ship_speed, int(target.id)))
                # subtract ships immediately for visualization
                src.ships = max(0, src.ships - int(num))
            except Exception:
                continue

        # each turn we produce 'frames_per_turn' frames to smooth motion
        for f in range(frames_per_turn):
            ax.clear()
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")
            # advance orbital positions for orbiting planets
            ang_vel = _get_angular_velocity()
            cx, cy = 50.0, 50.0
            for p in planets:
                if p.orbiting:
                    p.angle += ang_vel / frames_per_turn
                    p.x = cx + p.orbit_radius * math.cos(p.angle)
                    p.y = cy + p.orbit_radius * math.sin(p.angle)
            # draw sun
            sun = patches.Circle((50, 50), radius=10.0, color="#ffddaa")
            ax.add_patch(sun)

            # draw planets
            for p in planets:
                color = "tab:blue" if p.owner == 0 else ("tab:green" if p.owner == -1 else "tab:red")
                circ = patches.Circle((p.x, p.y), radius=p.radius, color=color, alpha=0.9)
                ax.add_patch(circ)
                ax.text(p.x, p.y, f"{p.id}\n{p.ships}", ha="center", va="center", color="white", fontsize=8)

            # step fleets and draw
            for fleet in list(fleets):
                fleet.step(1.0 / frames_per_turn)
                ax.arrow(fleet.x, fleet.y, math.cos(fleet.angle) * 0.5, math.sin(fleet.angle) * 0.5, head_width=0.8, color="k")
                # check arrival
                try:
                    tgt = find_target_by_pid(planets, fleet.target_pid)
                except KeyError:
                    continue
                if math.hypot(fleet.x - tgt.x, fleet.y - tgt.y) <= tgt.radius + 0.5:
                    # resolve combat simply
                    if fleet.ships > tgt.ships:
                        tgt.owner = fleet.owner
                        tgt.ships = fleet.ships - tgt.ships
                    else:
                        tgt.ships = max(0, tgt.ships - fleet.ships)
                    try:
                        fleets.remove(fleet)
                    except ValueError:
                        pass

            # production occurs once per turn at the end of the turn (visualize at last frame)
            if f == frames_per_turn - 1:
                for p in planets:
                    if p.owner >= 0:
                        p.ships += p.production

            ax.set_xticks([])
            ax.set_yticks([])
            fig.canvas.draw()
            # Some backends expose ARGB buffer; convert to RGB array.
            buf = fig.canvas.tostring_argb()
            data = np.frombuffer(buf, dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            data = data.reshape((h, w, 4))
            # ARGB -> RGB
            data = data[:, :, 1:4].copy()
            imgs.append(data)

    # save using imageio-ffmpeg via imageio
    import imageio

    out_path = "orbit_wars_demo.gif"
    print(f"Saving {out_path} ({len(imgs)} frames)...")
    # Save as GIF using imageio (avoids ffmpeg dependency in minimal envs).
    imageio.mimsave(out_path, imgs, duration=1.0 / fps)
    print("Saved.")


if __name__ == "__main__":
    simulate_and_render(duration_turns=40, fps=20, frames_per_turn=3)
