"""Numerical Gymnasium interface to Night Shift; no graphics dependency.

Import this module to register ``NightShift-v2``. The observation describes
vehicle dynamics, the task, three patrol cars and local obstacle clearance.
Police positions are visible through walls, as on the game's minimap, but
their private routing memory is hidden: this is a partially observed task.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry

from simulation import CAR_RADIUS, ESCAPE_UNLOCK, NITRO_SPEED, ROAD_CENTERS, WORLD_SIZE, Controls, Game
from pursuit import PursuitDirector


ENV_ID = "NightShift-v2"
CONTROL_DT = 0.1  # One agent decision per 12 unchanged 120 Hz physics steps.
RAY_RANGE = 600.0
RAY_COUNT = 16
MAX_COPS = 3
WORLD_DIAGONAL = math.sqrt(2) * WORLD_SIZE
VELOCITY_SCALE = 2 * NITRO_SPEED

# Fixed discrete controls can be used by both DQN and PPO. Negative throttle
# brakes first and becomes reverse at low speed, just like S in the game.
ACTION_NAMES = (
    "coast", "accelerate", "accelerate_left", "accelerate_right",
    "brake_reverse", "brake_reverse_left", "brake_reverse_right",
    "coast_left", "coast_right", "drift_left", "drift_right",
    "nitro", "nitro_left", "nitro_right", "handbrake",
)
ACTIONS = (
    Controls(),
    Controls(throttle=1),
    Controls(throttle=1, steer=-1),
    Controls(throttle=1, steer=1),
    Controls(throttle=-1),
    Controls(throttle=-1, steer=-1),
    Controls(throttle=-1, steer=1),
    Controls(steer=-1),
    Controls(steer=1),
    Controls(throttle=1, steer=-1, brake=True),
    Controls(throttle=1, steer=1, brake=True),
    Controls(throttle=1, nitro=True),
    Controls(throttle=1, steer=-1, nitro=True),
    Controls(throttle=1, steer=1, nitro=True),
    Controls(brake=True),
)

OBSERVATION_NAMES = (
    "x", "y", "heading_cos", "heading_sin", "velocity_forward", "velocity_right",
    "yaw_rate", "steering", "traction_loss", "reverse_gear", "health", "nitro",
    "capture", "nitro_cooldown", "nitro_empty", "damage_cooldown", "goal_forward",
    "goal_right", "escape_open", "unlock_remaining", "episode_remaining", "road_dx", "road_dy",
) + tuple(
    f"cop_{i}_{feature}" for i in range(MAX_COPS)
    for feature in ("present", "forward", "right", "velocity_forward", "velocity_right",
                    "heading_cos", "heading_sin")
) + tuple(f"wall_{i:02d}" for i in range(RAY_COUNT))

REWARD_TERMS = ("survival", "time", "progress", "damage", "capture", "terminal")


class NightShiftEnv(gym.Env):
    """The existing chase, controlled at 10 Hz using 60 numerical features.

    ``max_steps`` is an external rollout cap (default: 180 simulated seconds).
    Success/capture/destruction terminate; an unfinished chase at the cap is
    truncated. ``render_mode='ansi'`` supplies a text summary for debugging.
    Neither construction nor stepping imports Pygame or opens a window.
    ``reset(options={'start_offset':(dx,dy,degrees)})`` can move the player
    by at most 30px on each axis and rotate the heading by at most 10degrees.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None, max_steps: int = 1800):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(len(OBSERVATION_NAMES),), dtype=np.float32)
        self.game = Game()
        self.episode_steps = 0
        self.episode_seed = None
        self.start_offset = (0.0,0.0,0.0)
        self._needs_reset = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = {} if options is None else options
        if not isinstance(options,dict) or set(options)-{"start_offset"}:
            raise ValueError("Only reset option 'start_offset' is supported")
        offset = options.get("start_offset",(0.0,0.0,0.0))
        if isinstance(offset,np.ndarray):
            offset = offset.tolist()
        if not isinstance(offset,(tuple,list)) or len(offset)!=3:
            raise ValueError("start_offset must be (dx,dy,angle_degrees)")
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) for x in offset):
            raise ValueError("start_offset must contain three real numbers")
        offset = tuple(float(x) for x in offset)
        if any(not math.isfinite(x) or abs(x)>limit for x,limit in zip(offset,(30,30,10))):
            raise ValueError("start_offset limits are 30px,30px and10degrees")
        super().reset(seed=seed)
        # Do not reseed the global RNG or reuse Game.reset(None)'s old seed.
        # A seed alone keeps the fixed start; offsets are supplied explicitly.
        self.episode_seed = int(self.np_random.integers(0, 2**31))
        self.game.reset(self.episode_seed)
        self.start_offset = offset
        player = self.game.player
        player.x += offset[0]
        player.y += offset[1]
        player.angle += math.radians(offset[2])
        if not (CAR_RADIUS<player.x<WORLD_SIZE-CAR_RADIUS
                and CAR_RADIUS<player.y<WORLD_SIZE-CAR_RADIUS):
            raise ValueError("The start must stay inside the city")
        for building in self.game.buildings:
            if (building.x-CAR_RADIUS<=player.x<=building.x+building.w+CAR_RADIUS
                    and building.y-CAR_RADIUS<=player.y<=building.y+building.h+CAR_RADIUS):
                raise ValueError("The start must not intersect a building")
        if any(math.hypot(c.x-player.x,c.y-player.y)<=2*CAR_RADIUS for c in self.game.cops):
            raise ValueError("The start must not intersect a police car")
        if any(offset):
            # Dispatch must observe the moved player, not the old start.
            self.game._pursuit = PursuitDirector(self.game,Controls,ROAD_CENTERS,WORLD_SIZE)
        self.episode_steps = 0
        self._needs_reset = False
        return self._get_obs(), self._get_info()

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping or after an episode ends")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {len(ACTIONS) - 1}]")

        game = self.game
        elapsed_before = game.elapsed
        health_before = game.health
        capture_before = game.capture
        distance_before = self._goal_distance()
        escape_was_open = game.escape_open

        game.step(ACTIONS[int(action)], CONTROL_DT)
        self.episode_steps += 1
        terminated = game.status in ("won", "lost")
        truncated = self.episode_steps >= self.max_steps and not terminated
        self._needs_reset = terminated or truncated

        # Use actual simulated time: a win or collision can end an action early.
        delta = game.elapsed - elapsed_before
        locked_time = max(0.0, min(game.elapsed, ESCAPE_UNLOCK) - min(elapsed_before, ESCAPE_UNLOCK))
        terms = {
            "survival": 0.05 * locked_time,
            "time": -0.05 * max(0.0, delta - locked_time),
            # Signed progress cancels over an undiscounted round trip. This
            # is heuristic shaping, not a guarantee of policy invariance.
            # Apply only after unlock; the unlock event itself awards nothing.
            "progress": 0.002 * (distance_before - self._goal_distance()) if escape_was_open else 0.0,
            "damage": -0.1 * max(0.0, health_before - game.health),
            "capture": -0.5 * max(0.0, game.capture - capture_before),
            "terminal": 20.0 if game.status == "won" else (-10.0 if game.status == "lost" else 0.0),
        }
        reward = float(sum(terms.values()))
        return self._get_obs(), reward, terminated, truncated, self._get_info(terms)

    def _goal_distance(self):
        return math.hypot(self.game.escape[0] - self.game.player.x,
                          self.game.escape[1] - self.game.player.y)

    def _get_info(self, reward_terms=None):
        game = self.game
        return {
            "env_version": 2,
            "start_offset": self.start_offset,
            "capture_state": game.capture_state,
            "elapsed": game.elapsed,
            "health": game.health,
            "nitro": game.nitro,
            "capture": game.capture,
            "status": game.status,
            "reason": game.reason,
            "success": game.status == "won",
            "episode_steps": self.episode_steps,
            "episode_seed": self.episode_seed,
            "distance_to_safehouse": self._goal_distance(),
            "reward_terms": dict(reward_terms) if reward_terms is not None else dict.fromkeys(REWARD_TERMS, 0.0),
        }

    def _get_obs(self):
        game, p = self.game, self.game.player
        c, s = math.cos(p.angle), math.sin(p.angle)

        def local(x, y):
            # Positive right follows the game's clockwise, screen-Y-down axes.
            return x * c + y * s, -x * s + y * c

        forward, right = local(p.vx, p.vy)
        goal_forward, goal_right = local(game.escape[0] - p.x, game.escape[1] - p.y)
        values = [
            2 * p.x / WORLD_SIZE - 1, 2 * p.y / WORLD_SIZE - 1,
            c, s, forward / VELOCITY_SCALE, right / VELOCITY_SCALE,
            p.yaw_rate / 3.0, p._steering, p._traction_loss, float(p._reverse_gear),
            game.health / 100, game.nitro / 100, game.capture,
            game._nitro_cooldown / 1.25, float(game._nitro_empty), game._damage_cooldown / 0.65,
            goal_forward / WORLD_DIAGONAL, goal_right / WORLD_DIAGONAL,
            float(game.escape_open), max(0.0, ESCAPE_UNLOCK - game.elapsed) / ESCAPE_UNLOCK,
            max(0.0, 1 - self.episode_steps / self.max_steps),
            (min(ROAD_CENTERS, key=lambda x: abs(x - p.x)) - p.x) / 210,
            (min(ROAD_CENTERS, key=lambda y: abs(y - p.y)) - p.y) / 210,
        ]
        # Stable patrol order avoids changing an identity when two cars cross.
        for i in range(MAX_COPS):
            if i >= len(game.cops):
                values.extend([0.0] * 7)
                continue
            cop = game.cops[i]
            dx, dy = local(cop.x - p.x, cop.y - p.y)
            vx, vy = local(cop.vx - p.vx, cop.vy - p.vy)
            relative_angle = cop.angle - p.angle
            values.extend((1.0, dx / WORLD_DIAGONAL, dy / WORLD_DIAGONAL,
                           vx / VELOCITY_SCALE, vy / VELOCITY_SCALE,
                           math.cos(relative_angle), math.sin(relative_angle)))
        values.extend(self._wall_distances())
        # Clipping keeps collision impulses within the declared input bounds.
        # Construct a fresh array every time; rollout buffers retain old frames.
        return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)

    def _wall_distances(self):
        """Normalized clearance rays from the car centre, starting forward.

        Buildings are expanded by the collision radius, world edges inset by
        it. Rectangle expansion is conservative at corners. Cars are handled
        by separate observation features, not by these static-obstacle rays.
        """
        p = self.game.player
        rectangles = [(b.x - CAR_RADIUS, b.y - CAR_RADIUS,
                       b.x + b.w + CAR_RADIUS, b.y + b.h + CAR_RADIUS)
                      for b in self.game.buildings]
        result = []
        for i in range(RAY_COUNT):
            angle = p.angle + i * math.tau / RAY_COUNT
            dx, dy = math.cos(angle), math.sin(angle)
            distance = RAY_RANGE
            for pos, direction in ((p.x, dx), (p.y, dy)):
                if abs(direction) > 1e-10:
                    boundary = WORLD_SIZE - CAR_RADIUS if direction > 0 else CAR_RADIUS
                    distance = min(distance, max(0.0, (boundary - pos) / direction))
            for left, top, right, bottom in rectangles:
                enter, leave = 0.0, distance
                for pos, direction, low, high in ((p.x, dx, left, right), (p.y, dy, top, bottom)):
                    if abs(direction) < 1e-10:
                        if pos < low or pos > high:
                            enter, leave = 1.0, -1.0
                            break
                    else:
                        a, b = (low - pos) / direction, (high - pos) / direction
                        enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
                    if enter > leave:
                        break
                if enter <= leave:
                    distance = min(distance, enter)
            result.append(distance / RAY_RANGE)
        return result

    def render(self):
        if self.render_mode is None:
            return None
        if self._needs_reset and self.episode_seed is None:
            raise gym.error.ResetNeeded("Call reset() before rendering")
        game = self.game
        return (f"Night Shift V2 | {game.status} | time={game.elapsed:.1f}s | "
                f"speed={game.player.speed * .32:.0f}km/h | health={game.health:.1f}% | "
                f"nitro={game.nitro:.1f}% | safehouse={'open' if game.escape_open else 'locked'}")

    def close(self):
        self._needs_reset = True


if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="gym_env:NightShiftEnv")
