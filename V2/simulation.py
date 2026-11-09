"""Deterministic arcade driving and pursuit, independent of graphics.

Coordinates are pixels, seconds and radians; positive angles turn clockwise.
Keeping this module free of Pygame makes it usable by a later RL environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from pursuit import PursuitDirector
from handling import drive_player


WORLD_SIZE = 2880
ROAD_WIDTH = 144
ROAD_CENTERS = [180 + 420 * i for i in range(7)]
ESCAPE_UNLOCK = 45.0
CAR_RADIUS = 17.0
NORMAL_SPEED = 330.0
NITRO_SPEED = 470.0
CAPTURE_DISTANCE = 72.0
CAPTURE_SPEED = 65.0
CAPTURE_SECONDS = 3.2
CAPTURE_RELEASE_DISTANCE = 120.0
CAPTURE_DECAY = 0.65


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def angle_difference(target: float, current: float) -> float:
    return (target - current + math.pi) % math.tau - math.pi


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


@dataclass
class Controls:
    throttle: float = 0.0
    steer: float = 0.0
    brake: bool = False
    nitro: bool = False


@dataclass
class Car:
    x: float
    y: float
    angle: float
    vx: float = 0.0
    vy: float = 0.0
    police: bool = False
    drifting: bool = False
    boosting: bool = False
    # Private navigation state is part of the deterministic simulation too.
    _route: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _route_age: float = 0.0
    _stuck: float = 0.0
    _reversing: float = 0.0
    _drift_timer: float = 0.0
    _reverse_gear: bool = False
    yaw_rate: float = 0.0
    _steering: float = 0.0
    _traction_loss: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


class Game:
    """One medium-difficulty chase. Call ``step(Controls(...), seconds)``.

    No wall clock, global RNG, drawing, input or filesystem calls occur here.
    A fixed seed and identical action/timestep sequence reproduce a run.
    """

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self.buildings = [
            Rect(x + 90, y + 90, 240, 240)
            for x in ROAD_CENTERS[:-1]
            for y in ROAD_CENTERS[:-1]
        ]
        self.player = Car(1440.0, 1860.0, -math.pi / 2)
        self.cops = [
            Car(1440.0, 2350.0, -math.pi / 2, police=True),
            Car(1020.0, 2280.0, -math.pi / 2, police=True),
            Car(1860.0, 2280.0, -math.pi / 2, police=True),
        ]
        self.elapsed = 0.0
        self.health = 100.0
        self.nitro = 100.0
        self.capture = 0.0
        self.status = "playing"
        self.reason = ""
        self.escape = (180.0, 180.0)
        self.drift_score = 0.0
        self.distance = 0.0
        self.last_impact = 0.0
        self._damage_cooldown = 0.0
        self._nitro_cooldown = 0.0
        self._nitro_empty = False
        self._police_bias = [self._rng.randrange(4) for _ in self.cops]
        self._pursuit = PursuitDirector(self, Controls, ROAD_CENTERS, WORLD_SIZE)

    @property
    def escape_open(self) -> bool:
        return self.elapsed >= ESCAPE_UNLOCK

    @property
    def capture_state(self) -> str:
        nearest = min((math.hypot(c.x-self.player.x,c.y-self.player.y)
                       for c in self.cops),default=math.inf)
        if nearest < CAPTURE_DISTANCE and self.player.speed < CAPTURE_SPEED:
            return "growing"
        if nearest >= CAPTURE_RELEASE_DISTANCE:
            return "releasing"
        return "held"

    def _update_capture(self,dt: float) -> None:
        # V2: speeding up beside a cop holds the meter; only distance clears it.
        state = self.capture_state
        if state == "growing":
            self.capture = min(1.0,self.capture+dt/CAPTURE_SECONDS)
        elif state == "releasing":
            self.capture = max(0.0,self.capture-dt*CAPTURE_DECAY)

    def step(self, controls: Controls, dt: float) -> None:
        """Advance by at most 0.1 s, subdivided to prevent collision tunneling.

        The UI pauses on focus loss rather than simulating a large time jump.
        Callers doing offline simulation should use a fixed timestep.
        """
        if self.status != "playing" or not math.isfinite(dt) or dt <= 0:
            return
        dt = min(dt, 0.1)
        count = max(1, math.ceil(dt / (1 / 120)))
        substep = dt / count
        for _ in range(count):
            if self.status != "playing":
                break
            self._step(controls, substep)

    def _step(self, controls: Controls, dt: float) -> None:
        self.elapsed += dt
        self._damage_cooldown = max(0.0, self._damage_cooldown - dt)
        self._nitro_cooldown = max(0.0, self._nitro_cooldown - dt)
        self.last_impact = max(0.0, self.last_impact - dt * 2.6)
        if not controls.nitro:
            self._nitro_empty = False
        boost = bool(
            controls.nitro and controls.throttle > 0 and self.nitro > 0
            and not self._nitro_empty
        )
        if boost:
            self.nitro = max(0.0, self.nitro - 32.0 * dt)
            self._nitro_cooldown = 1.25
            if self.nitro <= 0:
                self._nitro_empty = True
        elif self._nitro_cooldown <= 0:
            self.nitro = min(100.0, self.nitro + 11.0 * dt)
        self.player.boosting = boost

        old_x, old_y = self.player.x, self.player.y
        self._drive(self.player, controls, dt, NITRO_SPEED if boost else NORMAL_SPEED)
        self._resolve_scenery(self.player)

        self._pursuit.observe(dt)
        for index, cop in enumerate(self.cops):
            self._update_police(cop, index, dt)
            self._resolve_scenery(cop)
        for cop in self.cops:
            self._collide_cars(self.player, cop, hurt_player=True)
        for i, first in enumerate(self.cops):
            for second in self.cops[i + 1:]:
                self._collide_cars(first, second, hurt_player=False)
        # Vehicle contacts must never push a car through a building.
        self._resolve_scenery(self.player)
        for cop in self.cops:
            self._resolve_scenery(cop)

        self.distance += math.hypot(self.player.x - old_x, self.player.y - old_y)
        if self.player.drifting:
            self.drift_score += dt * self.player.speed * 0.08

        self._update_capture(dt)

        if self.health <= 0:
            self.status = "lost"
            self.reason = "Your car is wrecked"
        elif self.capture >= 1.0:
            self.status = "lost"
            self.reason = "The police boxed you in"
        elif self.escape_open and math.hypot(
            self.player.x - self.escape[0], self.player.y - self.escape[1]
        ) < 75:
            self.status = "won"
            self.reason = "You escaped the pursuit"

    def _drive(self, car: Car, controls: Controls, dt: float, top_speed: float) -> None:
        if not car.police:
            drive_player(car, controls, dt, top_speed, ROAD_CENTERS, ROAD_WIDTH)
            return
        throttle = clamp(controls.throttle, -1.0, 1.0)
        steer = clamp(controls.steer, -1.0, 1.0)
        speed = car.speed
        forward_before = car.vx * math.cos(car.angle) + car.vy * math.sin(car.angle)
        handbrake = bool(controls.brake and not car.police)
        # A sideways car must not suddenly invert its steering when the nose
        # crosses the velocity vector. Reverse steering belongs to reverse gear.
        if speed < 35:
            if throttle < -0.1:
                car._reverse_gear = True
            elif throttle > 0.1:
                car._reverse_gear = False
        elif throttle > 0 and forward_before > 25:
            car._reverse_gear = False
        direction = -1.0 if car._reverse_gear else 1.0
        steering_rate = min(speed / (75.0 if car.police else 85.0), 1.0) * (3.6 if car.police else 3.75)
        if handbrake:
            steering_rate *= 1.32
            # Keep the drift controllable near a broadside slide. Countersteer
            # retains full authority; holding into a slide doesn't flip the car.
            if speed > 60:
                slip = angle_difference(car.angle, math.atan2(car.vy, car.vx))
                if steer * slip > 0 and abs(slip) > 1.1:
                    steering_rate *= max(0.28, 1.0 - (abs(slip) - 1.1) / 0.65)
        # Steering responds earlier at low speed and turns a much tighter arc.
        car.angle = (car.angle + steer * steering_rate * direction * dt) % math.tau
        fx, fy = math.cos(car.angle), math.sin(car.angle)
        forward = car.vx * fx + car.vy * fy
        lateral = -car.vx * fy + car.vy * fx

        acceleration = 300.0 if car.police else (345.0 if car.boosting else 230.0)
        if throttle >= 0:
            forward += throttle * acceleration * dt
        elif forward > 8:
            forward += throttle * (430.0 if car.police else 330.0) * dt
        else:
            forward += throttle * (210.0 if car.police else 155.0) * dt
        forward *= math.exp(-0.42 * dt)
        if handbrake:
            forward *= math.exp(-0.18 * dt)
        if handbrake and speed > 60:
            car._drift_timer = 0.45
        else:
            car._drift_timer = max(0.0, car._drift_timer - dt)
        if car.police:
            grip = 12.5
        elif handbrake:
            grip = 0.75
        else:
            # Rear tyres regain grip progressively when Space is released.
            grip = 11.5 - 8.5 * (car._drift_timer / 0.45)
        lateral *= math.exp(-grip * dt)

        # Parks and verges are traversable but slower than paved roads.
        road_distance = min(
            min(abs(car.x - x) for x in ROAD_CENTERS),
            min(abs(car.y - y) for y in ROAD_CENTERS),
        )
        if road_distance > ROAD_WIDTH / 2 + 12:
            forward *= math.exp(-1.1 * dt)
            top_speed *= 0.67
        # Retain momentum for a moment after releasing boost.
        forward_limit = max(top_speed, forward_before - 220.0 * dt)
        forward = clamp(forward, -145.0 if car.police else -105.0, forward_limit)
        car.vx, car.vy = fx * forward - fy * lateral, fy * forward + fx * lateral
        total = car.speed
        # Decelerate smoothly when nitro ends instead of snapping velocity.
        allowed = max(top_speed, total - 220 * dt)
        if total > allowed and total > 0:
            car.vx *= allowed / total
            car.vy *= allowed / total
        car.drifting = bool((handbrake or car._drift_timer > 0) and speed > 70 and abs(lateral) > 24)
        car.x += car.vx * dt
        car.y += car.vy * dt

    def _damage(self, strength: float) -> None:
        if strength <= 48:
            return
        self.last_impact = max(self.last_impact, min(1.0, strength / 330.0))
        if self._damage_cooldown <= 0:
            self.health = max(0.0, self.health - clamp((strength - 35) * 0.062, 2, 22))
            self._damage_cooldown = 0.65

    def _resolve_scenery(self, car: Car) -> None:
        # Resolve each contact as a circle vs. rectangle. At 120 Hz even nitro
        # advances under 4 px, much less than the collision radius.
        r = CAR_RADIUS
        for rect in self.buildings:
            nx = clamp(car.x, rect.x, rect.x + rect.w)
            ny = clamp(car.y, rect.y, rect.y + rect.h)
            dx, dy = car.x - nx, car.y - ny
            dist_sq = dx * dx + dy * dy
            if dist_sq >= r * r:
                continue
            if dist_sq > 1e-10:
                dist = math.sqrt(dist_sq)
                normal_x, normal_y = dx / dist, dy / dist
                depth = r - dist
            else:
                sides = [
                    (car.x - rect.x, -1.0, 0.0),
                    (rect.x + rect.w - car.x, 1.0, 0.0),
                    (car.y - rect.y, 0.0, -1.0),
                    (rect.y + rect.h - car.y, 0.0, 1.0),
                ]
                inset, normal_x, normal_y = min(sides, key=lambda side: side[0])
                depth = r + inset
            car.x += normal_x * (depth + 0.01)
            car.y += normal_y * (depth + 0.01)
            incoming = car.vx * normal_x + car.vy * normal_y
            if incoming < 0:
                if car is self.player:
                    self._damage(-incoming)
                car.vx -= incoming * normal_x * 1.10
                car.vy -= incoming * normal_y * 1.10

        for axis in ("x", "y"):
            pos = getattr(car, axis)
            vel_name = "v" + axis
            vel = getattr(car, vel_name)
            if pos < r or pos > WORLD_SIZE - r:
                setattr(car, axis, clamp(pos, r, WORLD_SIZE - r))
                if (pos < r and vel < 0) or (pos > WORLD_SIZE - r and vel > 0):
                    if car is self.player:
                        self._damage(abs(vel))
                    setattr(car, vel_name, -vel * 0.12)

    def _collide_cars(self, first: Car, second: Car, hurt_player: bool) -> None:
        dx, dy = second.x - first.x, second.y - first.y
        dist = math.hypot(dx, dy)
        minimum = CAR_RADIUS * 2
        if dist >= minimum:
            return
        nx, ny = (dx / dist, dy / dist) if dist > 1e-8 else (1.0, 0.0)
        overlap = minimum - dist + 0.01
        first.x -= nx * overlap * 0.5
        first.y -= ny * overlap * 0.5
        second.x += nx * overlap * 0.5
        second.y += ny * overlap * 0.5
        closing = (first.vx - second.vx) * nx + (first.vy - second.vy) * ny
        if closing > 0:
            impulse = closing * 0.57
            first.vx -= impulse * nx
            first.vy -= impulse * ny
            second.vx += impulse * nx
            second.vy += impulse * ny
            if hurt_player:
                self._damage(closing * 0.72)

    def _clear_segment(self, x: float, y: float, tx: float, ty: float) -> bool:
        return self._pursuit.segment_clear(x, y, tx, ty)

    def _update_police(self, cop: Car, index: int, dt: float) -> None:
        self._pursuit.drive(cop, index, dt)
