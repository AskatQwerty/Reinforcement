"""Night Shift: a standalone arcade chase. Run: python main.py.

The presentation layer lives here; simulation.py is independent of Pygame.
All art and audio are generated locally; no downloads or assets are needed.
"""
from __future__ import annotations

import argparse
from array import array
from collections import deque
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame as pg
except ImportError:
    print("Install dependencies first: python -m pip install -r requirements.txt")
    raise SystemExit(1)

from simulation import Game, Controls, WORLD_SIZE, ROAD_CENTERS, ROAD_WIDTH, ESCAPE_UNLOCK

WIDTH, HEIGHT = 1280, 800
INK = (10, 17, 25)
WHITE = (255, 248, 238)
MUTED = (175, 167, 157)
TEAL = (80, 237, 205)
AMBER = (255, 191, 94)
RED = (255, 92, 107)
ORANGE = (255, 155, 76)
ORANGE_DEEP = (154, 58, 19)
PANEL_INK = (43, 31, 24)
PANEL_MUTED = (105, 70, 47)
PANEL_DANGER = (152, 29, 38)


def rounded(surface, color, rect, radius=12, width=0):
    pg.draw.rect(surface, color, pg.Rect(rect), width, border_radius=radius)


class SoundBank:
    """Quiet synthesized loops, optional on machines without an audio device."""

    def __init__(self):
        self.enabled = True
        self.available = False
        self.engine_index = -1
        try:
            if not pg.mixer.get_init():
                pg.mixer.init(22050, -16, 1, 512)
            self.engines = [self.tone("engine", 48 + i * 16, 0.5) for i in range(7)]
            self.siren = self.tone("siren", 600, 2.0)
            self.boost = self.tone("boost", 160, 1.0)
            self.hit = self.tone("hit", 90, 0.18)
            self.engine_channel = pg.mixer.Channel(0)
            self.siren_channel = pg.mixer.Channel(1)
            self.boost_channel = pg.mixer.Channel(2)
            self.hit_channel = pg.mixer.Channel(3)
            self.siren_channel.play(self.siren, loops=-1)
            self.boost_channel.play(self.boost, loops=-1)
            self.available = True
        except (pg.error, ValueError):
            pass

    def tone(self, kind, freq, seconds):
        rng = random.Random(44)
        rate, _, channels = pg.mixer.get_init()
        data = array("h")
        count = int(rate * seconds)
        for i in range(count):
            t = i / rate
            if kind == "engine":
                value = (math.sin(math.tau * freq * t) * .5
                         + math.sin(math.tau * freq * 2 * t) * .25
                         + math.sin(math.tau * freq * 3 * t) * .15)
            elif kind == "siren":
                phase = math.tau * 720 * t - 110 * math.cos(math.tau * t) / 1.0
                value = math.sin(phase) * .35
            elif kind == "hit":
                value = rng.uniform(-1, 1) * (1 - i / count) ** 2
            else:
                value = rng.uniform(-.2, .2) + math.sin(math.tau * freq * t) * .15
            sample = int(max(-1, min(1, value)) * 14000)
            for _ in range(channels):
                data.append(sample)
        return pg.mixer.Sound(buffer=data.tobytes())

    def update(self, game, active):
        if not self.available:
            return
        audible = self.enabled and active
        index = min(6, int(game.player.speed / 75))
        if index != self.engine_index:
            self.engine_channel.play(self.engines[index], loops=-1, fade_ms=70)
            self.engine_index = index
        self.engine_channel.set_volume(.16 if audible else 0)
        nearest = min((math.hypot(c.x - game.player.x, c.y - game.player.y)
                       for c in game.cops), default=1000)
        self.siren_channel.set_volume(max(0, .13 * (1 - nearest / 900)) if audible else 0)
        self.boost_channel.set_volume(.22 if audible and game.player.boosting else 0)

    def impact(self):
        if self.available and self.enabled:
            self.hit_channel.set_volume(.22)
            self.hit_channel.play(self.hit)


class Renderer:
    def __init__(self, screen, game):
        self.screen = screen
        self.frame = pg.Surface((WIDTH, HEIGHT)).convert()
        self.fonts = {}
        self.ui_cache = {}
        self.font_path = pg.font.match_font("segoeui,dejavusans,arial")
        self.bold_path = pg.font.match_font("segoeui,dejavusans,arial", bold=True)
        windows_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
        if windows_font.is_file():
            self.font_path = str(windows_font)
        self.rng = random.Random(29)
        self.map = self.build_city(game)
        self.mini = pg.transform.smoothscale(self.map, (180, 180))
        self.camera = [game.player.x, game.player.y]
        self.camera_angle = (game.player.angle + math.pi / 2) % math.tau
        self.zoom = .90
        self._camera_turn_rate = 0.0
        self._camera_impact_hold = 0.0
        self.skids = deque(maxlen=1100)
        self.particles = deque(maxlen=180)
        self.car_images = {False: self.car_sprite(False), True: self.car_sprite(True)}
        self.last_rear = None
        self.last_health = game.health
        self.buttons = {}
        self.veil = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA)
        self.light_layer = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA)
        # Only the static city is drawn at lower resolution during rotation;
        # vehicles, effects and interface retain the full window resolution.
        self.map_scale = .625
        self.map_frame = pg.Surface((int(WIDTH * self.map_scale),
                                     int(HEIGHT * self.map_scale))).convert()

    def text(self, text, xy, size=20, color=WHITE, bold=False, anchor="topleft", surface=None):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = pg.font.Font(self.bold_path if bold else self.font_path, size)
        label = self.fonts[key].render(str(text), True, color)
        rect = label.get_rect(**{anchor: xy})
        (surface or self.frame).blit(label, rect)
        return rect

    def build_city(self, game):
        city = pg.Surface((WORLD_SIZE, WORLD_SIZE)).convert()
        city.fill((22, 36, 43))
        rng = random.Random(81)
        for center in ROAD_CENTERS:
            pg.draw.rect(city, (39, 52, 61), (center - ROAD_WIDTH // 2 - 12, 0, ROAD_WIDTH + 24, WORLD_SIZE))
            pg.draw.rect(city, (39, 52, 61), (0, center - ROAD_WIDTH // 2 - 12, WORLD_SIZE, ROAD_WIDTH + 24))
        for center in ROAD_CENTERS:
            pg.draw.rect(city, (29, 40, 49), (center - ROAD_WIDTH // 2, 0, ROAD_WIDTH, WORLD_SIZE))
            pg.draw.rect(city, (29, 40, 49), (0, center - ROAD_WIDTH // 2, WORLD_SIZE, ROAD_WIDTH))
        for center in ROAD_CENTERS:
            for at in range(12, WORLD_SIZE, 48):
                if all(abs(at - cross) > ROAD_WIDTH * .64 for cross in ROAD_CENTERS):
                    pg.draw.line(city, (100, 100, 82), (center, at), (center, at + 23), 2)
                    pg.draw.line(city, (100, 100, 82), (at, center), (at + 23, center), 2)
            for cross in ROAD_CENTERS:
                for stripe in range(-48, 49, 16):
                    pg.draw.rect(city, (85, 98, 103), (center + stripe, cross - 84, 8, 16))
                    pg.draw.rect(city, (85, 98, 103), (center + stripe, cross + 68, 8, 16))
                for dx, dy in [(-84, -84), (84, 84)]:
                    pg.draw.circle(city, (44, 63, 65), (center + dx, cross + dy), 10)
                    pg.draw.circle(city, (152, 181, 161), (center + dx, cross + dy), 3)
        palettes = [(43, 60, 70), (47, 57, 73), (38, 58, 63), (53, 61, 72)]
        for index, b in enumerate(game.buildings):
            rect = pg.Rect(b.x, b.y, b.w, b.h)
            pg.draw.rect(city, (15, 24, 31), rect.move(10, 12), border_radius=7)
            pg.draw.rect(city, (78, 92, 96), rect.inflate(8, 8), border_radius=5)
            color = palettes[index % len(palettes)]
            pg.draw.rect(city, color, rect, border_radius=4)
            pg.draw.rect(city, (32, 44, 54), rect.inflate(-16, -16), 3, border_radius=3)
            for offset in range(22, int(b.w) - 18, 30):
                tint = (111, 138, 123) if rng.random() > .5 else (64, 86, 96)
                pg.draw.rect(city, tint, (b.x + offset, b.y + 2, 13, 3))
                pg.draw.rect(city, tint, (b.x + offset, b.y + b.h - 5, 13, 3))
            for _ in range(3):
                ax = int(rng.uniform(b.x + 22, b.x + b.w - 46))
                ay = int(rng.uniform(b.y + 22, b.y + b.h - 46))
                pg.draw.rect(city, (26, 39, 48), (ax + 3, ay + 4, 26, 22), border_radius=2)
                pg.draw.rect(city, (64, 79, 88), (ax, ay, 26, 22), border_radius=2)
                for line in range(4):
                    pg.draw.line(city, (39, 54, 63), (ax + 4, ay + 4 + line * 4), (ax + 21, ay + 4 + line * 4))
            # Rooftop landing marks and solar panels give districts distinct silhouettes.
            if index % 7 == 0 and b.w > 120 and b.h > 120:
                center = rect.center
                pg.draw.circle(city, (93, 108, 105), center, 39, 2)
                pg.draw.line(city, (93, 108, 105), (center[0]-13, center[1]-20), (center[0]-13, center[1]+20), 3)
                pg.draw.line(city, (93, 108, 105), (center[0]+13, center[1]-20), (center[0]+13, center[1]+20), 3)
                pg.draw.line(city, (93, 108, 105), (center[0]-13, center[1]), (center[0]+13, center[1]), 3)
            elif index % 3 == 0:
                for col in range(3):
                    pg.draw.rect(city, (24, 48, 65), (b.x + 25 + col * 34, b.y + 30, 28, 46))
                    pg.draw.rect(city, (71, 104, 121), (b.x + 25 + col * 34, b.y + 30, 28, 46), 1)
        # Small trees stay on the sidewalk, outside the driveable road.
        for cx in ROAD_CENTERS:
            for cy in ROAD_CENTERS:
                for dx, dy in [(-100, 112), (110, -104)]:
                    x, y = cx + dx, cy + dy
                    pg.draw.circle(city, (12, 29, 34), (x + 4, y + 5), 16)
                    pg.draw.circle(city, (33, 69, 62), (x, y), 15)
                    pg.draw.circle(city, (43, 85, 71), (x - 3, y - 4), 9)
        return city

    def car_sprite(self, police):
        car = pg.Surface((72, 46), pg.SRCALPHA)
        for x in (15, 48):
            rounded(car, (7, 12, 18), (x, 5, 13, 8), 2)
            rounded(car, (7, 12, 18), (x, 33, 13, 8), 2)
        body = (216, 228, 230) if police else (255, 142, 54)
        outline = (122, 150, 166) if police else (171, 76, 22)
        pg.draw.polygon(car, outline, [(9, 12), (17, 8), (54, 8), (64, 14), (66, 31), (55, 38), (17, 38), (9, 32)])
        pg.draw.polygon(car, body, [(12, 14), (19, 10), (52, 10), (62, 16), (63, 29), (53, 35), (19, 35), (12, 31)])
        rounded(car, (22, 38, 48), (26, 12, 23, 22), 4)
        pg.draw.polygon(car, (69, 104, 119), [(42, 13), (48, 15), (50, 31), (42, 33)])
        pg.draw.polygon(car, (51, 83, 96), [(26, 14), (29, 14), (29, 32), (24, 30)])
        rounded(car, (27, 44, 56) if police else (218, 105, 38), (31, 14, 10, 18), 2)
        pg.draw.line(car, (255, 211, 151) if not police else (245, 249, 250), (53, 15), (59, 18), 2)
        pg.draw.rect(car, (255, 241, 186), (61, 13, 4, 7), border_radius=1)
        pg.draw.rect(car, (255, 241, 186), (61, 26, 4, 7), border_radius=1)
        pg.draw.rect(car, (255, 74, 89), (10, 13, 3, 7))
        pg.draw.rect(car, (255, 74, 89), (10, 27, 3, 7))
        if police:
            pg.draw.rect(car, (22, 40, 57), (16, 11, 7, 24))
            pg.draw.rect(car, (37, 113, 249), (33, 11, 5, 11))
            pg.draw.rect(car, (255, 70, 88), (33, 23, 5, 11))
        else:
            pg.draw.line(car, (70, 40, 26), (14, 9), (14, 37), 4)
            pg.draw.line(car, (255, 224, 176), (53, 20), (61, 20), 1)
        return car

    def reset(self, game):
        self.camera[:] = [game.player.x, game.player.y]
        self.camera_angle = (game.player.angle + math.pi / 2) % math.tau
        self.zoom = .90
        self._camera_turn_rate = 0.0
        self._camera_impact_hold = 0.0
        self.skids.clear()
        self.particles.clear()
        self.last_rear = None
        self.last_health = game.health

    def world(self, x, y):
        dx, dy = x - self.camera[0], y - self.camera[1]
        c, s = math.cos(self.camera_angle), math.sin(self.camera_angle)
        return ((dx * c + dy * s) * self.zoom + WIDTH / 2,
                (-dx * s + dy * c) * self.zoom + HEIGHT / 2 + 50)

    def draw_map(self):
        """Rotate only the visible city region; cover every viewport corner.

        The inverse projection determines the source rectangle. Its actual
        pixel centre is projected back so fractional camera positions cannot
        make the streets drift away from vehicles, lights or tyre marks.
        """
        c, s = math.cos(self.camera_angle), math.sin(self.camera_angle)
        corners = []
        for sx, sy in ((0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)):
            dx = (sx - WIDTH / 2) / self.zoom
            dy = (sy - (HEIGHT / 2 + 50)) / self.zoom
            corners.append((self.camera[0] + dx * c - dy * s,
                            self.camera[1] + dx * s + dy * c))
        left = math.floor(min(p[0] for p in corners)) - 4
        top = math.floor(min(p[1] for p in corners)) - 4
        right = math.ceil(max(p[0] for p in corners)) + 4
        bottom = math.ceil(max(p[1] for p in corners)) + 4
        source = pg.Rect(left, top, right - left, bottom - top)
        if self.map.get_rect().contains(source):
            crop = self.map.subsurface(source)
        else:
            crop = pg.Surface(source.size)
            crop.fill((14, 27, 35))
            crop.blit(self.map, (-left, -top))
        rotated = pg.transform.rotozoom(crop, math.degrees(self.camera_angle),
                                       self.zoom * self.map_scale)
        centre = self.world(left + source.w / 2, top + source.h / 2)
        self.map_frame.fill((14, 27, 35))
        self.map_frame.blit(rotated, rotated.get_rect(
            center=(centre[0] * self.map_scale, centre[1] * self.map_scale)))
        pg.transform.smoothscale(self.map_frame, (WIDTH, HEIGHT), self.frame)

    def effects(self, game, dt, active):
        p = game.player
        if active and p.drifting and p.speed > 85:
            c, s = math.cos(p.angle), math.sin(p.angle)
            rear = [(p.x - c * 20 - s * offset, p.y - s * 20 + c * offset) for offset in (-13, 13)]
            if self.last_rear:
                for start, end in zip(self.last_rear, rear):
                    self.skids.append((start, end))
            self.last_rear = rear
            if self.rng.random() < .65:
                self.particles.append([rear[0][0], rear[0][1], self.rng.uniform(-16, 16), self.rng.uniform(-16, 16), .6, .6, "smoke"])
        else:
            self.last_rear = None
        if active and p.boosting:
            self.particles.append([p.x - math.cos(p.angle) * 29, p.y - math.sin(p.angle) * 29,
                                   -math.cos(p.angle) * 65, -math.sin(p.angle) * 65, .22, .22, "boost"])
        if game.health < self.last_health:
            for _ in range(14):
                self.particles.append([p.x, p.y, self.rng.uniform(-150, 150), self.rng.uniform(-150, 150), .35, .35, "spark"])
        self.last_health = game.health
        if active:
            remaining = deque(maxlen=180)
            for v in self.particles:
                v[0] += v[2] * dt
                v[1] += v[3] * dt
                v[4] -= dt
                if v[4] > 0:
                    remaining.append(v)
            self.particles = remaining

    def scene(self, game, dt, active):
        p = game.player
        if active:
            # Position follows the car without velocity look-ahead: a sudden
            # stop or impact cannot pull the camera backwards or change scale.
            blend = 1 - math.exp(-dt * 4)
            for i, coordinate in enumerate((p.x, p.y)):
                self.camera[i] += (coordinate - self.camera[i]) * blend
            self._camera_impact_hold = max(0.0, self._camera_impact_hold - dt)
            if game.health < self.last_health:
                self._camera_impact_hold = .65
            forwards = p.vx * math.cos(p.angle) + p.vy * math.sin(p.angle)
            if p.speed > 70 and forwards > 35 and self._camera_impact_hold <= 0:
                # Follow the travel direction rather than every body twitch
                # in a drift. A dead zone filters out small steering movements.
                target_angle = math.atan2(p.vy, p.vx) + math.pi / 2
                error = (target_angle - self.camera_angle + math.pi) % math.tau - math.pi
                remaining = max(0.0, abs(error) - .10)
                desired_rate = math.copysign(min(1.35, remaining * 2.2), error)
                self._camera_turn_rate += (desired_rate - self._camera_turn_rate) * (1 - math.exp(-dt * 6))
                rotation = self._camera_turn_rate * dt
                if rotation * error < 0 or remaining == 0:
                    rotation = 0.0
                else:
                    rotation = math.copysign(min(abs(rotation), remaining), rotation)
                self.camera_angle = (self.camera_angle + rotation) % math.tau
            else:
                self._camera_turn_rate = 0.0
        else:
            self._camera_turn_rate = 0.0
        self.effects(game, dt, active)
        self.draw_map()
        for start, end in self.skids:
            a, b = self.world(*start), self.world(*end)
            if -50 < b[0] < WIDTH + 50 and -50 < b[1] < HEIGHT + 50:
                pg.draw.line(self.frame, (15, 25, 31), a, b, max(2, int(self.zoom * 4)))
        ex, ey = self.world(*game.escape)
        pulse = 1 + .08 * math.sin(pg.time.get_ticks() / 180)
        escape_color = TEAL if game.escape_open else (149, 148, 111)
        pg.draw.circle(self.frame, escape_color, (int(ex), int(ey)), int(60 * pulse * self.zoom), 3)
        if -100 < ex < WIDTH + 100 and -100 < ey < HEIGHT + 100:
            self.text("SAFEHOUSE" if game.escape_open else "LOCKED", (ex, ey - 86 * self.zoom), 15, escape_color, True, "midtop")
        self.light_layer.fill((0, 0, 0, 0))
        for car in [*game.cops, p]:
            x, y = self.world(car.x, car.y)
            if not (-160 < x < WIDTH + 160 and -160 < y < HEIGHT + 160):
                continue
            relative_angle = car.angle - self.camera_angle
            c, s = math.cos(relative_angle), math.sin(relative_angle)
            cone = []
            for forward, side in [(25, -10), (135, -44), (145, 44), (25, 10)]:
                cone.append((x + (c * forward - s * side) * self.zoom, y + (s * forward + c * side) * self.zoom))
            pg.draw.polygon(self.light_layer, (229, 238, 185, 17), cone)
            if car.police:
                flash = (pg.time.get_ticks() // 130) % 2
                color = (57, 115, 255, 32) if flash else (255, 60, 81, 32)
                pg.draw.circle(self.light_layer, color, (int(x), int(y)), int(60 * self.zoom))
        self.frame.blit(self.light_layer, (0, 0))
        for car in [*game.cops, p]:
            x, y = self.world(car.x, car.y)
            if not (-100 < x < WIDTH + 100 and -100 < y < HEIGHT + 100):
                continue
            pg.draw.ellipse(self.frame, (13, 23, 29), (x - 29 * self.zoom + 5, y - 18 * self.zoom + 8, 58 * self.zoom, 36 * self.zoom))
            sprite = pg.transform.rotozoom(self.car_images[car.police],
                                          -math.degrees(car.angle - self.camera_angle), self.zoom)
            self.frame.blit(sprite, sprite.get_rect(center=(x, y)))
            if car.police:
                color = (67, 152, 255) if (pg.time.get_ticks() // 130) % 2 else (255, 100, 111)
                pg.draw.circle(self.frame, color, (int(x), int(y)), max(2, int(3 * self.zoom)))
        self.light_layer.fill((0, 0, 0, 0))
        for x, y, _, _, life, total, kind in self.particles:
            sx, sy = self.world(x, y)
            ratio = life / total
            if kind == "smoke":
                color, radius = (155, 178, 184, int(60 * ratio)), (1 - ratio) * 12 + 5
            elif kind == "boost":
                color, radius = (74, 209, 255, int(230 * ratio)), 5 * ratio + 3
            else:
                color, radius = (255, 194, 105, int(240 * ratio)), 2
            pg.draw.circle(self.light_layer, color, (int(sx), int(sy)), max(1, int(radius * self.zoom)))
        self.frame.blit(self.light_layer, (0, 0))

    def gradient_surface(self, size, first, last, radius=13, alpha=250):
        """Cached warm gradient; panels add no per-frame pixel loops."""
        w, h = int(size[0]), int(size[1])
        key = (w, h, first, last, radius, alpha)
        if key not in self.ui_cache:
            layer = pg.Surface((w, h), pg.SRCALPHA)
            for x in range(w):
                t = (x / max(1, w - 1)) ** 1.35
                color = tuple(round(a + (b - a) * t) for a, b in zip(first, last))
                pg.draw.line(layer, (*color, alpha), (x, 0), (x, h - 1))
            if radius:
                mask = pg.Surface((w, h), pg.SRCALPHA)
                rounded(mask, (255, 255, 255, 255), (0, 0, w, h), radius)
                layer.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
            self.ui_cache[key] = layer
        return self.ui_cache[key]

    def gradient_text(self, text, xy, size=84):
        key = (size, True)
        if key not in self.fonts:
            self.fonts[key] = pg.font.Font(self.bold_path, size)
        mask = self.fonts[key].render(text, True, (255, 255, 255))
        label = self.gradient_surface(mask.get_size(), WHITE, ORANGE, 0, 255).copy()
        label.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        self.frame.blit(label, xy)

    def panel(self, rect, alpha=250):
        x, y, w, h = rect
        layer = self.gradient_surface((w, h), (255, 250, 243), (255, 181, 112), 13, alpha)
        self.frame.blit(layer, (x, y))
        rounded(self.frame, (248, 218, 186), rect, 13, 1)

    def bar(self, x, y, w, value, color):
        rounded(self.frame, (215, 160, 114), (x, y, w, 6), 3)
        if value > 0:
            length = max(3, int(w * min(1, max(0, value))))
            fill = self.gradient_surface((length, 6), color, ORANGE, 3, 255)
            self.frame.blit(fill, (x, y))

    def minimap(self, game):
        x, y = WIDTH - 234, 24
        self.panel((x, y, 210, 245))
        self.text("RADAR", (x + 15, y + 12), 13, PANEL_INK, True)
        self.text("N ↑", (x + 195, y + 12), 13, ORANGE_DEEP, True, "topright")
        mx, my = x + 15, y + 39
        self.frame.blit(self.mini, (mx, my))
        scale = 180 / WORLD_SIZE
        for c in game.cops:
            pg.draw.circle(self.frame, RED, (int(mx + c.x * scale), int(my + c.y * scale)), 3)
        ex, ey = int(mx + game.escape[0] * scale), int(my + game.escape[1] * scale)
        pg.draw.rect(self.frame, TEAL if game.escape_open else AMBER, (ex - 4, ey - 4, 8, 8), 2)
        p = game.player
        px, py = mx + p.x * scale, my + p.y * scale
        triangle = [(px + math.cos(p.angle + a) * r, py + math.sin(p.angle + a) * r)
                    for a, r in [(0, 6), (2.5, 5), (-2.5, 5)]]
        pg.draw.polygon(self.frame, WHITE, triangle)
        self.text("● Police", (x + 15, y + 222), 11, PANEL_DANGER)
        self.text("□ Safehouse", (x + 109, y + 222), 11, (29, 91, 64) if game.escape_open else PANEL_MUTED)

    def hud(self, game, muted=False):
        self.panel((24, 24, 470, 112))
        self.text("NIGHT SHIFT V2", (44, 38), 15, ORANGE_DEEP, True)
        self.text("POLICE PURSUIT", (473, 39), 11, PANEL_MUTED, False, "topright")
        title = "REACH THE SAFEHOUSE" if game.escape_open else "SURVIVE FOR 45 SECONDS"
        self.text(title, (44, 66), 23, PANEL_INK, True)
        if game.escape_open:
            dist = int(math.hypot(game.player.x-game.escape[0], game.player.y-game.escape[1]))
            sub = f"Green marker on the radar  /  {dist} m"
        else:
            sub = f"Safehouse opens in {max(0, math.ceil(ESCAPE_UNLOCK-game.elapsed)):02d} s"
        self.text(sub, (44, 101), 14, PANEL_MUTED)
        self.minimap(game)
        self.panel((24, HEIGHT - 136, 342, 112))
        self.text(f"{int(game.player.speed * .32):03d}", (44, HEIGHT - 124), 46, PANEL_INK, True)
        self.text("KM/H", (151, HEIGHT - 89), 12, PANEL_MUTED)
        pg.draw.line(self.frame, (224, 172, 126), (182, HEIGHT - 115), (182, HEIGHT - 47))
        self.text("CONDITION", (201, HEIGHT - 117), 11, PANEL_MUTED, True)
        self.text(f"{math.ceil(game.health)}%", (345, HEIGHT - 119), 13, PANEL_INK, False, "topright")
        self.bar(201, HEIGHT - 94, 144, game.health / 100, ORANGE_DEEP if game.health > 35 else PANEL_DANGER)
        self.text("NITRO  /  SHIFT", (201, HEIGHT - 79), 11, PANEL_MUTED, True)
        self.bar(201, HEIGHT - 55, 144, game.nitro / 100, (174, 80, 29))
        self.text(f"CHASE  {int(game.elapsed)//60:02d}:{int(game.elapsed)%60:02d}", (44, HEIGHT - 53), 12, PANEL_MUTED)
        self.panel((WIDTH - 270, HEIGHT - 100, 246, 76))
        self.text("DRIFT", (WIDTH - 250, HEIGHT - 88), 12, ORANGE_DEEP if game.player.drifting else PANEL_MUTED, True)
        self.text(f"{int(game.drift_score):,}".replace(",", " "), (WIDTH - 44, HEIGHT - 78), 31, PANEL_INK, True, "topright")
        self.text("POINTS", (WIDTH - 250, HEIGHT - 57), 11, PANEL_MUTED)
        self.text("WASD / ARROWS  ·  SPACE drift  ·  SHIFT nitro", (WIDTH / 2, HEIGHT - 41), 13, MUTED, False, "midbottom")
        self.text("ESC pause  ·  R restart  ·  M " + ("sound off" if muted else "sound on"), (WIDTH / 2, HEIGHT - 21), 11, MUTED, False, "midbottom")
        if game.capture > .03:
            self.panel((WIDTH / 2 - 180, 155, 360, 83))
            label = "LOSING THE POLICE" if game.capture_state == "releasing" else "BREAK AWAY FROM POLICE"
            self.text(label, (WIDTH / 2, 165), 15, PANEL_DANGER, True, "midtop")
            self.text("Create distance to reduce capture", (WIDTH / 2, 190), 12, PANEL_MUTED, False, "midtop")
            self.bar(WIDTH / 2 - 160, 217, 320, game.capture, PANEL_DANGER)
        if game.player.boosting:
            self.text("NITRO", (WIDTH / 2, HEIGHT / 2 + 108), 16, ORANGE, True, "midtop")
        elif game.player.drifting:
            self.text("DRIFT", (WIDTH / 2, HEIGHT / 2 + 108), 16, ORANGE, True, "midtop")
        if game.escape_open:
            dx, dy = game.escape[0] - game.player.x, game.escape[1] - game.player.y
            if math.hypot(dx, dy) > 330:
                a = math.atan2(dy, dx) - self.camera_angle
                px, py = WIDTH/2 + math.cos(a)*150, HEIGHT/2+50+math.sin(a)*150
                points = [(px + math.cos(a+off)*length, py + math.sin(a+off)*length)
                          for off, length in [(0, 13), (2.45, 10), (-2.45, 10)]]
                pg.draw.polygon(self.frame, TEAL, points)

    def button(self, name, label, rect, primary=True):
        self.buttons[name] = pg.Rect(rect)
        wx, wy = self.screen.get_size()
        mx, my = pg.mouse.get_pos()
        scale = min(wx / WIDTH, wy / HEIGHT)
        mx, my = (mx - (wx-WIDTH*scale)/2)/scale, (my - (wy-HEIGHT*scale)/2)/scale
        hovered = self.buttons[name].collidepoint(mx, my)
        if primary:
            first, last = ((255, 251, 241), (255, 176, 90)) if hovered else ((255, 235, 207), (255, 139, 49))
        else:
            first, last = ((255, 251, 245), (249, 211, 170)) if hovered else ((251, 238, 219), (242, 193, 144))
        button = self.gradient_surface((rect[2], rect[3]), first, last, 8, 255)
        self.frame.blit(button, (rect[0], rect[1]))
        rounded(self.frame, (202, 128, 70), rect, 8, 1)
        self.text(label, self.buttons[name].center, 18, PANEL_INK, True, "center")

    def overlay(self, game, mode):
        self.buttons.clear()
        self.veil.fill((5, 12, 20, 158 if mode == "menu" else 190))
        self.frame.blit(self.veil, (0, 0))
        if mode == "menu":
            x, y = 80, 130
            self.text("V2  /  NIGHT CITY", (x, y), 14, ORANGE, True)
            self.text("NIGHT", (x - 4, y + 34), 84, WHITE, True)
            self.gradient_text("SHIFT", (x - 4, y + 120), 84)
            self.text("The streets are yours. Until they catch you.", (x, y + 230), 22, WHITE)
            self.text("Survive 45 seconds. Then make it to the safehouse.", (x, y + 269), 17, MUTED)
            self.text("Build distance. Speed alone will not clear capture.", (x, y + 297), 17, MUTED)
            self.button("start", "START CHASE · ENTER", (x, y + 356, 350, 58))
            self.text("ONE CITY  /  ONE MODE  /  ONE WAY OUT", (x, y + 434), 11, MUTED)
            self.panel((850, 390, 350, 286))
            self.text("BEHIND THE WHEEL", (876, 414), 14, ORANGE_DEEP, True)
            for i, (key, desc) in enumerate([("WASD / ARROWS", "Accelerate, brake, steer"),
                                              ("SPACE", "Handbrake · start a drift"),
                                              ("SHIFT", "Nitro · recharges over time"),
                                              ("ESC", "Pause")]):
                yy = 453 + i*48
                self.text(key, (876, yy), 13, PANEL_INK, True)
                self.text(desc, (876, yy + 19), 13, PANEL_MUTED)
        else:
            rect = (WIDTH/2-270, 215, 540, 380)
            self.panel(rect, 248)
            if mode == "paused":
                label, color, detail = "PAUSED", PANEL_INK, "The city can wait."
            elif game.status == "won":
                label, color, detail = "YOU GOT AWAY", ORANGE_DEEP, "You made it to the safehouse."
            else:
                label, color = "BUSTED", PANEL_DANGER
                detail = game.reason or "Try another route next time."
            self.text("NIGHT SHIFT V2", (WIDTH/2, 245), 13, PANEL_MUTED, True, "midtop")
            self.text(label, (WIDTH/2, 283), 31, color, True, "midtop")
            self.text(detail, (WIDTH/2, 334), 16, PANEL_MUTED, False, "midtop")
            self.text(f"Time {game.elapsed:.1f} s     /     Drift {int(game.drift_score)}", (WIDTH/2, 380), 17, PANEL_INK, False, "midtop")
            if mode == "paused":
                self.button("resume", "RESUME · ESC", (WIDTH/2-220, 434, 440, 50))
            else:
                self.button("start", "CHASE AGAIN · ENTER", (WIDTH/2-220, 434, 440, 50))
            self.button("menu", "MAIN MENU", (WIDTH/2-220, 503, 210, 46), False)
            self.button("quit", "QUIT", (WIDTH/2+10, 503, 210, 46), False)

    def present(self):
        w, h = self.screen.get_size()
        scale = min(w / WIDTH, h / HEIGHT)
        size = (int(WIDTH * scale), int(HEIGHT * scale))
        self.screen.fill((5, 10, 15))
        surface = self.frame if size == (WIDTH, HEIGHT) else pg.transform.smoothscale(self.frame, size)
        self.screen.blit(surface, ((w-size[0])//2, (h-size[1])//2))
        pg.display.flip()

    def clicked(self, pos):
        w, h = self.screen.get_size()
        scale = min(w / WIDTH, h / HEIGHT)
        point = ((pos[0]-(w-WIDTH*scale)/2)/scale, (pos[1]-(h-HEIGHT*scale)/2)/scale)
        return next((name for name, rect in self.buttons.items() if rect.collidepoint(point)), None)


def main():
    parser = argparse.ArgumentParser(description="Night Shift — top-down police chase")
    parser.add_argument("--smoke-test", action="store_true", help="Run a finite headless rendering check")
    parser.add_argument("--screenshot", type=Path, help="Save a PNG at the end of the smoke check")
    args = parser.parse_args()
    if args.smoke_test:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pg.mixer.pre_init(22050, -16, 1, 512)
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT), pg.RESIZABLE)
    pg.display.set_caption("Night Shift V2 | Police Chase")
    game = Game(seed=7)
    renderer = Renderer(screen, game)
    pg.display.set_icon(renderer.car_images[False])
    sound = SoundBank()
    clock = pg.time.Clock()
    mode = "playing" if args.smoke_test else "menu"
    countdown = 0.0
    accumulator = 0.0
    frames = 0
    running = True

    def begin():
        nonlocal mode, countdown, accumulator
        game.reset()
        renderer.reset(game)
        mode, countdown, accumulator = "playing", 2.6, 0.0

    while running:
        dt = min(clock.tick(60) / 1000, .05) if not args.smoke_test else 1/60
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.WINDOWFOCUSLOST and mode == "playing":
                mode = "paused"
                accumulator = 0.0
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    if mode == "playing":
                        mode = "paused"
                    elif mode == "paused":
                        mode = "playing"
                    elif mode == "end":
                        mode = "menu"
                    accumulator = 0.0
                elif event.key in (pg.K_RETURN, pg.K_KP_ENTER) and mode in ("menu", "end"):
                    begin()
                elif event.key == pg.K_r and mode != "menu":
                    begin()
                elif event.key == pg.K_m:
                    sound.enabled = not sound.enabled
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                action = renderer.clicked(event.pos)
                if action == "start":
                    begin()
                elif action == "resume":
                    mode = "playing"
                elif action == "menu":
                    mode = "menu"
                elif action == "quit":
                    running = False
        active = mode == "playing" and countdown <= 0
        old_health = game.health
        if mode == "playing":
            countdown = max(0, countdown - dt)
        if active:
            keys = pg.key.get_pressed()
            controls = Controls(
                throttle=float(keys[pg.K_w] or keys[pg.K_UP]) - float(keys[pg.K_s] or keys[pg.K_DOWN]),
                steer=float(keys[pg.K_d] or keys[pg.K_RIGHT]) - float(keys[pg.K_a] or keys[pg.K_LEFT]),
                brake=bool(keys[pg.K_SPACE]),
                nitro=bool(keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]),
            )
            if args.smoke_test:
                controls = Controls(throttle=1, nitro=60 < frames < 120,
                                    steer=.65 if 160 < frames < 190 else 0,
                                    brake=160 < frames < 190)
            accumulator += dt
            while accumulator >= 1/120:
                game.step(controls, 1/120)
                accumulator -= 1/120
            if game.status != "playing":
                mode = "end"
        if game.health < old_health:
            sound.impact()
        sound.update(game, active)
        renderer.scene(game, dt, active)
        if mode != "menu":
            renderer.hud(game, not sound.enabled)
        if mode in ("menu", "paused", "end"):
            renderer.overlay(game, mode)
        else:
            renderer.buttons.clear()
            if countdown > 0:
                renderer.text(str(math.ceil(countdown)), (WIDTH/2, HEIGHT/2-35), 100, WHITE, True, "center")
                renderer.text("GET READY", (WIDTH/2, HEIGHT/2+35), 15, ORANGE, True, "midtop")
        renderer.present()
        frames += 1
        if args.smoke_test and frames >= 240:
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                pg.image.save(renderer.frame, str(args.screenshot))
            print(f"Render smoke check: {frames} frames; status={game.status}; health={game.health:.1f}")
            running = False
    pg.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
