"""Coordinated pursuit using shared sightings and collision-safe street routes.

Only observe() reads the player's current position. Patrol cars steer towards
radio reports, predicted interception points and street waypoints; they never
teleport or ignore the same physical collisions as the player.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math


def _clamp(value, low, high):
    return max(low, min(high, value))


def _angle(target, current):
    return (target - current + math.pi) % math.tau - math.pi


@dataclass
class _Navigation:
    route: list = field(default_factory=list)
    replan: float = 0.0
    last_sample: tuple | None = None
    sample_time: float = 0.0
    stalled: float = 0.0
    recovering: float = 0.0
    recovery_turn: float = 0.0
    target: tuple = (0.0, 0.0)
    commanded: float = 0.0
    junction: tuple | None = None
    stage: tuple | None = None
    approach: tuple | None = None
    suspect_direction: tuple = (0.0, 0.0)
    plan_started: float = 0.0
    ram_until: float = 0.0
    intercept_cooldown: float = 0.0


class PursuitDirector:
    SIGHT_RANGE = 780.0
    POLICE_SPEED = 382.0

    def __init__(self, game, controls_type, road_centers, world_size):
        self.game = game
        self.controls_type = controls_type
        self.roads = tuple(road_centers)
        self.world_size = world_size
        self.time = 0.0
        # Dispatch knows where this chase started; further updates require LOS.
        self.last_seen = (game.player.x, game.player.y)
        self.last_velocity = (game.player.vx, game.player.vy)
        self.last_seen_time = 0.0
        self.visible_indices: set[int] = set()
        self.targets: list[tuple[float, float]] = []
        self.roles: list[str] = []
        self._states: list[_Navigation] = []
        self._sight_timer = 0.0
        self._tactic_timer = 0.0
        self._nodes = [(ix, iy) for ix in range(len(self.roads)) for iy in range(len(self.roads))]
        self._ensure_states()
        self._assign_targets()

    @property
    def contact_age(self):
        return self.time - self.last_seen_time

    def _ensure_states(self):
        while len(self._states) < len(self.game.cops):
            self._states.append(_Navigation())
        self._states = self._states[:len(self.game.cops)]

    @staticmethod
    def _intersects(x, y, tx, ty, left, top, right, bottom):
        """Slab intersection of a segment and an expanded building rectangle."""
        enter, leave = 0.0, 1.0
        for start, delta, lo, hi in ((x, tx-x, left, right), (y, ty-y, top, bottom)):
            if abs(delta) < 1e-10:
                if start <= lo or start >= hi:
                    return False
                continue
            a, b = (lo-start)/delta, (hi-start)/delta
            if a > b:
                a, b = b, a
            enter, leave = max(enter, a), min(leave, b)
            if enter >= leave:
                return False
        return leave > 0.0 and enter < 1.0

    def segment_clear(self, x, y, tx, ty, margin=22):
        if not (17 <= tx <= self.world_size-17 and 17 <= ty <= self.world_size-17):
            return False
        for b in self.game.buildings:
            if self._intersects(x, y, tx, ty, b.x-margin, b.y-margin,
                                b.x+b.w+margin, b.y+b.h+margin):
                return False
        return True

    def observe(self, dt):
        self.time += dt
        self._ensure_states()
        self._sight_timer -= dt
        self._tactic_timer -= dt
        if self._sight_timer <= 0:
            self._sight_timer = .12
            player = self.game.player
            self.visible_indices = {
                index for index, cop in enumerate(self.game.cops)
                if math.hypot(player.x-cop.x, player.y-cop.y) <= self.SIGHT_RANGE
                and self.segment_clear(cop.x, cop.y, player.x, player.y, margin=0)
            }
            if self.visible_indices:
                self.last_seen = (player.x, player.y)
                self.last_velocity = (player.vx, player.vy)
                self.last_seen_time = self.time
        if self._tactic_timer <= 0:
            self._tactic_timer = .36
            self._assign_targets()

    def _road_point(self, point):
        x, y = point
        x = _clamp(x, self.roads[0], self.roads[-1])
        y = _clamp(y, self.roads[0], self.roads[-1])
        rx = min(self.roads, key=lambda r: abs(r-x))
        ry = min(self.roads, key=lambda r: abs(r-y))
        return (rx, y) if abs(rx-x) < abs(ry-y) else (x, ry)

    def _forecast(self, seconds):
        # Predict along the current road, not diagonally through a city block.
        px, py = self.last_seen
        vx, vy = self.last_velocity
        x, y = self._road_point((px, py))
        travel = min(math.hypot(vx, vy), 500) * seconds
        if abs(vx) > abs(vy):
            y = min(self.roads, key=lambda r: abs(r-py))
            x += math.copysign(travel, vx)
        elif abs(vy) > 1:
            x = min(self.roads, key=lambda r: abs(r-px))
            y += math.copysign(travel, vy)
        return (_clamp(x, self.roads[0], self.roads[-1]),
                _clamp(y, self.roads[0], self.roads[-1]))

    def _travel_direction(self):
        vx, vy = self.last_velocity
        return (math.copysign(1.0, vx), 0.0) if abs(vx) > abs(vy) else (0.0, math.copysign(1.0, vy))

    def _clear_interception(self, state):
        state.junction = state.stage = state.approach = None
        state.ram_until = 0.0
        state.replan = 0.0

    def _valid_interception(self, state):
        if state.junction is None:
            return False
        if state.ram_until and self.time >= state.ram_until:
            state.intercept_cooldown = self.time + 3.0
            return False
        if self.contact_age > 1.6 or self.time - state.plan_started > 13:
            return False
        speed = math.hypot(*self.last_velocity)
        if speed < 85 or self._travel_direction() != state.suspect_direction:
            return False
        dx, dy = state.suspect_direction
        ahead = ((state.junction[0]-self.last_seen[0])*dx
                 + (state.junction[1]-self.last_seen[1])*dy)
        cross = abs((state.junction[0]-self.last_seen[0])*dy
                    - (state.junction[1]-self.last_seen[1])*dx)
        return ahead > -130 and cross < 140

    def _route_distance_to_node(self, cop, goal):
        # Entry points must be physically visible; Manhattan distance by itself
        # can underestimate a route that first has to go around a city block.
        best = math.inf
        for node in self._nodes:
            point = self._point(node)
            if not self.segment_clear(cop.x, cop.y, *point, margin=18):
                continue
            distance = math.hypot(point[0]-cop.x, point[1]-cop.y)
            distance += abs(point[0]-goal[0]) + abs(point[1]-goal[1])
            turn = abs(_angle(math.atan2(point[1]-cop.y, point[0]-cop.x), cop.angle))
            best = min(best, distance + turn*70)
        return best

    def _choose_interception(self, cop, occupied):
        """Choose the earliest reachable junction, not an arbitrary lead point."""
        direction = self._travel_direction()
        dx, dy = direction
        estimate = self._forecast(min(self.contact_age, .8))
        speed = max(85, math.hypot(*self.last_velocity))
        axis = 0 if dx else 1
        sign = dx or dy
        road = min(self.roads, key=lambda r: abs(r-estimate[1-axis]))
        ahead_nodes = [r for r in self.roads if (r-estimate[axis])*sign > 100]
        ahead_nodes.sort(key=lambda r: (r-estimate[axis])*sign)
        for coordinate in ahead_nodes:
            junction = (coordinate, road) if axis == 0 else (road, coordinate)
            if junction in occupied:
                continue
            suspect_eta = (coordinate-estimate[axis])*sign/speed
            if suspect_eta > 10:
                break
            choices = []
            for side in (-1, 1):
                cross_coordinate = road + side*420
                if cross_coordinate not in self.roads:
                    continue
                approach = ((coordinate, cross_coordinate) if axis == 0
                            else (cross_coordinate, coordinate))
                stage = ((coordinate, road+side*145) if axis == 0
                         else (road+side*145, coordinate))
                # Approach from the side facing the junction, never by driving
                # through the junction and then making a U-turn at the stage.
                along = math.hypot(cop.x-stage[0], cop.y-stage[1])
                to_junction = math.hypot(cop.x-junction[0], cop.y-junction[1])
                if (self.segment_clear(cop.x, cop.y, *stage, margin=22)
                        and to_junction > 130):
                    route_length = along
                else:
                    route_length = self._route_distance_to_node(cop, approach) + 275
                # Account for cornering and starting from low speed, plus the
                # final acceleration across the last 145 px of the side street.
                police_eta = route_length/315 + max(0, 250-cop.speed)/600 + 1.1
                if police_eta <= suspect_eta + .12:
                    choices.append((police_eta, stage, approach))
            if choices:
                police_eta, stage, approach = min(choices)
                return junction, stage, approach, direction
        return None

    def _assign_targets(self):
        if not self.game.cops:
            self.roles, self.targets = [], []
            return
        estimate = self._forecast(min(self.contact_age, 1.1))
        locked = set()
        for index, state in enumerate(self._states):
            if self._valid_interception(state):
                locked.add(index)
            elif state.junction is not None:
                self._clear_interception(state)
        candidates = [i for i in range(len(self.game.cops)) if i not in locked]
        closest = min(candidates or range(len(self.game.cops)), key=lambda i:
                      math.hypot(self.game.cops[i].x-estimate[0], self.game.cops[i].y-estimate[1]))
        self.roles = ["intercept"] * len(self.game.cops)
        self.roles[closest] = "chase"
        self.targets = [estimate] * len(self.game.cops)
        order = [i for i in range(len(self.game.cops)) if i != closest]
        speed = math.hypot(*self.last_velocity)
        self.targets[closest] = self._forecast(min(self.contact_age, 1.0) + .22)
        occupied = {self._states[i].junction for i in locked}
        for rank, index in enumerate(order):
            state = self._states[index]
            if index in locked:
                self.roles[index] = "ram" if state.ram_until > self.time else "ambush"
                self.targets[index] = state.junction if self.roles[index] == "ram" else state.stage
                continue
            self.roles[index] = "intercept" if rank % 2 == 0 else "block"
            cop = self.game.cops[index]
            separation = math.hypot(cop.x-estimate[0], cop.y-estimate[1])
            if (separation > 550 and speed > 85 and self.contact_age < 1.0
                    and state.intercept_cooldown <= self.time):
                plan = self._choose_interception(cop, occupied)
                if plan is not None:
                    state.junction, state.stage, state.approach, state.suspect_direction = plan
                    state.plan_started = self.time
                    state.replan = 0.0
                    occupied.add(state.junction)
                    self.roles[index], self.targets[index] = "ambush", state.stage
                    continue
            lead = (1.2 if rank % 2 == 0 else 2.4) + min(.8, separation/1600)
            target = self._forecast(min(self.contact_age, 1.0) + lead)
            if speed < 55 or self.contact_age > 5:
                # Search exits around the last report instead of silently reading
                # the player's new position after losing visual contact.
                ix = min(range(len(self.roads)), key=lambda i: abs(self.roads[i]-estimate[0]))
                iy = min(range(len(self.roads)), key=lambda i: abs(self.roads[i]-estimate[1]))
                if self.contact_age > 5:
                    offsets = ((1, 0), (0, 1), (-1, 0), (0, -1))
                    dx, dy = offsets[(rank + int(self.contact_age/4)) % 4]
                    target = (self.roads[int(_clamp(ix+dx, 0, len(self.roads)-1))],
                              self.roads[int(_clamp(iy+dy, 0, len(self.roads)-1))])
                else:
                    # At low speed converge to contain the suspect; no orbiting
                    # around an arbitrary future intersection.
                    target = self._road_point(self.last_seen)
            self.targets[index] = target

    def _point(self, node):
        return self.roads[node[0]], self.roads[node[1]]

    def _path(self, start, goal, index):
        queue = [(0, 0, start)]
        previous, cost = {start: None}, {start: 0.0}
        direction_order = ((1, 0), (0, 1), (-1, 0), (0, -1))
        direction_order = direction_order[index % 4:] + direction_order[:index % 4]
        serial = 0
        # Small reservations prefer another approach at equal-length forks.
        occupied = set()
        for other_index, state in enumerate(self._states):
            if other_index != index:
                for a, b in zip(state.route, state.route[1:]):
                    occupied.add((a, b))
                    occupied.add((b, a))
        while queue:
            _, _, node = heapq.heappop(queue)
            if node == goal:
                break
            for dx, dy in direction_order:
                nxt = node[0]+dx, node[1]+dy
                if not (0 <= nxt[0] < len(self.roads) and 0 <= nxt[1] < len(self.roads)):
                    continue
                a, b = self._point(node), self._point(nxt)
                distance = math.hypot(b[0]-a[0], b[1]-a[1])
                extra = 36 if (a, b) in occupied else 0
                new_cost = cost[node]+distance+extra
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt], previous[nxt] = new_cost, node
                    serial += 1
                    heuristic = abs(self.roads[nxt[0]]-self.roads[goal[0]]) + abs(self.roads[nxt[1]]-self.roads[goal[1]])
                    heapq.heappush(queue, (new_cost+heuristic, serial, nxt))
        result = []
        node = goal
        while node is not None:
            result.append(self._point(node))
            node = previous.get(node)
        return list(reversed(result))

    def _plan(self, cop, index, state):
        target = self.targets[index]
        state.target = target
        goal_nodes = sorted(self._nodes, key=lambda n:
                            math.hypot(self.roads[n[0]]-target[0], self.roads[n[1]]-target[1]))
        goal = next((node for node in goal_nodes
                     if self.segment_clear(*self._point(node), *target, margin=23)), goal_nodes[0])
        if state.junction is not None and not state.ram_until and state.approach is not None:
            # Close to the stage the car can finish directly. Further away it
            # must arrive via the outer end of that perpendicular street.
            if (math.hypot(cop.x-target[0], cop.y-target[1]) < 310
                    and self.segment_clear(cop.x, cop.y, *target, margin=22)):
                state.route = [target]
                state.replan = .42
                return
            goal = (self.roads.index(state.approach[0]), self.roads.index(state.approach[1]))
        candidates = [node for node in self._nodes
                      if self.segment_clear(cop.x, cop.y, *self._point(node), margin=23)]
        if not candidates:
            # A car resting against a wall is only 17 px from it. Its first
            # escape segment can leave that wall with physical-radius clearance.
            candidates = [node for node in self._nodes
                          if self.segment_clear(cop.x, cop.y, *self._point(node), margin=16.5)]
        if not candidates:
            state.route = [self._road_point((cop.x, cop.y))]
        else:
            def score(node):
                x, y = self._point(node)
                error = abs(_angle(math.atan2(y-cop.y, x-cop.x), cop.angle))
                return (math.hypot(x-cop.x, y-cop.y)
                        + abs(self.roads[node[0]]-self.roads[goal[0]])
                        + abs(self.roads[node[1]]-self.roads[goal[1]])
                        + error * min(cop.speed, 170) * .32)
            entry = min(candidates, key=score)
            state.route = self._path(entry, goal, index)
            if math.hypot(state.route[-1][0]-target[0], state.route[-1][1]-target[1]) > 12:
                state.route.append(target)
        state.replan = .42 + index*.035

    def _start_recovery(self, cop, state, target):
        # Reverse straight first to create clearance, then turn the nose toward
        # the route. This is a physical three-point turn, never a position reset.
        error = _angle(math.atan2(target[1]-cop.y, target[0]-cop.x), cop.angle)
        state.recovery_turn = -1.0 if error >= 0 else 1.0
        state.recovering = 1.1
        state.stalled = 0.0
        state.replan = 0.0

    def drive(self, cop, index, dt):
        self._ensure_states()
        if index >= len(self.targets):
            self._assign_targets()
        state = self._states[index]
        # A staged patrol launches when its acceleration-limited crossing time
        # matches the suspect's reported ETA. It deliberately crosses the road.
        if state.junction is not None and not state.ram_until and self.contact_age < .8:
            dx, dy = state.suspect_direction
            estimate = self._forecast(min(self.contact_age, .8))
            ahead = (state.junction[0]-estimate[0])*dx + (state.junction[1]-estimate[1])*dy
            suspect_eta = ahead/max(85, math.hypot(*self.last_velocity))
            jx, jy = state.junction[0]-cop.x, state.junction[1]-cop.y
            junction_distance = math.hypot(jx, jy)
            facing = abs(_angle(math.atan2(jy, jx), cop.angle)) < .65
            if junction_distance < 255 and facing and self.segment_clear(cop.x, cop.y, *state.junction, margin=22):
                approach_speed = max(0, (cop.vx*jx+cop.vy*jy)/max(1, junction_distance))
                distance, crossing_eta = 0.0, 0.0
                while distance < junction_distance and crossing_eta < 2.5:
                    approach_speed = min(self.POLICE_SPEED, approach_speed+(300-.42*approach_speed)*.04)
                    distance += approach_speed*.04
                    crossing_eta += .04
                if -.15 < suspect_eta <= crossing_eta + .04:
                    state.ram_until = self.time + 1.9
                    state.replan = 0.0
                    self.roles[index] = "ram"
        ramming = state.ram_until > self.time
        state.replan -= dt
        if state.replan <= 0 or not state.route:
            self._plan(cop, index, state)
        while len(state.route) > 1:
            first, second = state.route[:2]
            reached = math.hypot(first[0]-cop.x, first[1]-cop.y) < 31
            # Only skip straight, unobstructed waypoints. At corners the car
            # must enter the intersection before it starts its next street.
            aim = math.atan2(second[1]-cop.y, second[0]-cop.x)
            aligned = abs(_angle(aim, cop.angle)) < .28
            if reached or (aligned and self.segment_clear(cop.x, cop.y, *second, margin=27)):
                state.route.pop(0)
            else:
                break
        report = self.last_seen
        velocity = self.last_velocity
        distance = math.hypot(report[0]-cop.x, report[1]-cop.y)
        recent = self.contact_age < .8
        direct_point = (report[0]+velocity[0]*.16, report[1]+velocity[1]*.16)
        direct = (state.junction is None and recent and (self.roles[index] == "chase" or distance < 245)
                  and distance < 670 and self.segment_clear(cop.x, cop.y, *direct_point, margin=24))
        target = direct_point if direct else state.route[0]
        staging = False
        if state.junction is not None and not ramming:
            jx, jy = state.junction[0]-cop.x, state.junction[1]-cop.y
            junction_distance = math.hypot(jx, jy)
            if (junction_distance < 480
                    and abs(_angle(math.atan2(jy, jx), cop.angle)) < .8
                    and self.segment_clear(cop.x, cop.y, *state.junction, margin=22)):
                # Keep facing into the crossing while waiting. A generic point
                # follower would overshoot the staging point and turn away.
                staging = True
                target = state.junction
        if ramming:
            # Cross through the meeting point at speed instead of braking at
            # bumper distance like the normal pursuit/containment behaviour.
            sx, sy = state.stage
            jx, jy = state.junction
            length = math.hypot(jx-sx, jy-sy)
            target = (jx+(jx-sx)/length*145, jy+(jy-sy)/length*145)
        delta_x, delta_y = target[0]-cop.x, target[1]-cop.y
        waypoint_distance = math.hypot(delta_x, delta_y)
        error = _angle(math.atan2(delta_y, delta_x), cop.angle)
        desired = self.POLICE_SPEED + (index % 3)*4
        if abs(error) > .35:
            desired = min(desired, max(88, 245-abs(error)*92))
        if not direct and not ramming and not staging and len(state.route) > 1:
            nx, ny = state.route[1]
            turn = abs(_angle(math.atan2(ny-target[1], nx-target[0]), math.atan2(delta_y, delta_x)))
            if turn > .55:
                # v² = u² + 2as: brake before a junction, not after hitting its wall.
                desired = min(desired, math.sqrt(110**2 + 2*330*max(0, waypoint_distance-30)))
        holding = False
        if staging:
            desired = min(desired, math.sqrt(2*340*max(0, waypoint_distance-145)))
            holding = waypoint_distance < 175
        elif direct:
            suspect_speed = math.hypot(*velocity)
            desired = min(desired, max(0, (distance-43)*1.65 + suspect_speed*.95))
            holding = distance < 65 and suspect_speed < 35
        elif not ramming and len(state.route) == 1 and waypoint_distance < 100:
            desired = min(desired, max(0, (waypoint_distance-20)*2))
            holding = waypoint_distance < 50
        # Look ahead along actual velocity and the nose; it catches a slide
        # towards a wall even when the center-line route itself is valid.
        forward_x, forward_y = math.cos(cop.angle), math.sin(cop.angle)
        look = max(40, cop.speed*.32)
        blocked = not self.segment_clear(cop.x, cop.y,
                                         cop.x+forward_x*look, cop.y+forward_y*look, margin=20)
        if cop.speed > 50:
            blocked = blocked or not self.segment_clear(
                cop.x, cop.y, cop.x+cop.vx*.32, cop.y+cop.vy*.32, margin=20)
        if blocked:
            desired = min(desired, 55)
        # Feed-forward counters the simulation's .42/s rolling drag, so a
        # 382 px/s pursuit command really reaches that speed on a straight.
        throttle = _clamp(.42*desired/300 + (desired-cop.speed)/70, -1, 1)
        steer = _clamp(error*2.0, -1, 1)

        state.sample_time += dt
        if state.last_sample is None:
            state.last_sample = (cop.x, cop.y)
        if state.sample_time >= .35:
            moved = math.hypot(cop.x-state.last_sample[0], cop.y-state.last_sample[1])
            if not holding and (moved < 17 or (blocked and moved < 36)):
                state.stalled += state.sample_time
            else:
                state.stalled = max(0, state.stalled-state.sample_time*2)
            state.sample_time = 0
            state.last_sample = (cop.x, cop.y)
        if state.stalled > .65 and state.recovering <= 0:
            self._start_recovery(cop, state, target)
        if state.recovering > 0:
            state.recovering -= dt
            throttle = -1.0
            steer = 0.0 if state.recovering > .72 else state.recovery_turn
            back_x = cop.x-forward_x*52
            back_y = cop.y-forward_y*52
            if not self.segment_clear(cop.x, cop.y, back_x, back_y, margin=16.5):
                state.recovering = 0
                state.replan = 0
                throttle, steer = 1.0, -state.recovery_turn
        state.commanded = desired
        cop._route = list(state.route)
        cop._route_age = state.replan
        cop._stuck = state.stalled
        cop._reversing = state.recovering
        self.game._drive(cop, self.controls_type(throttle=throttle, steer=steer), dt,
                         self.POLICE_SPEED+(index % 3)*4)
