"""Player vehicle dynamics: angular inertia and tyres with limited side force.

Heading and travel direction are separate. The tyres cannot redirect all of a
fast car's momentum instantly, so corners need an approach and a recovery.
This module has no graphics, clock, input or third-party dependencies.
"""
import math


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def drive_player(car, controls, dt, top_speed, road_centers, road_width):
    throttle = _clamp(controls.throttle, -1.0, 1.0)
    steering = _clamp(controls.steer, -1.0, 1.0)
    speed = car.speed
    fx, fy = math.cos(car.angle), math.sin(car.angle)
    forward = car.vx * fx + car.vy * fy
    sideways = -car.vx * fy + car.vy * fx
    slip = math.atan2(abs(sideways), max(abs(forward), 20.0))
    handbrake = bool(controls.brake and speed > 30)

    if speed < 30:
        if throttle < -.1:
            car._reverse_gear = True
        elif throttle > .1:
            car._reverse_gear = False
    elif throttle > 0 and forward > 25:
        car._reverse_gear = False
    direction = -1.0 if car._reverse_gear else 1.0

    # Steering angle shrinks with speed. At a crawl the car can manoeuvre,
    # while a fast 90-degree corner spans a considerable stretch of street.
    car._steering += (steering - car._steering) * (1.0 - math.exp(-dt * 11.5))
    wheel_angle = (.63 - .37 * min(speed / 330.0, 1.0)
                   - .07 * _clamp((speed - 330.0) / 140.0, 0.0, 1.0))
    target_yaw = direction * speed / 46.5 * math.tan(car._steering * wheel_angle)
    target_yaw = _clamp(target_yaw, -2.35, 2.35)
    if handbrake:
        target_yaw *= 1.16

    # Tyres break traction from turning demand, not just from the handbrake.
    # Slip itself keeps them in the sliding regime until the driver settles it.
    demand = speed * abs(target_yaw)
    load_slide = _clamp((demand - 395.0) / 220.0, 0.0, 1.0)
    angle_slide = _clamp((slip - .22) / .53, 0.0, 1.0)
    wanted_slide = max(load_slide, angle_slide) if speed > 95 else 0.0
    if handbrake:
        wanted_slide = 1.0
    countersteer = abs(sideways) > 30 and steering * sideways > .1
    # In a right-hand drift the velocity points to the left of the nose:
    # sideways<0, so steering left is the corrective input.
    if throttle <= 0 and countersteer and not handbrake:
        wanted_slide *= .65
    slide_response = 7.0 if wanted_slide > car._traction_loss else 1.05
    car._traction_loss += (wanted_slide - car._traction_loss) * (1 - math.exp(-dt * slide_response))
    sliding = car._traction_loss

    # Rotation has its own momentum. Centering the steering wheel does not
    # instantly stop the body rotating. Countersteer supplies opposite torque.
    if abs(steering) > .08:
        yaw_response = 6.0 if countersteer else 4.8
    else:
        yaw_response = (5.5 * (1 - sliding) + 2.3 * sliding)
        if throttle <= 0:
            yaw_response += .5
    yaw_acceleration = (target_yaw - car.yaw_rate) * yaw_response
    yaw_limit = 7.4 if countersteer else 5.4
    car.yaw_rate += _clamp(yaw_acceleration, -yaw_limit, yaw_limit) * dt
    if speed < 8:
        car.yaw_rate *= math.exp(-dt * 14)
    car.angle = (car.angle + car.yaw_rate * dt) % math.tau
    fx, fy = math.cos(car.angle), math.sin(car.angle)
    forward = car.vx * fx + car.vy * fy
    sideways = -car.vx * fy + car.vy * fx

    # Bounded side force is what makes momentum survive a change in heading.
    # Keeping the throttle open sustains a slide; lifting helps the tyres bite.
    drift_grip = 1.9 if throttle > .15 else 2.95
    grip = 8.0 * (1 - sliding) + drift_grip * sliding
    lateral_limit = 510.0 * (1 - sliding) + 295.0 * sliding
    if handbrake:
        grip, lateral_limit = 1.12, 202.0
    lateral_acceleration = _clamp(-sideways * grip, -lateral_limit, lateral_limit)
    ax, ay = -fy * lateral_acceleration, fx * lateral_acceleration

    if throttle >= 0:
        force = throttle * (345.0 if car.boosting else 230.0)
        ax += fx * force
        ay += fy * force
    elif not car._reverse_gear and speed > 12:
        # The foot brake also sheds sideways momentum before a tight corner.
        braking = min(330.0 * -throttle, speed / dt)
        ax -= car.vx / speed * braking
        ay -= car.vy / speed * braking
    else:
        ax += fx * throttle * 155.0
        ay += fy * throttle * 155.0
    car.vx += ax * dt
    car.vy += ay * dt
    resistance = .25 + (.14 if handbrake else 0.0)
    road_distance = min(min(abs(car.x - x) for x in road_centers),
                        min(abs(car.y - y) for y in road_centers))
    if road_distance > road_width / 2 + 12:
        resistance += .8
        top_speed *= .67
    decay = math.exp(-resistance * dt)
    car.vx *= decay
    car.vy *= decay

    current_speed = car.speed
    # Nitro momentum bleeds off over time instead of disappearing on release.
    limit = max(top_speed, speed - 220 * dt)
    if car._reverse_gear:
        limit = min(limit, 105.0)
    if current_speed > limit:
        factor = limit / current_speed
        car.vx *= factor
        car.vy *= factor

    sideways = -car.vx * fy + car.vy * fx
    forward = car.vx * fx + car.vy * fy
    slip = math.atan2(abs(sideways), max(abs(forward), 20.0))
    car.drifting = bool(car.speed > 70 and abs(sideways) > 35 and slip > .23)
    car.x += car.vx * dt
    car.y += car.vy * dt
