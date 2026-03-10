from numpy.random.mtrand import standard_cauchy
from drone_prototype import Ai_Drone
import math
import numpy as np

def hover_scorer(drones: list[Ai_Drone], screen_width, screen_height, meters_to_pixels,
                 screen, clock, limit=10):
    import pygame as pg

    # fps font
    font = pg.font.SysFont(None, 24)

    center = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))
    R = math.hypot(screen_width, screen_height) / (2 * meters_to_pixels) # diagonal length
    # fixed target in center
    target = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))

    if limit >= 40:
        # 100% of center->corner distance, random direction (circle)
        theta = 2 * math.pi * np.random.rand()
        start_pos = center + R * 1.5 * np.array([math.cos(theta), math.sin(theta)])
        start_orientation = 2 * math.pi * np.random.rand()

    elif limit >= 10:
        # 50% of center->corner distance, random direction (circle)
        theta = 2 * math.pi * np.random.rand()
        start_pos = center + (.5 * R) * np.array([math.cos(theta), math.sin(theta)])
        start_orientation = 0

    else:
        start_pos = center.copy()  # same as target
        start_orientation = 0

    # initialize all drones
    for drone in drones:
        drone.reset_state(start_pos)
        drone.angle = start_orientation
        drone.waypoint = target.copy()

    scores = scores = np.zeros(len(drones))
    top_drones: list[int] = list(np.argsort(scores)[-10:])
    time = 0.0
    dt = 0.016
    frame_count = 0
    distance_limit = R * 2
    deg45_rads = np.deg2rad(45)

    return_code = 0
    while time < limit:
        # get indexes of top 10 drones:
        top_drones: list[int] = list(np.argsort(scores)[-10:])

        # visual mode clock
        dt = clock.tick(60) / 1000
        for event in pg.event.get():
            if event.type == pg.QUIT:
                time = 10
                return_code = 1
        screen.fill((20, 20, 20))

        if return_code:
            break

        for i, drone in enumerate(drones):
            # skip if disabled
            if not drone.enabled:
                continue

            # update drone
            drone.handle_input(None, dt)
            drone.update(dt)

            # calculate score
            dx = drone.pos[0] - target[0]
            dy = drone.pos[1] - target[1]
            dist = math.hypot(dx, dy)

            score = 1 / (1.0 + dist)**2  # reward smaller distances more heavily
            scores[i] += score * max((1 - (abs(drone.angle)/deg45_rads)), 0.7)  # penalize bad angles

            # disable if too far
            if dist > distance_limit:
                drone.enabled = False

        # draw particles and body
        for i, drone in enumerate(drones):
            if i == top_drones[-1]:
                drone.draw_particles(screen, dt)
            elif i in top_drones:
                drone.draw_particles(screen, dt, a=100)
        for i, drone in enumerate(drones):
            if i == top_drones[-1]:
                drone.draw_body(screen)
            elif i in top_drones:
                drone.draw_body(screen, a=100)

        # draw target
        pg.draw.circle(screen, (100, 230, 100), (int(target[0]*meters_to_pixels), int(screen_height - target[1]*meters_to_pixels)), 2)

        # fps + time counter
        fps_surf = font.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 255))
        time_surf = font.render(f"Time: {time:.1f}s", True, (255, 255, 255))
        screen.blit(fps_surf, (10, 10))
        screen.blit(time_surf, (10, 30))

        pg.display.flip()

        time += dt
        frame_count += 1

    return return_code, scores, frame_count

def hover_scorer_headless(drones: list[Ai_Drone], screen_width, screen_height, meters_to_pixels, limit=10):
    center = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))
    R = math.hypot(screen_width, screen_height) / (2 * meters_to_pixels) # diagonal length
    # fixed target in center
    target = center
    start_pos = center.copy()  # same as target

    # initialize all drones
    for drone in drones:
        drone.reset_state(start_pos)
        drone.waypoint = target.copy()

    scores = scores = np.zeros(len(drones))
    time = 0.0
    dt = 0.016
    frame_count = 0
    distance_limit = R * 2
    deg45_rads = np.deg2rad(45)

    return_code = 0
    while time < limit:
        if return_code:
            break

        for i, drone in enumerate(drones):
            # skip if disabled
            if not drone.enabled:
                continue

            # update drone
            drone.handle_input(None, dt)
            drone.update(dt)

            # calculate score
            dx = drone.pos[0] - target[0]
            dy = drone.pos[1] - target[1]
            dist = math.hypot(dx, dy)

            score = 1 / (1.0 + dist)**2  # reward smaller distances more heavily
            scores[i] += score * max((1 - (abs(drone.angle)/deg45_rads)), 0.7)  # penalize bad angles

            # disable if too far
            if dist > distance_limit:
                drone.enabled = False

        time += dt
        frame_count += 1

    return return_code, scores, frame_count

def stage1(drones: list[Ai_Drone], screen_width, screen_height, meters_to_pixels, limit=10, diff=10): # static target aquisition
    # GOAL
    # 1. Max speed towards goal
    # 2. Precise breaking
    # 3. Stable hovering at position

    center = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))
    R = math.hypot(screen_width, screen_height) / (2 * meters_to_pixels) # diagonal length from center
    # fixed target in center
    target = center
    # difficulty distance, random direction (circle)
    theta = 2 * math.pi * np.random.rand()
    spawn = center + diff * np.array([math.cos(theta), math.sin(theta)])

    # Fixed Vector Component Math
    eps = 1e-8                                # epsilon (near 0)
    a = target - spawn                        # ideal path, spawn --> target
    a0 = a / (math.hypot(a[0], a[1]) + eps)   # unit vector of ideal path

    # Max drone acceleration
    max_a = drones[0].thruster_force * 2 / drones[0].M

    # Scoring elements
    d_initial = math.hypot(target[0] - spawn[0], target[1] - spawn[1])
    base_bonus = max(400, d_initial * 4)

    # Drone init
    for drone in drones:
        drone.reset_state(spawn)
        drone.waypoint = target

    # per drone tracking
    hovertime = np.zeros(len(drones))
    scores = np.zeros(len(drones))
    prev_d = np.full(len(drones), d_initial)

    time = 0
    dt = 0.016
    completions = []
    while time < limit:
        if not any([d.enabled for d in drones]):
            break

        for ix, drone in enumerate(drones):
            if not drone.enabled:
                continue

            # update drone
            drone.handle_input(None, dt)
            drone.update(dt)

            # Variable Vector Component Math
            p, v = drone.pos, drone.v
            v_mag: float = math.hypot(v[0], v[1])

            r = target - p                                         # to objective vector
            d: float = math.hypot(r[0], r[1])                      # distance
            u = r / ( d + eps )                                    # objective unit vector

            v_par_s: float = np.dot(v, u)                          # signed mag of parallel v
            v_par_v = v_par_s * u                                  # vel in objective direction
            v_perp_v = v - v_par_v                                 # vector component not in ojective direction
            v_perp_s: float = math.hypot(v_perp_v[0], v_perp_v[1]) # mag of perp v

            r0 = p - spawn                                         # position from spawn
            e_perp_v = r0 - np.dot(r0, a0) * a0                    # component of position perp to ideal path
            e_perp_s: float = math.hypot(e_perp_v[0], e_perp_v[1]) # magnitude of perp error

            safe_v = np.sqrt( 2 * max_a * d)

            score = 0.0

            # 1. Max speed Reward
            if d > 1 and v_par_s > 0:
                reward = 5 * dt * min(v_par_s / safe_v, 1)
                overspeed = dt * max(v_par_s - safe_v, 0) ** 2
                score += reward - overspeed

            # 2. Penalise retreating at all distances
            if v_par_s < 0:
                score -= dt * abs(v_par_s) * 1.0              # symmetric with approach reward

            # 3. Lateral penalty — cheap far away, brutal near target
            prox = 1.0 + 1.5 / max(d, 0.3)                   # d=0.3 → 6x, d=3 → 1.5x, d=inf → 1x
            score -= dt * v_perp_s * prox

            # 4. Ideal path penalty
            score -= dt * 0.1 * e_perp_s
            # 5. Distance potential
            score += 0.5 * (prev_d[ix] - d)
            # Dilly Dally penalty
            score -= 1 * dt

            # 6. Precision zone — penalise any motion, reward level attitude
            if d < 1.0:
                score -= dt * v_mag * 3.0                      # damps all oscillation
                score -= dt * d * 4.0                          # pulls toward exact center
                score -= dt * abs(drone.angle) * 0.5           # level hover

            # 7. Hover zone + completion
            in_zone = d < 0.5 and v_mag < 0.5 and abs(drone.angle) < 0.1
            if in_zone:
                hovertime[ix] += dt
                score += dt * 3.0
                score -= dt * drone.angle ** 2 * 1.5          # upright hover strongly rewarded
                if hovertime[ix] > 0.1 * limit:
                    score += base_bonus                        # always 4x max approach reward
                    score += base_bonus * (1 - time / limit)  # speed bonus, up to 2x base_bonus
                    drone.enabled = False
                    completions.append(time)
            else:
                if hovertime[ix] > 0:
                    hovertime[ix] -= dt

            # 8. Out of bounds
            if d > R * 2:
                drone.enabled = False
                scores[ix] /= 2

            scores[ix] += score
            scores[ix] = max(scores[ix], 0)

            # Update prev vals
            prev_d[ix] = d

        time += dt
    return 0, scores, completions

def stage1_viz(
    drones: list[Ai_Drone],
    screen_width,
    screen_height,
    meters_to_pixels,
    screen,
    clock,
    limit=10,
):
    import pygame as pg

    font = pg.font.SysFont(None, 24)

    # ----- Setup (world units) -----
    eps = 1e-8
    dt = 0.016

    center = np.array(
        (screen_width / (2 * meters_to_pixels), screen_height / (2 * meters_to_pixels)),
        dtype=float,
    )
    R = math.hypot(screen_width, screen_height) / (2 * meters_to_pixels)
    target = center.copy()

    # random spawn anywhere in arena
    spawn = np.array(
        [
            np.random.rand() * screen_width / meters_to_pixels,
            np.random.rand() * screen_height / meters_to_pixels,
        ],
        dtype=float,
    )

    # ideal path (spawn -> target)
    a = target - spawn
    a0 = a / (math.hypot(a[0], a[1]) + eps)

    # stop distance estimate
    max_a = drones[0].thruster_force * 2 / drones[0].M

    def stop_d(v_close: float) -> float:
        return (v_close * v_close) / (2 * max_a + eps)

    # success dwell requirement
    hovertime = np.zeros(len(drones), dtype=float)

    # Scoring elements
    d_initial = math.hypot(target[0] - spawn[0], target[1] - spawn[1])
    base_bonus = max(400, d_initial * 4)

    # initialize all drones
    for drone in drones:
        drone.reset_state(spawn)
        drone.waypoint = target.copy()
        drone.enabled = True

    scores = np.zeros(len(drones))
    top_drones: list[int] = list(np.argsort(scores)[-10:])
    time = 0.0
    completions = []

    return_code = 0
    while time < limit:
        top_drones = list(np.argsort(scores)[-10:])

        # visual mode clock
        dt = clock.tick(60) / 1000
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return_code = 1
                time = limit

        screen.fill((20, 20, 20))
        if return_code:
            break

        for i, drone in enumerate(drones):
            if not drone.enabled:
                continue

            # update drone
            drone.handle_input(None, dt)
            drone.update(dt)

            # ----- Vector math -----
            p, v = drone.pos, drone.v
            v_mag = math.hypot(v[0], v[1])

            r = target - p
            d = math.hypot(r[0], r[1])
            u = r / (d + eps)  # unit direction to target

            v_par = float(np.dot(v, u))  # signed closing speed
            v_par_v = v_par * u
            v_perp_v = v - v_par_v
            v_perp = math.hypot(v_perp_v[0], v_perp_v[1])

            r0 = p - spawn
            e_perp_v = r0 - float(np.dot(r0, a0)) * a0
            e_perp = math.hypot(e_perp_v[0], e_perp_v[1])


            score = 0.0
            # 1. Approach reward — total integrates to d_initial, completion always 4x this
            if v_par > 0:
                if d > stop_d(v_par):
                    score += dt * v_par                      # +v m/s per second, sums to distance traveled
                else:
                    score -= dt * v_par * 0.75                # underbraking penalty, 2.5x the approach reward

            # 2. Penalise retreating at all distances
            if v_par < 0:
                score -= dt * abs(v_par) * 1.0              # symmetric with approach reward

            # 3. Lateral penalty — cheap far away, brutal near target
            prox = 1.0 + 1.5 / max(d, 0.3)                   # d=0.3 → 6x, d=3 → 1.5x, d=inf → 1x
            score -= dt * v_perp * prox

            # 4. Straight-line path bias
            score += dt / (1.0 + e_perp)

            # 5. Progress reward — sums to ~log(1+d_initial), minor but consistent
            score += dt / (1.0 + d)

            # 6. Precision zone — penalise any motion, reward level attitude
            if d < 1.0:
                score -= dt * v_mag * 3.0                      # damps all oscillation
                score -= dt * d * 4.0                          # pulls toward exact center
                score -= dt * abs(drone.angle) * 0.5           # level hover

            # 7. Hover zone + completion
            in_zone = d < 0.5 and v_mag < 0.4
            if in_zone:
                hovertime[i] += dt
                score += dt * 3.0
                score -= dt * drone.angle ** 2 * 1.5          # upright hover strongly rewarded
                if hovertime[i] > 0.1 * limit:
                    score += base_bonus                        # always 4x max approach reward
                    score += base_bonus * (1 - time / limit)  # speed bonus, up to 2x base_bonus
                    drone.enabled = False
                    completions.append(time)
            else:
                if hovertime[i] > 0:
                    hovertime[i] -= dt

            # 8. Out of bounds
            if d > R * 2:
                drone.enabled = False
                scores[i] /= 2

            scores[i] += score
            scores[i] = max(scores[i], 0)

        # draw target + spawn + ideal path line
        pg.draw.circle(
            screen,
            (100, 230, 100),
            (int(target[0] * meters_to_pixels), int(screen_height - target[1] * meters_to_pixels)),
            3,
        )
        pg.draw.circle(
            screen,
            (230, 100, 100),
            (int(spawn[0] * meters_to_pixels), int(screen_height - spawn[1] * meters_to_pixels)),
            3,
        )
        # line: spawn -> target
        pg.draw.line(
            screen,
            (80, 80, 200),
            (int(spawn[0] * meters_to_pixels), int(screen_height - spawn[1] * meters_to_pixels)),
            (int(target[0] * meters_to_pixels), int(screen_height - target[1] * meters_to_pixels)),
            1,
        )

        # draw particles and body
        for i, drone in enumerate(drones):
            if i == top_drones[-1]:
                drone.draw_particles(screen, dt)
            elif i in top_drones:
                drone.draw_particles(screen, dt, a=100)

        for i, drone in enumerate(drones):
            if i == top_drones[-1]:
                drone.draw_body(screen)
            elif i in top_drones:
                drone.draw_body(screen, a=100)

        # fps + time
        fps_surf = font.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 255))
        time_surf = font.render(f"Time: {time:.1f}s", True, (255, 255, 255))
        screen.blit(fps_surf, (10, 10))
        screen.blit(time_surf, (10, 30))

        pg.display.flip()
        time += dt

    return return_code, scores, completions
