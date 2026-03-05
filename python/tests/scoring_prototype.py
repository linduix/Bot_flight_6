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

def stage1(drones: list[Ai_Drone], screen_width, screen_height, meters_to_pixels, limit=10): # static target aquisition
    # GOAL
    # 1. Max speed towards goal
    # 2. Precise breaking
    # 3. Stable hovering at position

    center = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))
    R = math.hypot(screen_width, screen_height) / (2 * meters_to_pixels) # diagonal length from center
    # fixed target in center
    target = center
    spawn = np.array([np.random.rand() * screen_width / meters_to_pixels, 
                      np.random.rand() * screen_height/ meters_to_pixels])

    # Fixed Vector Component Math
    eps = 1e-8                                # epsilon (near 0)
    a = target - spawn                        # ideal path, spawn --> target
    a0 = a / (math.hypot(a[0], a[1]) + eps)   # unit vector of ideal path

    # Stop dist calcs
    max_a = drones[0].thruster_force * 2 / drones[0].M
    stop_d = lambda v: (v**2) / (2*max_a + eps)

    # Drone init
    for drone in drones:
        drone.reset_state(spawn)
        drone.waypoint = target

    hovertime = np.zeros(len(drones))
    scores = np.zeros(len(drones))
    time = 0
    dt = 0.16
    completions = 0
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

            # Objective reward
            score = 0
            if d < 0.1 and v_mag < 0.1:
                hovertime[ix] += dt
                score += dt
                if hovertime[ix] > 0.1 * limit: 
                    score += max(100, 10 * limit)
                    # speed component
                    speed = max(50, 5 * limit) / (eps + time)
                    score += speed
                    drone.enabled = False
                    completions += 1
            else:
                if hovertime[ix] > 0:
                    hovertime[ix] -= dt

            # positive velocity alignemnt reward
            if v_par_s > 0:
                if d > stop_d(v_par_s):
                    score += dt * v_par_s
                else:
                    score -= dt * v_par_s

            # sideways vel penalty
            score -= dt * v_perp_s * 0.05
            # Ideal path reward
            score += dt / (1 + e_perp_s)
            # passive distance reward
            score += dt / (10 + d)

            # append score
            scores[ix] += score

            # disable if too far / crash
            if d > R * 2:
                drone.enabled = False
                scores[ix] /= 2

            # fix negatives
            scores[ix] += score
            if scores[ix] < 0:
                scores[ix] = 0
            scores[ix] += dt / (1+d)

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
    required_hover = 1.0  # seconds
    hovertime = np.zeros(len(drones), dtype=float)

    # initialize all drones
    for drone in drones:
        drone.reset_state(spawn)
        drone.waypoint = target.copy()
        drone.enabled = True

    scores = np.zeros(len(drones))
    top_drones: list[int] = list(np.argsort(scores)[-10:])
    time = 0.0
    distance_limit = R * 2
    completions = 0

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

            # ----- Scoring -----
            score = 0.0

            # dwell success: be close + nearly stopped for >= required_hover
            in_zone = (d < 0.1) and (v_mag < 0.1)
            if in_zone:
                hovertime[i] += dt
                score += 2 * dt  # small sustain reward (optional)

                if hovertime[i] >= required_hover:
                    score += max(100.0, 10.0 * limit)
                    score += max(50.0, 5.0 * limit) / (eps + time)  # faster is better
                    drone.enabled = False
                    completions += 1
            else:
                if hovertime[i] > 0:
                    hovertime[i] -= dt

            # accelerate vs brake shaping (closing speed only)
            v_close = max(v_par, 0.0)
            if v_close > 0.0:
                if d > stop_d(v_close):
                    score += dt * v_close
                else:
                    score -= dt * v_close

            # sideways velocity penalty + straight-line bias + distance shaping
            score -= dt * v_perp * 0.05
            score += dt / (1.0 + e_perp)
            score += dt / (10.0 + d)

            # disable if too far
            if d > distance_limit:
                drone.enabled = False
                scores[i] /= 2

            # clip scores
            scores[i] += score
            if scores[i] < 0:
                scores[i] = 0
            scores[i] += dt / (1+d)

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