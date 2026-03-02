from numpy.random.mtrand import standard_cauchy
from drone_prototype import Ai_Drone
import math
import numpy as np

def hover_scorer(drones: list[Ai_Drone], screen_width, screen_height, meters_to_pixels,
                 screen, clock, limit=10):
    import pygame as pg

    # fps font
    font = pg.font.SysFont(None, 24)

    # fixed target in center
    target = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))

    # drone initialization
    if limit >= 20:
        start_pos = np.array([np.random.uniform(0, screen_width) / meters_to_pixels,
                             np.random.uniform(0, screen_height) / meters_to_pixels])
    else:
        start_pos = target
    start_orientation = 0 if np.random.rand() < 0.5 else np.pi

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
    distance_limit = np.sqrt(screen_height**2 + screen_width**2)

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

            score = 1 / (1.0 + dist)  # reward smaller distances more heavily
            scores[i] += score * max((1 - (abs(drone.angle)/np.deg2rad(45))), 0.9)  # penalize bad angles

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
    # fixed target in center
    target = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))
    start_pos = np.array([np.random.uniform(0, screen_width) / meters_to_pixels,
                        np.random.uniform(0, screen_height) / meters_to_pixels])
    start_orientation = 0 if np.random.rand() < 0.5 else np.pi

    # initialize all drones
    for drone in drones:
        drone.reset_state(start_pos)
        drone.angle = start_orientation
        drone.waypoint = target.copy()

    scores = scores = np.zeros(len(drones))
    time = 0.0
    dt = 0.016
    frame_count = 0
    distance_limit = np.sqrt(screen_height**2 + screen_width**2) / meters_to_pixels
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

            score = 1 / (1.0 + dist)  # reward smaller distances more heavily
            scores[i] += score * max((1 - (abs(drone.angle)/deg45_rads)), 0.9)  # penalize bad angles

            # disable if too far
            if dist > distance_limit:
                drone.enabled = False

        time += dt
        frame_count += 1

    return return_code, scores, frame_count
