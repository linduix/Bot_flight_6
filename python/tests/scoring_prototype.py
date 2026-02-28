from drone_prototype import Ai_Drone
import pygame as pg
import numpy as np

def hover_scorer(drones: list[Ai_Drone], screen_width, screen_height, meters_to_pixels, 
                 screen: pg.Surface, clock: pg.time.Clock, visualize=False, limit=10):
    # fps font
    font = pg.font.SysFont(None, 24)

    # fixed target in center
    target = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))
    
    # initialize all drones
    for drone in drones:
        start_pos = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))
        drone.reset_state(start_pos)
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
        if visualize:
            dt = clock.tick(60) / 1000
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    time = 10
                    return_code = 1
            screen.fill((20, 20, 20))

        for i, drone in enumerate(drones):
            # skip if disabled
            if not drone.enabled:
                continue

            # update drone
            drone.handle_input(None, dt)
            drone.update(dt)

            # calculate score
            dist = np.linalg.norm(drone.pos - target)
            # disable if too far
            # if dist > distance_limit:
            #     drone.enabled = False
            scores[i] += (time + 0.1) / (1.0 + dist)  # reward smaller distances more heavily

        if visualize:
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
            pg.draw.circle(screen, (100, 230, 100), (int(target[0]*meters_to_pixels), int(screen_height - target[1]*meters_to_pixels)), 5)

            # fps + time counter
            fps_surf = font.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 255))
            time_surf = font.render(f"Time: {time:.1f}s", True, (255, 255, 255))
            screen.blit(fps_surf, (10, 10))
            screen.blit(time_surf, (10, 30))
            
            pg.display.flip()

        time += dt
        frame_count += 1

    return return_code, scores, frame_count