from drone_prototype import Ai_Drone
import pygame as pg
import numpy as np

def hover_scorer(drones: list[Ai_Drone], screen_width, screen_height, meters_to_pixels, 
                 screen: pg.Surface, clock: pg.time.Clock, visualize=False):
    scores = []
    # fixed target in center
    target = (screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels))
    for drone in drones:
        # initialize drone
        start_pos = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))
        drone.reset_state(start_pos)
        drone.waypoint = np.array(target, dtype=float)
        
        score = 0.0
        time = 0.0

        dt = 0.016
        while time < 10.0:
            # visual mode clock
            if visualize:
                dt = clock.tick(60) / 1000
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        time = 10

            # update drone
            drone.handle_input(None, dt)
            drone.update(dt)

            # calculate score
            dist = np.linalg.norm(drone.pos - target)
            score += 1.0 / (1.0 + dist)  # reward smaller distances more heavily
            time += dt

            if visualize:
                # draw drone and target
                screen.fill((20, 20, 20))
                drone.draw(screen, dt)
                pg.draw.circle(screen, (230, 100, 100), (int(target[0]*meters_to_pixels), int(screen_height - target[1]*meters_to_pixels)), 5)
                pg.display.flip()
        
        # append score
        scores.append(score)
    
    return scores