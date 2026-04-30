from threading import RLock
import pygame as pg
import numpy as np
from drone import Drone
import sys
import math

def main():
    pg.init()

    # setup
    config = {
        "width": 800,
        "height": 600,
        "caption": "BotFlight6 Visualization"
    }
    screen = pg.display.set_mode((config["width"], config["height"]))
    pg.display.set_caption(config["caption"])
    clock = pg.time.Clock()

    meters_to_pixels = 10
    R = math.hypot(config['width'] / meters_to_pixels , config['height'] / meters_to_pixels)

    # drone creation
    spawn_pos = np.array([config["width"]/(2*meters_to_pixels), config["height"]/(2*meters_to_pixels)])
    drone = Drone(spawn_pos, meters_to_pixels, config["height"])

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        dt = clock.tick(60) / 1000
        keys = pg.key.get_pressed()

        d = math.hypot(spawn_pos[0] - drone.pos[0], spawn_pos[1] - drone.pos[1])
        if d > R * 1.5:
            drone.reset_state(spawn_pos)

        # Black background
        screen.fill((20, 20, 20))
        drone.handle_input(keys, dt)
        drone.update(dt)
        drone.draw_body(screen)
        drone.draw_particles(screen, dt)

        pg.display.flip()

    pg.quit()
    sys.exit()

if __name__ == "__main__":
    main()
