import pygame as pg
import numpy as np
from drone_prototype import Drone
import sys

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

    meters_to_pixels = 20

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
 