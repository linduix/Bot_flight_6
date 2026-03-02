from drone_prototype import Ai_Drone
import util_prototype as utils
import multiprocessing as mp
import pygame as pg
import numpy as np
import math
import sys

def exhibition(drone: Ai_Drone, screen_width, screen_height, meters_to_pixels, screen, clock):
    # fixed target in center
    target = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))

    # drone initialization
    start_pos = target

    # initialize all drones
    drone.reset_state(start_pos)
    drone.waypoint = target.copy()

    time = 0.0
    distance_limit = np.sqrt(screen_height**2 + screen_width**2) / meters_to_pixels

    while True:
        # update target
        target = pg.mouse.get_pos()
        target = (target[0] / meters_to_pixels, (screen_height - target[1]) / meters_to_pixels)
        drone.waypoint = np.array(target)

        # visual mode clock
        dt = clock.tick(60) / 1000
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return 1

        screen.fill((20, 20, 20))

        # update drone
        drone.handle_input(None, dt)
        drone.update(dt)

        # calculate score
        dx = drone.pos[0] - target[0]
        dy = drone.pos[1] - target[1]
        dist = math.hypot(dx, dy)

        # disable if too far
        if dist > distance_limit:
            drone.reset_state(target)

        # draw particles and body
        drone.draw_particles(screen, dt)
        drone.draw_body(screen)

        # draw target
        pg.draw.circle(screen, (100, 230, 100), (int(target[0]*meters_to_pixels), int(screen_height - target[1]*meters_to_pixels)), 2)
        pg.display.flip()
        time += dt

config = {
    "population": 100,
    "width": 800,
    "height": 600,
    "meters_to_pixels": 15
}

if __name__ == '__main__':
    # pygame setup
    pg.init()
    screen = pg.display.set_mode((config["width"], config["height"]))
    pg.display.set_caption("Visual Training")
    clock = pg.time.Clock()

    # node graph vis
    viz_queue = mp.Queue()
    viz = mp.Process(target=utils.viz_process, args=(viz_queue,))
    viz.start()

    # load saved state
    if utils.save_path.exists():
        state = utils.load()
    else:
        sys.exit()

    drone: Ai_Drone = Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], state['best_drone'])
    viz_queue.put(drone.brain.genome)

    done = False
    while not done:

        # run scorer
        return_code = exhibition(
            drone,
            config["width"],
            config["height"],
            config["meters_to_pixels"],
            screen,
            clock,
        )

        if return_code:
            break

    viz_queue.put(None)
    viz.join()
