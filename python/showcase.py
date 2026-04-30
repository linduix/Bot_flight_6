from drone import Ai_Drone
import util as utils
import multiprocessing as mp
import pygame as pg
import numpy as np
import math
import sys

def exhibition(drones, screen_width, screen_height, meters_to_pixels, screen, clock):
    font = pg.font.SysFont(None, 28)
    label_font = pg.font.SysFont(None, 22)
    # fixed target in center
    target = np.array((screen_width/(2*meters_to_pixels), screen_height/(2*meters_to_pixels)))

    # initialize all drones
    for name, drone, color, alpha in drones:
        drone.reset_state(target)
        drone.waypoint = target.copy()

    time = 0.0
    distance_limit = np.sqrt(screen_height**2 + screen_width**2) / meters_to_pixels

    while True:
        # update target
        target = pg.mouse.get_pos()
        target = (target[0] / meters_to_pixels, (screen_height - target[1]) / meters_to_pixels)
        target_arr = np.array(target)

        # visual mode clock
        dt = clock.tick(60) / 1000
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return 1

        screen.fill((20, 20, 20))

        for name, drone, color, alpha in drones:
            drone.waypoint = target_arr.copy()

            # update drone
            drone.handle_input(None, dt)
            drone.update(dt)

            # disable if too far
            dx = drone.pos[0] - target[0]
            dy = drone.pos[1] - target[1]
            dist = math.hypot(dx, dy)
            if dist > distance_limit:
                drone.reset_state(target_arr)

            # draw particles and body
            drone.draw_particles(screen, dt, a=alpha)
            drone.draw_body(screen, a=alpha)

            # draw label above drone (skip for F1 — too many overlap)
            if not name.startswith("F1"):
                px = int(drone.pos[0] * meters_to_pixels)
                py = int(screen_height - drone.pos[1] * meters_to_pixels) - 30
                label_surf = label_font.render(name, True, color)
                label_rect = label_surf.get_rect(center=(px, py))
                screen.blit(label_surf, label_rect)

        # draw target
        pg.draw.circle(screen, (100, 230, 100), (int(target[0]*meters_to_pixels), int(screen_height - target[1]*meters_to_pixels)), 2)

        # HUD
        info_y = 10
        for name, drone, color, alpha in drones:
            if name.startswith("F1"):
                continue
            dx = drone.pos[0] - target[0]
            dy = drone.pos[1] - target[1]
            dist = math.hypot(dx, dy)
            dist_text = font.render(f"{name}: {dist:.2f} m", True, color)
            screen.blit(dist_text, (10, info_y))
            info_y += 26

        fps_text = font.render(f"FPS: {clock.get_fps():.0f}", True, (150, 150, 150))
        screen.blit(fps_text, (screen_width - 100, 10))

        pg.display.flip()
        time += dt

config = {
    "population": 100,
    "width": 800,
    "height": 600,
    "meters_to_pixels": 10
}

best_path = utils.checkpoint_dir / "prototype_best.pkl"

if __name__ == '__main__':
    # pygame setup
    pg.init()
    screen = pg.display.set_mode((config["width"], config["height"]))
    pg.display.set_caption("Showcase")
    clock = pg.time.Clock()

    # node graph vis
    viz_queue = mp.Queue()
    viz = mp.Process(target=utils.viz_process, args=(viz_queue,))
    viz.start()

    # load both checkpoints
    drones = []

    if best_path.exists():
        best_state = utils.load(best_path)
        best_drone = Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], best_state['best_drone'])
        drones.append(("Best Save", best_drone, (100, 230, 130), 255))
        viz_queue.put(best_drone.genome)
        print(f"Best save loaded (gen {best_state['gen']})")

    if utils.save_path.exists():
        current_state = utils.load()
        top = current_state['current_gen'][:10]
        for i, g in enumerate(top):
            d = Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], g)
            drones.append((f"F1[{i}]", d, (100, 160, 230), 70))
        print(f"Current save loaded (gen {current_state['gen']}, showing top {len(top)} of current_gen)")

    if not drones:
        print("No checkpoints found.")
        sys.exit()

    done = False
    while not done:
        return_code = exhibition(
            drones,
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
