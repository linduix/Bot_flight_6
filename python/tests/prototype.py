import pygame as pg
import numpy as np
import sys

def create_drone(width, height, meters_to_pixels):
    width *= meters_to_pixels
    height *= meters_to_pixels
    scale = 10
    pad = int(height*1.75*2)
    big = pg.Surface((int(width*scale), int(pad*scale)), pg.SRCALPHA)
    
    # body rectangle
    pg.draw.rect(big, (220, 220, 220), (0, int((pad//2-height//2)*scale), int(width*scale), int(height*scale)))
    
    # center circle
    pg.draw.circle(big, (220, 220, 220), (int(width*scale//2), int(pad*scale//2)), int(height*1.75*scale))
    
    # top circle
    pg.draw.circle(big, (230, 100, 100), (int(width*scale//2), int((pad//2-height//1.5)*scale)), int(height*0.4*scale))
    
    surface = pg.transform.smoothscale(big, (int(width), int(pad)))
    return surface

def create_thruster(width, height, color, meters_to_pixels):
    scale = 10
    width *= meters_to_pixels
    height *= meters_to_pixels
    big = pg.Surface((int(width*scale), int(height*scale)), pg.SRCALPHA)
    points = [
        (0, 0),
        (int(width*scale), 0),
        (int(width*scale * 3//4), int(height*scale)),
        (int(width*scale * 1//4), int(height*scale)),
    ]
    pg.draw.polygon(big, color, points)
    surface = pg.transform.smoothscale(big, (int(width), int(height)))
    return surface

def main():
    pg.init()
    config = {
        "width": 800,
        "height": 600,
        "caption": "BotFlight6 Visualization"
    }
    screen = pg.display.set_mode((config["width"], config["height"]))
    pg.display.set_caption(config["caption"])
    clock = pg.time.Clock()

    meters_to_pixels = 50
    drone_size = (2, .175)

    # create once
    drone_surface = create_drone(drone_size[0], drone_size[1], meters_to_pixels)
    thruster1 = create_thruster(drone_size[1]*2, drone_size[1]*2.2, (150,150,150), meters_to_pixels)
    thruster2 = create_thruster(drone_size[1]*2, drone_size[1]*2.2, (150,150,150), meters_to_pixels)

    thruster1angle = 0
    thruster2angle = 0

    drone_pos_m = np.array([config["width"]/(2*meters_to_pixels), config["height"]/(2*meters_to_pixels)])
    drone_v = np.array([0.0, 0.0])
    drone_a = np.array([0.0, 0.0])

    g = np.array([0, -9.81])


    dt = 0
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        keys = pg.key.get_pressed()

        if keys[pg.K_a]:
            thruster1angle += 120*dt
        if keys[pg.K_w]:
            pass
        if keys[pg.K_s]:
            thruster1angle -= 120*dt

        if keys[pg.K_LEFT]:
            thruster2angle += 120*dt
        if keys[pg.K_UP]:
            pass
        if keys[pg.K_RIGHT]:
            thruster2angle -= 120*dt

        drone_a = g
        drone_v += drone_a * dt
        drone_pos_m += drone_v * dt

        drone_pos_pix = (drone_pos_m[0]*meters_to_pixels, config["height"]-drone_pos_m[1]*meters_to_pixels)

        # Black background
        screen.fill((20, 20, 20))  

        # draw each frame
        rect = drone_surface.get_rect(center=(drone_pos_pix[0], drone_pos_pix[1]))
        screen.blit(drone_surface, rect)

        thruster1_rotated = pg.transform.rotate(thruster1, thruster1angle)
        truster1rect = thruster1_rotated.get_rect(center=(drone_pos_pix[0]-drone_size[0]*meters_to_pixels/2, drone_pos_pix[1]))
        screen.blit(thruster1_rotated, truster1rect)

        thruster2_rotated = pg.transform.rotate(thruster2, thruster2angle)
        truster2rect = thruster2_rotated.get_rect(center=(drone_pos_pix[0]+drone_size[0]*meters_to_pixels/2, drone_pos_pix[1]))
        screen.blit(thruster2_rotated, truster2rect)

        pg.display.flip()
        dt = clock.tick(60) / 1000

    pg.quit()
    sys.exit()

if __name__ == "__main__":
    main()