import pygame as pg
import sys

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


    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        drone_pos = (400, 300)
        drone = pg.Rect(drone_pos, (30, 30))

        screen.fill((20, 20, 20))  # Black background
        pg.draw.rect(screen, (220, 220, 220), drone)
        pg.draw.circle(screen, (230, 100, 100), (drone_pos[0]+15, drone_pos[1]), 5)  # Example target
        pg.display.flip()
        clock.tick(60)

    pg.quit()
    sys.exit()

if __name__ == "__main__":
    main()