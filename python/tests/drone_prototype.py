import pygame as pg
import numpy as np
from network_prototype import NeatNN
from genome_prototype import Genome

def rotate_vector(v, angle):
    # rotation Matrix:
        # x' = x·cos(θ) - y·sin(θ)
        # y' = x·sin(θ) + y·cos(θ)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a, cos_a]])
    return R @ v

def m_to_pixel_position(position: np.ndarray, surface_height, meters_to_pixels):
    position = position * meters_to_pixels
    return np.array([position[0], surface_height - position[1]])

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

class Particle:
    def __init__(self, pos, vel, lifetime):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self, dt):
        self.pos += self.vel * dt
        self.lifetime -= dt

    @property
    def alive(self):
        return self.lifetime > 0

    @property
    def alpha(self):
        return int(255 * (self.lifetime / self.max_lifetime))

class Drone:
    def __init__(self, pos, meters_to_pixels, surface_height):
        # set blank state
        self.reset_state(pos)

        # constants
        self.mtp = meters_to_pixels
        self.surface_height = surface_height
        self.size = (2, 0.175) # Meters

        self.g = np.array([0.0, -9.81])
        self.M = 4     # Kg
        self.I = 2.383 # Kg*m2

        self.thruster_offset = np.array([self.size[0] / 2, 0.0])
        self.thruster_rotation_speed = np.deg2rad(120)
        self.thruster_max_angle = (np.deg2rad(-60), np.deg2rad(60))
        self.thruster_force = 9.81 * self.M * 0.9

        # surfaces
        self.body_surf = create_drone(self.size[0], self.size[1], self.mtp)
        self.thruster = create_thruster(self.size[1] * 2, self.size[1] * 2.2, (175, 175, 175), self.mtp)

    def reset_state(self, pos):
        # state
        self.F = np.array([0.0, 0.0]) # Net force
        self.T = 0.0                  # Net tourque
        # Metres
        self.pos = np.array(pos, dtype=float)
        self.v = np.array([0.0, 0.0])
        self.a = np.array([0.0, 0.0])
        # Rads
        self.angle = 0.0
        self.av = 0.0
        self.aa = 0.0
        # thruster angles (relative to drone)
        self.t1angle = 0.0
        self.t2angle = 0.0
        # thruster thrusts
        self.t1_thrust = 0.0
        self.t2_thrust = 0.0
        # particles
        self.particles = []

    def handle_input(self, keys, dt):
        if keys[pg.K_a]:
            self.t1angle += self.thruster_rotation_speed * dt
        if keys[pg.K_s]:
            self.t1angle -= self.thruster_rotation_speed * dt
        self.t1angle = np.clip(self.t1angle, *self.thruster_max_angle)
        if keys[pg.K_w]:
            self.t1_thrust = 1
        else:
            self.t1_thrust = 0

        if keys[pg.K_LEFT]:
            self.t2angle += self.thruster_rotation_speed * dt
        if keys[pg.K_RIGHT]:
            self.t2angle -= self.thruster_rotation_speed * dt
        self.t2angle = np.clip(self.t2angle, *self.thruster_max_angle)
        if keys[pg.K_UP]:
            self.t2_thrust = 1
        else:
            self.t2_thrust = 0

    def calculate_forces(self):
        # relative thruster forces
        f1 = rotate_vector(np.array([0, self.t1_thrust * self.thruster_force]), self.t1angle)
        f2 = rotate_vector(np.array([0, self.t2_thrust * self.thruster_force]), self.t2angle)
        self.F = rotate_vector(f1 + f2, self.angle)

        # relative tourque forces
        self.tau1 = np.cross(-self.thruster_offset, f1)
        self.tau2 = np.cross(self.thruster_offset, f2)
        self.T = self.tau1 + self.tau2

    def update(self, dt):
        self.calculate_forces()

        self.aa = self.T / self.I
        self.av += self.aa * dt
        self.angle += self.av * dt

        self.a = (self.F / self.M) + self.g
        self.v += self.a * dt
        self.pos += self.v * dt

    def draw(self, screen, dt):
        pos_pix = m_to_pixel_position(self.pos, self.surface_height, self.mtp)

        # thrusetr position calculations
        t1pos = self.pos - rotate_vector(self.thruster_offset, self.angle)
        t2pos = self.pos + rotate_vector(self.thruster_offset, self.angle)

        # draw particles
        if self.t1_thrust:
            self.spawn_particles(t1pos, self.t1angle + self.angle)

        if self.t2_thrust:
            self.spawn_particles(t2pos, self.t2angle + self.angle)

        self.particles = [p for p in self.particles if p.alive]
        for p in self.particles:
            p.update(dt)  # or pass dt into draw
            pix = m_to_pixel_position(p.pos, self.surface_height, self.mtp)
            color = (255, 150, 50, p.alpha)
            pg.draw.circle(screen, color[:3], tuple(pix.astype(int)), max(1, int(0.07*self.mtp)))

        # draw body
        rotated_body = pg.transform.rotate(self.body_surf, np.rad2deg(self.angle))
        rect = rotated_body.get_rect(center=pos_pix)
        screen.blit(rotated_body, rect)

        # draw thrusters
        t1pos_pix = m_to_pixel_position(t1pos, self.surface_height, self.mtp)
        t1_rotated = pg.transform.rotate(self.thruster, np.rad2deg(self.t1angle) + np.rad2deg(self.angle))
        screen.blit(t1_rotated, t1_rotated.get_rect(center=t1pos_pix))

        t2pos_pix = m_to_pixel_position(t2pos, self.surface_height, self.mtp)
        t2_rotated = pg.transform.rotate(self.thruster, np.rad2deg(self.t2angle) + np.rad2deg(self.angle))
        screen.blit(t2_rotated, t2_rotated.get_rect(center=t2pos_pix))
    
    def spawn_particles(self, thruster_pos, thruster_world_angle):
        # emit downward relative to thruster direction
        direction = rotate_vector(np.array([0, -1]), thruster_world_angle)
        for _ in range(2):  # spawn 3 per frame
            spread = np.random.uniform(-0.1, 0.1)
            speed = np.random.uniform(2, 5)
            vel = rotate_vector(direction * speed + self.v/2, spread)
            self.particles.append(Particle(thruster_pos.copy(), vel, lifetime=0.25))

class Ai_Drone(Drone):
    def __init__(self, pos, meters_to_pixels, surface_height, genome: Genome):
        super().__init__(pos, meters_to_pixels, surface_height)
        
        self.brain = NeatNN(genome)
        self.waypoint: np.ndarray = np.array(pos, dtype=float)

    def handle_input(self, keys, dt):
        # pass data through brain
        outputs = self.brain.forward(
            delta_x = self.waypoint[0] - self.pos[0],
            delta_y = self.waypoint[1] - self.pos[1],
            angle = self.angle,
            vel_x = self.v[0],
            vel_y = self.v[1],
            angular_vel = self.av,
            t1_angle = self.t1angle,
            t2_angle = self.t2angle
        )

        t1turn, t2turn, t1throttle, t2throttle = outputs

        # set thruster angles
        self.t1angle += t1turn * self.thruster_rotation_speed * dt
        self.t2angle += t2turn * self.thruster_rotation_speed * dt
        
        self.t1angle = np.clip(self.t1angle, *self.thruster_max_angle)
        self.t2angle = np.clip(self.t2angle, *self.thruster_max_angle)

        # set thruter throttles
        self.t1_thrust = max(0, t1throttle)
        self.t2_thrust = max(0, t2throttle)