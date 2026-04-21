import pygame as pg
import numpy as np
import math
from network_prototype import NeatNN_fast
from genome_prototype import Genome
from numba import njit

def rotate_vector(v, angle):
    # rotation Matrix:
        # x' = x·cos(θ) - y·sin(θ)
        # y' = x·sin(θ) + y·cos(θ)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([v[0]*cos_a - v[1]*sin_a,
                     v[0]*sin_a + v[1]*cos_a])

def normalize(x):
    return math.copysign(math.log1p(abs(x)), x)

@njit
def _nn_inputs(wx, wy, px, py, vx, vy, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    dx = wx - px
    dy = wy - py
    dxlocal = math.copysign(math.log1p(abs(-dx * s + dy * c)), -dx * s + dy * c)
    dylocal = math.copysign(math.log1p(abs( dx * c + dy * s)),  dx * c + dy * s)
    vxlocal = math.copysign(math.log1p(abs(-vx * s + vy * c)), -vx * s + vy * c)
    vylocal = math.copysign(math.log1p(abs( vx * c + vy * s)),  vx * c + vy * s)
    return dxlocal, dylocal, vxlocal, vylocal

@njit
def _apply_outputs(t1turn, t2turn, t1throttle, t2throttle,
                   t1angle, t2angle, rot_speed, angle_min, angle_max, dt):
    t1angle += t1turn * rot_speed * dt
    t2angle += t2turn * rot_speed * dt
    t1angle = min(max(t1angle, angle_min), angle_max)
    t2angle = min(max(t2angle, angle_min), angle_max)
    t1_thrust = max(0.0, t1throttle)
    t2_thrust = max(0.0, t2throttle)
    return t1angle, t2angle, t1_thrust, t2_thrust

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
    def __init__(self, pos, vel, lifetime, starting_a=255):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.starting_a = starting_a

    def update(self, dt):
        self.pos += self.vel * dt
        self.lifetime -= dt

    @property
    def alive(self):
        return self.lifetime > 0

    @property
    def alpha(self):
        return int(self.starting_a * (self.lifetime / self.max_lifetime))

@njit
def _calculate_forces_shit(t1_thrust, t2_thrust, thruster_force, t1angle, t2angle, angle, thruster_offset):
        # f1 rotation
        f1x = -(t1_thrust * thruster_force) * math.sin(t1angle)
        f1y =  (t1_thrust * thruster_force) * math.cos(t1angle)

        # f2 rotation
        f2x = -(t2_thrust * thruster_force) * math.sin(t2angle)
        f2y =  (t2_thrust * thruster_force) * math.cos(t2angle)

        # combined force rotated by drone angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        F0 = (f1x+f2x)*cos_a - (f1y+f2y)*sin_a
        F1 = (f1x+f2x)*sin_a + (f1y+f2y)*cos_a

        # relative tourque forces
        tau1 = -thruster_offset*f1y
        tau2 =  thruster_offset*f2y
        T = tau1 + tau2

        return F0, F1, tau1, tau2, T


@njit
def _update_shit(T, I, av, angle, F0, F1, M, g0, g1, v0, v1, pos0, pos1, dt):
        aa = T / I
        av += aa * dt
        angle += av * dt

        # in-place (no new arrays)
        a0 = F0 / M + g0
        a1 = F1 / M + g1

        v0 += a0 * dt
        v1 += a1 * dt

        pos0 += v0 * dt
        pos1 += v1 * dt

        return aa, av, angle, a0, a1, v0, v1, pos0, pos1


class Drone:
    def __init__(self, pos, meters_to_pixels, surface_height, headless=False):
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
        self.thruster_force = 9.81 * self.M * 1.2

        # particle amount
        self.particle_count = 3

        # surfaces
        if headless:
            self.body_surf = None
            self.thruster = None
        else:
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
        # enabled
        self.enabled = True

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
        # # f1 rotation
        F0, F1, tau1, tau2, T = _calculate_forces_shit(
            self.t1_thrust, self.t2_thrust, self.thruster_force,
            self.t1angle, self.t2angle, self.angle, self.thruster_offset[0]
        )

        self.F[0] = F0
        self.F[1] = F1
        self.tau1 = tau1
        self.tau2 = tau2
        self.T = T

    def update(self, dt):
        self.calculate_forces()

        aa, av, angle, a0, a1, v0, v1, pos0, pos1 = _update_shit(
            self.T, self.I, self.av, self.angle, 
            self.F[0], self.F[1], self.M, 
            self.g[0], self.g[1], 
            self.v[0], self.v[1], 
            self.pos[0], self.pos[1], dt
        )

        self.aa, self.av, self.angle = aa, av, angle
        self.a[0], self.a[1] = a0, a1
        self.v[0], self.v[1] = v0, v1
        self.pos[0], self.pos[1] = pos0, pos1


    def draw_body(self, screen, a=255):
        pos_pix = m_to_pixel_position(self.pos, self.surface_height, self.mtp)

        # thrusetr position calculations
        t1pos = self.pos - rotate_vector(self.thruster_offset, self.angle)
        t2pos = self.pos + rotate_vector(self.thruster_offset, self.angle)

        # draw body
        rotated_body = pg.transform.rotate(self.body_surf, np.rad2deg(self.angle))
        rotated_body.set_alpha(a)
        rect = rotated_body.get_rect(center=pos_pix)
        screen.blit(rotated_body, rect)

        # draw thrusters
        t1pos_pix = m_to_pixel_position(t1pos, self.surface_height, self.mtp)
        t1_rotated = pg.transform.rotate(self.thruster, np.rad2deg(self.t1angle) + np.rad2deg(self.angle))
        t1_rotated.set_alpha(a)
        screen.blit(t1_rotated, t1_rotated.get_rect(center=t1pos_pix))

        t2pos_pix = m_to_pixel_position(t2pos, self.surface_height, self.mtp)
        t2_rotated = pg.transform.rotate(self.thruster, np.rad2deg(self.t2angle) + np.rad2deg(self.angle))
        t2_rotated.set_alpha(a)
        screen.blit(t2_rotated, t2_rotated.get_rect(center=t2pos_pix))

    def draw_particles(self, screen, dt, a=255):
        # thrusetr position calculations
        t1pos = self.pos - rotate_vector(self.thruster_offset, self.angle)
        t2pos = self.pos + rotate_vector(self.thruster_offset, self.angle)

        # draw particles
        if self.t1_thrust:
            self.spawn_particles(t1pos, self.t1angle + self.angle, a)

        if self.t2_thrust:
            self.spawn_particles(t2pos, self.t2angle + self.angle, a)

        self.particles = [p for p in self.particles if p.alive]
        radius = max(1, int(0.07*self.mtp))
        for p in self.particles:
            p.update(dt)  # or pass dt into draw
            pix = m_to_pixel_position(p.pos, self.surface_height, self.mtp)
            surf = pg.Surface((radius*2, radius*2), pg.SRCALPHA)
            pa = max(0, min(255, int(p.alpha)))
            pg.draw.circle(surf, (255, 150, 50, pa), (radius, radius), radius)
            screen.blit(surf, (int(pix[0])-radius, int(pix[1])-radius))

    def spawn_particles(self, thruster_pos, thruster_world_angle, start_a=255):
        # emit downward relative to thruster direction
        direction = rotate_vector(np.array([0, -1]), thruster_world_angle)
        for _ in range(self.particle_count):
            spread = np.random.uniform(-0.1, 0.1)
            speed = np.random.uniform(2, 5)
            vel = rotate_vector(direction * speed + self.v/2, spread)
            self.particles.append(Particle(thruster_pos.copy(), vel, lifetime=0.25, starting_a=start_a))

class Ai_Drone(Drone):
    def __init__(self, pos, meters_to_pixels, surface_height, genome: Genome, headless=False):
        super().__init__(pos, meters_to_pixels, surface_height, headless=headless)

        self.genome = genome
        self.brain = NeatNN_fast(genome)
        self.waypoint: np.ndarray = np.array(pos, dtype=float)

    def reset_state(self, pos):
        return super().reset_state(pos)
        self.enabled = True

    def handle_input(self, keys, dt):
        dxlocal, dylocal, vxlocal, vylocal = _nn_inputs(
            self.waypoint[0], self.waypoint[1],
            self.pos[0], self.pos[1],
            self.v[0], self.v[1], self.angle
        )

        t1turn, t2turn, t1throttle, t2throttle = self.brain.forward(
            delta_x=dxlocal, delta_y=dylocal,
            angle=self.angle, vel_x=vxlocal, vel_y=vylocal,
            angular_vel=self.av, t1_angle=self.t1angle, t2_angle=self.t2angle
        )

        self.t1angle, self.t2angle, self.t1_thrust, self.t2_thrust = _apply_outputs(
            t1turn, t2turn, t1throttle, t2throttle,
            self.t1angle, self.t2angle,
            self.thruster_rotation_speed,
            self.thruster_max_angle[0], self.thruster_max_angle[1], dt
        )
