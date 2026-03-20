from drone_prototype import Ai_Drone
import math
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
LATERAL_K   = 1.5       # lateral velocity penalty scale (increase over training)
FEATHER_K   = 0.25      # thrust-change penalty coefficient
HOVER_DIST  = 0.5       # meters — hover transition boundary
HOVER_VEL   = 0.5       # m/s — max velocity to count as hovering
HOVER_REWARD = 3.0      # per-second reward inside hover zone
BRAKE_ZONE  = 5.0       # meters — feathering disabled inside this

# ── 8 compass directions for balanced training ───────────────────────────────
DIRECTIONS = {
    'N':  np.pi / 2,
    'NE': np.pi / 4,
    'E':  0.0,
    'SE': -np.pi / 4,
    'S':  -np.pi / 2,
    'SW': -3 * np.pi / 4,
    'W':  np.pi,
    'NW': 3 * np.pi / 4,
}
DIR_NAMES  = list(DIRECTIONS.keys())
DIR_ANGLES = list(DIRECTIONS.values())


DEFAULT_DIFFICULTY = 15.0


def pick_direction(dir_stats: dict) -> tuple[str, float, float]:
    """Pick a direction weighted toward lowest difficulty (weakest).

    Returns (dir_name, theta, difficulty).
    """
    diffs = np.array([dir_stats[name]['difficulty'] for name in DIR_NAMES])

    # Lower difficulty → higher weight.  Invert relative to max.
    max_diff = diffs.max()
    # weight = max_diff - diff + floor  (floor keeps mastered directions in rotation)
    weights = max_diff - diffs + max(max_diff * 0.1, 1.0)
    weights /= weights.sum()

    idx = int(np.random.choice(len(DIR_NAMES), p=weights))
    name = DIR_NAMES[idx]
    return name, DIR_ANGLES[idx], dir_stats[name]['difficulty']


def adjust_dir_difficulty(dir_stats: dict, dir_name: str, completions: int, pop: int,
                          target_rate: float = 0.1):
    """Adjust difficulty for a single direction based on its completion rate."""
    s = dir_stats[dir_name]
    rate = completions / pop
    error = rate - target_rate
    if abs(error) > 0.02:
        s['difficulty'] *= np.sqrt(error + 1)
        s['difficulty'] = max(s['difficulty'], 10)


def format_dir_rates(dir_stats: dict) -> str:
    parts = []
    for name in DIR_NAMES:
        s = dir_stats[name]
        parts.append(f"{name}:{s['difficulty']:.1f}m")
    return " ".join(parts)


def make_dir_stats(initial_difficulty: float = DEFAULT_DIFFICULTY) -> dict:
    """Create a fresh direction-stats dict with per-direction difficulty."""
    return {name: {'difficulty': initial_difficulty} for name in DIR_NAMES}


# ── Stage 1: static target acquisition (headless) ───────────────────────────
def stage1(
    drones: list[Ai_Drone],
    screen_width: int,
    screen_height: int,
    meters_to_pixels: float,
    limit: float = 10,
    diff: float = 10,
    theta: float | None = None,
):
    eps = 1e-8
    dt  = 0.016
    N   = len(drones)

    # ── World setup ──────────────────────────────────────────────────────
    center = np.array((
        screen_width  / (2 * meters_to_pixels),
        screen_height / (2 * meters_to_pixels),
    ))
    target = center.copy()

    # Spawn direction
    if theta is None:
        theta = 2 * math.pi * np.random.rand()
    spawn = center + diff * np.array([math.cos(theta), math.sin(theta)])

    # Drone physics
    max_a = drones[0].thruster_force * 2 / drones[0].M

    # Scoring constants — derived from drone physics, never hardcoded
    d_initial    = math.hypot(target[0] - spawn[0], target[1] - spawn[1])
    base_bonus   = max(400, d_initial * 4)
    safe_v_brake = math.sqrt(2 * max_a * BRAKE_ZONE)

    # Out-of-bounds: circle centered on midpoint of spawn↔target
    oob_center = (spawn + target) / 2
    oob_radius = d_initial * 1.1 / 2

    # Hover transition: parabola peak value at boundary (v_ratio=1 → score=1.0)
    HOVER_BOUNDARY_SCORE = 1.0

    # ── Per-drone state ──────────────────────────────────────────────────
    scores      = np.zeros(N)
    hover_time  = np.zeros(N)
    total_dist  = np.zeros(N)
    entry_speed = np.full(N, -1.0)
    prev_t1     = np.zeros(N)
    prev_t2     = np.zeros(N)
    in_hover    = np.zeros(N, dtype=bool)

    # Drone init
    for drone in drones:
        drone.reset_state(spawn)
        drone.waypoint = target.copy()

    time = 0.0
    completions: list[float] = []
    completed: set[int] = set()

    # ── Simulation loop ──────────────────────────────────────────────────
    while time < limit:
        if not any(d.enabled for d in drones):
            break

        for ix, drone in enumerate(drones):
            if not drone.enabled:
                continue

            drone.handle_input(None, dt)
            drone.update(dt)

            # ── Vector math ──────────────────────────────────────────
            p, v = drone.pos, drone.v
            v_mag = math.hypot(v[0], v[1])

            r = target - p
            d = math.hypot(r[0], r[1])
            u = r / (d + eps)

            v_par   = float(np.dot(v, u))       # signed closing speed
            v_par_v = v_par * u
            v_perp_v = v - v_par_v
            v_perp  = math.hypot(v_perp_v[0], v_perp_v[1])

            safe_v  = math.sqrt(2 * max_a * d)
            v_ratio = v_par / (safe_v + eps)

            # ── Scoring ──────────────────────────────────────────────
            if d >= HOVER_DIST:
                # === PURSUIT MODE ===
                # Core: asymmetric parabola
                if v_ratio <= 1:
                    frame_score = dt * (1 - (v_ratio - 1) ** 2)
                else:
                    frame_score = dt * (1 - 16 * (v_ratio - 1) ** 2)

                # Lateral penalty
                prox_factor = 1 + LATERAL_K / max(d, 0.1)
                frame_score -= dt * v_perp * prox_factor

                # Feathering multiplier (positive scores only, outside brake zone)
                if frame_score > 0 and d >= BRAKE_ZONE:
                    thrust_change = (abs(drone.t1_thrust - prev_t1[ix])
                                   + abs(drone.t2_thrust - prev_t2[ix]))
                    frame_score *= max(1 - FEATHER_K * thrust_change, 0)
            else:
                # === HOVER TRANSITION ===
                # v_ratio OFF — smooth transition to proximity reward
                proximity   = 1.0 - (d / HOVER_DIST)
                frame_score = dt * (HOVER_BOUNDARY_SCORE + proximity * HOVER_REWARD)

                # Penalize velocity — should be stopping
                frame_score -= dt * v_mag * 3.0

            # ── Hover zone (stopped at target) ───────────────────────
            if d < HOVER_DIST and v_mag < HOVER_VEL:
                in_hover[ix] = True
                hover_time[ix] += dt
            else:
                in_hover[ix] = False
                if hover_time[ix] > 0:
                    hover_time[ix] -= dt

            # ── Completion ───────────────────────────────────────────
            if hover_time[ix] > 0.1 * limit:
                # Efficiency (applied here since drone is about to be disabled)
                if scores[ix] + frame_score > 0:
                    eff = math.sqrt(max(d_initial, eps) / max(total_dist[ix], d_initial, eps))
                    scores[ix] = (scores[ix] + frame_score) * eff
                    frame_score = 0  # already folded in

                stop_precision = 1 / (1 + d)
                e_bonus = max(entry_speed[ix], 0) / safe_v_brake

                scores[ix] += base_bonus * stop_precision
                scores[ix] += base_bonus * (1 - time / limit)
                scores[ix] += base_bonus * 0.5 * min(e_bonus, 1.0)

                drone.enabled = False
                completions.append(time)
                completed.add(ix)
                prev_t1[ix], prev_t2[ix] = drone.t1_thrust, drone.t2_thrust
                continue

            # ── Path tracking (skip in hover) ────────────────────────
            if not in_hover[ix]:
                total_dist[ix] += v_mag * dt

            # ── Entry speed (first time inside brake zone) ───────────
            if d < BRAKE_ZONE and entry_speed[ix] < 0:
                entry_speed[ix] = max(v_par, 0)

            # ── Out of bounds ────────────────────────────────────────
            dist_oob = math.hypot(p[0] - oob_center[0], p[1] - oob_center[1])
            if dist_oob > oob_radius:
                drone.enabled = False
                if scores[ix] > 0:
                    scores[ix] /= 2

            # ── Accumulate ───────────────────────────────────────────
            scores[ix] += frame_score
            scores[ix] = max(scores[ix], 0)

            # ── Update previous thrust ───────────────────────────────
            prev_t1[ix], prev_t2[ix] = drone.t1_thrust, drone.t2_thrust

        time += dt

    # ── End of sim: apply efficiency to ALL non-completed drones ─────────
    for ix in range(N):
        if scores[ix] > 0 and ix not in completed:
            eff = math.sqrt(max(d_initial, eps) / max(total_dist[ix], d_initial, eps))
            scores[ix] *= eff

    return 0, scores, completions
