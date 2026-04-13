from drone_prototype import Ai_Drone
from numba import njit
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

    # Lower difficulty → higher weight. Square the linear gap for strong convergence pressure.
    max_diff = diffs.max()
    gap = max_diff - diffs                             # gap=0 for hardest, largest for easiest
    weights = (gap + max(max_diff * 0.05, 1.0)) ** 2   # floor ensures hardest direction still gets picked (rarely)
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
        s['difficulty'] *= error + 1
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

@njit
def math_shit(p, v, target, eps, max_a, dt, limit, t1_thrust, t2_thrust, prev_t1, prev_t2):
    v_mag = math.hypot(v[0], v[1])

    r = target - p
    d = math.hypot(r[0], r[1])
    u = r / (d + eps)

    v_par   = float(np.dot(v, u))
    safe_v  = math.sqrt(2 * max_a * d)
    v_ratio = v_par / (safe_v + eps)

    # ── v_max parabola ────────────────────────────────────
    if v_ratio <= 1:
        frame_score = dt * (1 - (v_ratio - 1) ** 2) / limit
    else:
        frame_score = dt * (1 - 16 * (v_ratio - 1) ** 2) / limit

    # ── Feathering penalty ────────────────────────────────
    if frame_score > 0 and d >= BRAKE_ZONE:
        f1 = max(1 - FEATHER_K * abs(t1_thrust - prev_t1), 0)
        f2 = max(1 - FEATHER_K * abs(t2_thrust - prev_t2), 0)
        frame_score *= f1 * f2

    return v_mag, d, frame_score


def stage1_vmax_test(
    genomes,
    screen_width: int,
    screen_height: int,
    meters_to_pixels: float,
    limit: float = 10,
    diff: float = 10,
):
    """Stage 1 vmax test: runs all 8 compass directions simultaneously.
    Each genome gets 8 drone instances (one per direction). Scores are averaged."""
    eps = 1e-8
    dt  = 0.016
    N   = len(genomes)
    D   = len(DIR_NAMES)
    total = D * N

    center = np.array((
        screen_width  / (2 * meters_to_pixels),
        screen_height / (2 * meters_to_pixels),
    ))
    target = center.copy()

    # ── Per-direction setup ────────────────────────────────────────────
    spawns      = []
    oob_centers = []
    oob_radii   = []

    d_initial  = diff
    base_bonus = diff * 4

    for d_idx in range(D):
        theta = DIR_ANGLES[d_idx]
        spawn = center + diff * np.array([math.cos(theta), math.sin(theta)])
        spawns.append(spawn)
        oob_centers.append((spawn + target) / 2)
        oob_radii.append(d_initial * 1.5 / 2)

    # ── Create 8*N drones ──────────────────────────────────────────────
    drones = []
    for d_idx in range(D):
        for genome in genomes:
            drones.append(Ai_Drone((0, 0), meters_to_pixels, screen_height, genome, headless=True))

    # First drone for physics constants
    max_a = drones[0].thruster_force * 2 / drones[0].M

    # ── Per-drone state ────────────────────────────────────────────────
    vmax_scores = np.zeros(total)
    comp_scores = np.zeros(total)
    hover_time  = np.zeros(total)
    total_dist  = np.zeros(total)
    prev_t1     = np.zeros(total)
    prev_t2     = np.zeros(total)
    completed: set[int] = set()  # flat indices

    # Init drones to their direction's spawn
    for ix, drone in enumerate(drones):
        d_idx = ix // N
        drone.reset_state(spawns[d_idx])
        drone.waypoint = target.copy()

    time = 0.0
    completions: list[float] = []

    # ── Simulation loop ────────────────────────────────────────────────
    while time < limit:
        if not any(d.enabled for d in drones):
            break

        for ix, drone in enumerate(drones):
            if not drone.enabled:
                continue

            d_idx = ix // N

            drone.handle_input(None, dt)
            drone.update(dt)

            p, v = drone.pos, drone.v
            v_mag, d, frame_score = math_shit(
                p, v, target, eps, max_a, dt, limit, drone.t1_thrust, drone.t2_thrust, prev_t1[ix], prev_t2[ix]
            ) # type: ignore

            # ── Hover dwell ───────────────────────────────────────
            if d < HOVER_DIST and v_mag < HOVER_VEL:
                hover_time[ix] += dt
            else:
                hover_time[ix] = max(hover_time[ix] - dt, 0.0)

            # ── Completion ────────────────────────────────────────
            if hover_time[ix] > 0.1 * limit:
                eff = math.sqrt(max(d_initial, eps) / max(total_dist[ix], d_initial, eps))
                comp_scores[ix] = base_bonus * eff
                drone.enabled = False
                completions.append(time)
                completed.add(ix)
                continue

            total_dist[ix] += v_mag * dt

            # ── Out of bounds ─────────────────────────────────────
            dist_oob = math.hypot(p[0] - oob_centers[d_idx][0], p[1] - oob_centers[d_idx][1])
            if dist_oob > oob_radii[d_idx]:
                drone.enabled = False
                if vmax_scores[ix] > 0:
                    vmax_scores[ix] /= 2

            vmax_scores[ix] = max(vmax_scores[ix] + frame_score, 0.0)

            prev_t1[ix], prev_t2[ix] = drone.t1_thrust, drone.t2_thrust

        time += dt

    # ── Normalize components to 0-1 per drone ───────────────────────────
    #   completion theoretical max = base_bonus
    scores = np.zeros(total)
    for d_idx in range(D):
        sl = slice(d_idx * N, (d_idx + 1) * N)
        comp_norm = comp_scores[sl] / (base_bonus + eps)
        scores[sl] = 0.5 * np.clip(vmax_scores[sl], 0, 1) + 0.5 * np.clip(comp_norm, 0, 1)

    # ── Weighted min/mean across 8 directions per genome ────────────────
    dir_scores = np.zeros((D, N))
    for d_idx in range(D):
        dir_scores[d_idx] = scores[d_idx * N : (d_idx + 1) * N]
    genome_scores = 0.9 * dir_scores.mean(axis=0) + 0.1 * dir_scores.min(axis=0)

    # ── Difficulty scaling for cross-generation comparison ─────────────
    #   log(1+diff)/log(1+base) → 1.0× at 10m, grows with difficulty
    BASE_DIFF = 10.0
    diff_scale = math.log(1 + diff) / math.log(1 + BASE_DIFF)
    genome_scores *= diff_scale

    # Average completion count: total completions / 8 directions
    avg_completions = len(completed) / D

    return 0, genome_scores, completions, avg_completions
