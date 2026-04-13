from drone_prototype import Ai_Drone
from prototype_stage1 import HOVER_DIST, HOVER_VEL, FEATHER_K, BRAKE_ZONE
import math
import numpy as np
from prototype_stage1 import math_shit

# ── Constants ────────────────────────────────────────────────────────────────
CAPTURE_RADIUS    = 0.5     # meters — touch-to-capture for non-dwell waypoints
NUM_WAYPOINTS     = 5
NUM_CHAINS        = 4       # chains in pool, all tested every gen
MIN_LEG_DIST      = 2.0     # meters — minimum distance between consecutive waypoints
REF_MAX_DIST      = 10.0    # meters — reference max leg for pattern generation (scaled to difficulty)
HOVER_DWELL_TIME  = 0.5     # seconds — hover requirement at the dwell waypoint
POOL_REFRESH_GENS = 100     # regenerate chain pool every N generations


def generate_chain(rng, origin, num_waypoints, min_dist, max_dist):
    """Generate a chain of waypoints starting from origin.
    Each waypoint is a random direction and uniform distance [min_dist, max_dist]
    from the previous point. Uses the provided rng for reproducibility."""
    chain = []
    prev = origin.copy()
    for _ in range(num_waypoints):
        theta = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(min_dist, max_dist)
        wp = prev + dist * np.array([math.cos(theta), math.sin(theta)])
        chain.append(wp)
        prev = wp
    return chain


def _scale_chain(chain, origin, target_total):
    """Proportionally scale leg distances so chain total path length = target_total.
    Preserves angles and relative leg proportions (same pattern, different size)."""
    raw_total = 0.0
    prev = origin
    for wp in chain:
        raw_total += np.linalg.norm(wp - prev)
        prev = wp
    if raw_total < 1e-8:
        return chain
    scale = target_total / raw_total
    scaled = []
    prev_raw = origin.copy()
    prev_scaled = origin.copy()
    for wp in chain:
        delta = wp - prev_raw
        new_pos = prev_scaled + delta * scale
        scaled.append(new_pos)
        prev_scaled = new_pos
        prev_raw = wp
    return scaled

# ── Stage 2: sequential waypoint acquisition (headless) ────────────────────
def stage2_vmax_test(
    genomes,
    screen_width: int,
    screen_height: int,
    meters_to_pixels: float,
    limit: float = 15,
    diff: float = 10,
    seed: int = 0,
):
    eps = 1e-8
    dt  = 0.016
    N   = len(genomes)
    C   = NUM_CHAINS
    W   = NUM_WAYPOINTS
    total = C * N

    # ── World setup ──────────────────────────────────────────────────────
    center = np.array((
        screen_width  / (2 * meters_to_pixels),
        screen_height / (2 * meters_to_pixels),
    ))

    # ── Seed-based chain generation (deterministic across MP chunks) ─────
    # Generate at fixed reference scale so same seed = same pattern regardless
    # of difficulty, then proportionally scale total chain length to match diff.
    rng = np.random.default_rng(seed)
    chains_raw = [generate_chain(rng, center, W, MIN_LEG_DIST, REF_MAX_DIST) for _ in range(C)]
    chains = [_scale_chain(ch, center, diff) for ch in chains_raw]
    dwell_idx = [int(rng.integers(0, W)) for _ in range(C)]

    # Average leg distance across all chains
    all_legs = []
    for chain in chains:
        prev = center
        for wp in chain:
            all_legs.append(np.linalg.norm(wp - prev))
            prev = wp
    avg_leg_dist = float(np.mean(all_legs))

    # ── Create C*N drones ────────────────────────────────────────────────
    # Layout: [chain0_genome0, chain0_genome1, ..., chain1_genome0, ...]
    drones = []
    for c_idx in range(C):
        for genome in genomes:
            drones.append(Ai_Drone((0, 0), meters_to_pixels, screen_height, genome, headless=True))

    max_a = drones[0].thruster_force * 2 / drones[0].M

    # ── Per-drone state ──────────────────────────────────────────────────
    vmax_scores     = np.zeros(total)
    comp_scores     = np.zeros(total)
    hover_time      = np.zeros(total)
    leg_dist_actual = np.zeros(total)
    leg_dist_ideal  = np.zeros(total)
    prev_t1         = np.zeros(total)
    prev_t2         = np.zeros(total)
    current_wp      = np.zeros(total, dtype=int)
    completed: set[int] = set()

    # ── Init drones at center, waypoint = first in chain ─────────────────
    for ix, drone in enumerate(drones):
        c_idx = ix // N
        drone.reset_state(center.copy())
        drone.waypoint = chains[c_idx][0].copy()
        leg_dist_ideal[ix] = np.linalg.norm(chains[c_idx][0] - center)

    time_elapsed = 0.0
    completions: list[float] = []

    # ── Simulation loop ──────────────────────────────────────────────────
    while time_elapsed < limit:
        if not any(d.enabled for d in drones):
            break

        for ix, drone in enumerate(drones):
            if not drone.enabled:
                continue

            c_idx = ix // N
            wp_idx = current_wp[ix]

            drone.handle_input(None, dt)
            drone.update(dt)

            # ── Vector math ──────────────────────────────────────────
            p, v = drone.pos, drone.v
            target = chains[c_idx][wp_idx]

            v_mag, d, frame_vmax = math_shit(
                p, v, target, eps, max_a, dt, limit, drone.t1_thrust, drone.t2_thrust, prev_t1[ix], prev_t2[ix]
            ) # type: ignore

            vmax_scores[ix] = max(vmax_scores[ix] + frame_vmax, 0.0)

            # ── Track leg distance ───────────────────────────────────
            leg_dist_actual[ix] += v_mag * dt

            # ── Waypoint logic ───────────────────────────────────────
            is_dwell = (wp_idx == dwell_idx[c_idx])

            if is_dwell:
                # Dwell waypoint: hover to capture
                if d < HOVER_DIST and v_mag < HOVER_VEL:
                    hover_time[ix] += dt
                else:
                    hover_time[ix] = max(hover_time[ix] - dt, 0.0)

                if hover_time[ix] >= HOVER_DWELL_TIME:
                    eff = math.sqrt(max(leg_dist_ideal[ix], eps) / max(leg_dist_actual[ix], leg_dist_ideal[ix], eps))
                    comp_scores[ix] += 1.0 * eff
                    hover_time[ix] = 0.0

                    # Advance or complete
                    current_wp[ix] += 1
                    if current_wp[ix] >= W:
                        drone.enabled = False
                        completions.append(time_elapsed)
                        completed.add(ix)
                    else:
                        next_target = chains[c_idx][current_wp[ix]]
                        leg_dist_ideal[ix] = np.linalg.norm(next_target - target)
                        leg_dist_actual[ix] = 0.0
                        drone.waypoint = next_target.copy()
            else:
                # Non-dwell waypoint: touch-to-capture
                if d < CAPTURE_RADIUS:
                    eff = math.sqrt(max(leg_dist_ideal[ix], eps) / max(leg_dist_actual[ix], leg_dist_ideal[ix], eps))
                    comp_scores[ix] += 1.0 * eff

                    # Advance or complete
                    current_wp[ix] += 1
                    if current_wp[ix] >= W:
                        drone.enabled = False
                        completions.append(time_elapsed)
                        completed.add(ix)
                    else:
                        next_target = chains[c_idx][current_wp[ix]]
                        leg_dist_ideal[ix] = np.linalg.norm(next_target - target)
                        leg_dist_actual[ix] = 0.0
                        drone.waypoint = next_target.copy()

            prev_t1[ix], prev_t2[ix] = drone.t1_thrust, drone.t2_thrust

        time_elapsed += dt

    # ── Partial credit for in-progress waypoint ─────────────────────────
    # Mirrors the 1.0 * eff awarded per capture but scaled by proximity.
    # Fills the gap between discrete capture plateaus so Pareto ranking
    # can differentiate genomes that are partway through a leg.
    for ix, drone in enumerate(drones):
        if ix in completed:
            continue
        c_idx  = ix // N
        wp_idx = current_wp[ix]
        if wp_idx >= W:
            continue
        target   = chains[c_idx][wp_idx]
        end_dist = np.linalg.norm(drone.pos - target)
        ideal    = max(leg_dist_ideal[ix], eps)
        actual   = max(leg_dist_actual[ix], ideal, eps)
        progress = max(1.0 - end_dist / ideal, 0.0)
        eff      = math.sqrt(ideal / actual)
        comp_scores[ix] += progress * eff

    # ── Score aggregation ────────────────────────────────────────────────
    # Normalise both components to [0,1] and blend 50/50 (matches stage 1).
    # Without this, comp_scores dominates with discrete plateaus (0,1,2..5
    # waypoints) that cluster genomes at the same score, bloating Pareto F1.
    vmax_norm = np.clip(vmax_scores, 0, 1)
    comp_norm = comp_scores / (W + eps)  # theoretical max = W (perfect eff on every wp)
    drone_scores = 0.5 * vmax_norm + 0.5 * np.clip(comp_norm, 0, 1)

    # Reshape to (C, N) and aggregate across chains
    per_chain = drone_scores.reshape(C, N)
    genome_scores = 0.7 * per_chain.mean(axis=0) + 0.3 * per_chain.min(axis=0)

    avg_completions = len(completed) / C

    # Waypoint capture stats: IQR across genomes
    wp_per_chain = current_wp.reshape(C, N)
    genome_wp = wp_per_chain.mean(axis=0)  # avg across chains per genome
    wp_stats = {
        'min': float(np.min(genome_wp)),
        'q1': float(np.percentile(genome_wp, 25)),
        'q3': float(np.percentile(genome_wp, 75)),
        'max': float(np.max(genome_wp)),
    }

    return 0, genome_scores, completions, avg_completions, wp_stats, avg_leg_dist
