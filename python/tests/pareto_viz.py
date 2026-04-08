"""
2D Pareto front visualization: raw score vs genetic distance from historical best genome.
Alpha shape upper envelope with breeding band.
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from breeding_prototype import distance
from genome_prototype import Genome

checkpoint_dir = Path(__file__).parent.parent.parent / "data" / "checkpoints"

# ── Load checkpoints ──────────────────────────────────────────────────
best_path = checkpoint_dir / "prototype_best.pkl"
save_path = checkpoint_dir / "prototype_save.pkl"

with open(best_path, 'rb') as f:
    best_state = pickle.load(f)
with open(save_path, 'rb') as f:
    save_state = pickle.load(f)

ref_genome = best_state['best_drone']
ref_genome._conn_cache = None

pop_best = best_state['current_gen']
pop_save = save_state['current_gen']
for g in pop_best + pop_save:
    g._conn_cache = None

print(f"Best: gen {best_state['gen']}, pop {len(pop_best)}")
print(f"Save: gen {save_state['gen']}, pop {len(pop_save)}")

# ── Scoring ───────────────────────────────────────────────────────────
from prototype_stage2 import stage2_vmax_test
from prototype_stage1 import stage1_vmax_test
from scoring_prototype import hover_scorer_headless
from drone_prototype import Ai_Drone

config = {'width': 1920, 'height': 1080, 'meters_to_pixels': 100}
limit = 15


def score_population(pop, state):
    stage = state.get('stage', 1)
    difficulty = state.get('difficulty', 15)
    seed = state.get('pool_seed', state.get('validation_seed', 42))
    if stage >= 2:
        _, scores, *_ = stage2_vmax_test(
            pop, config['width'], config['height'],
            config['meters_to_pixels'], limit=limit, diff=difficulty, seed=seed)
    elif stage == 1:
        scores = stage1_vmax_test(
            pop, config['width'], config['height'],
            config['meters_to_pixels'], limit=limit, diff=difficulty)
    else:
        drones = [Ai_Drone((0, 0), config['meters_to_pixels'], config['height'], g) for g in pop]
        scores = hover_scorer_headless(drones, config['width'], config['height'], config['meters_to_pixels'], limit=limit)
    return np.array(scores)


def compute_distances(pop, ref):
    dists = []
    for g in pop:
        try:
            dists.append(distance(g, ref))
        except Exception:
            dists.append(np.nan)
    return np.array(dists)


# ── Alpha shape upper envelope ────────────────────────────────────────

def alpha_upper_envelope(x_vals, y_vals, alpha=1.0):
    """Get the upper boundary of the alpha shape.
    Returns indices of boundary vertices that form the ceiling,
    sorted by x with only the highest-y vertex per x-slice."""
    points = np.column_stack([x_vals, y_vals])
    tri = Delaunay(points)

    edge_count = {}
    for simplex in tri.simplices:
        pts = points[simplex]
        a = np.linalg.norm(pts[0] - pts[1])
        b = np.linalg.norm(pts[1] - pts[2])
        c = np.linalg.norm(pts[2] - pts[0])
        s = (a + b + c) / 2
        area = max(np.sqrt(max(s * (s-a) * (s-b) * (s-c), 0)), 1e-12)
        circumradius = (a * b * c) / (4 * area)
        if circumradius < 1.0 / alpha:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_edges = {e for e, c in edge_count.items() if c == 1}
    if len(boundary_edges) < 3:
        hull = ConvexHull(points)
        boundary_verts = set(hull.vertices)
    else:
        boundary_verts = set()
        for e in boundary_edges:
            boundary_verts.update(e)

    # Chain boundary edges into ordered polygon(s)
    adj = {}
    for e in boundary_edges:
        a, b = e
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Walk all connected components, keep the longest
    visited = set()
    chains = []
    for start in adj:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        cur = start
        while True:
            nxt = None
            for n in adj[cur]:
                if n not in visited:
                    nxt = n
                    break
            if nxt is None:
                break
            chain.append(nxt)
            visited.add(nxt)
            cur = nxt
        chains.append(chain)

    chain = max(chains, key=len)
    cx = np.array([x_vals[v] for v in chain])
    cy = np.array([y_vals[v] for v in chain])

    # Two paths between leftmost and rightmost vertex
    left_pos = int(np.argmin(cx))
    right_pos = int(np.argmax(cx))
    n = len(chain)

    def walk(start, end, direction):
        path = []
        i = start
        while True:
            path.append(chain[i])
            if i == end:
                break
            i = (i + direction) % n
        return path

    pathA = walk(left_pos, right_pos, +1)
    pathB = walk(left_pos, right_pos, -1)

    avgA = np.mean([y_vals[v] for v in pathA])
    avgB = np.mean([y_vals[v] for v in pathB])
    upper = pathA if avgA >= avgB else pathB

    mask = np.zeros(len(x_vals), dtype=bool)
    mask[upper] = True
    return mask


def breeding_band(x_vals, y_vals, envelope_mask, band_pct=0.15):
    """Select genomes within the top band_pct of score at each distance slice.
    Uses the envelope points to define distance bins."""
    px = x_vals[envelope_mask]
    n_bins = max(len(px), 10)
    bin_edges = np.linspace(x_vals.min(), x_vals.max() + 1e-9, n_bins + 1)

    band_mask = np.zeros(len(x_vals), dtype=bool)
    for i in range(n_bins):
        in_bin = (x_vals >= bin_edges[i]) & (x_vals < bin_edges[i + 1])
        if not in_bin.any():
            continue
        bin_scores = y_vals[in_bin]
        threshold = np.percentile(bin_scores, (1 - band_pct) * 100)
        above = in_bin & (y_vals >= threshold)
        band_mask |= above

    return band_mask


# ── Compute ───────────────────────────────────────────────────────────
print(f"\nScoring gen {best_state['gen']}...")
scores_best = score_population(pop_best, best_state)
dists_best = compute_distances(pop_best, ref_genome)

print(f"Scoring gen {save_state['gen']}...")
scores_save = score_population(pop_save, save_state)
dists_save = compute_distances(pop_save, ref_genome)

global_max = max(scores_best.max(), scores_save.max())
score_floor = 0.10 * global_max
print(f"Global best: {global_max:.4f}, 10% floor: {score_floor:.4f}")


def filter_data(dists, scores, floor):
    valid = ~np.isnan(dists) & (scores >= floor)
    return dists[valid], scores[valid]

x1, y1 = filter_data(dists_best, scores_best, score_floor)
x2, y2 = filter_data(dists_save, scores_save, score_floor)

spread1 = max(x1.max() - x1.min(), y1.max() - y1.min())
spread2 = max(x2.max() - x2.min(), y2.max() - y2.min())
alpha1, alpha2 = 2.0 / spread1, 2.0 / spread2

env1 = alpha_upper_envelope(x1, y1, alpha=alpha1)
env2 = alpha_upper_envelope(x2, y2, alpha=alpha2)
band1 = breeding_band(x1, y1, env1, band_pct=0.15)
band2 = breeding_band(x2, y2, env2, band_pct=0.15)

print(f"Gen {best_state['gen']}: envelope {env1.sum()} pts, band {band1.sum()} pts ({band1.sum()/len(x1)*100:.0f}%)")
print(f"Gen {save_state['gen']}: envelope {env2.sum()} pts, band {band2.sum()} pts ({band2.sum()/len(x2)*100:.0f}%)")

# ── Plot ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
x_max = max(x1.max(), x2.max()) * 1.05
y_max = max(y1.max(), y2.max()) * 1.1

for ax, x, y, env, band, gen in [
    (ax1, x1, y1, env1, band1, best_state['gen']),
    (ax2, x2, y2, env2, band2, save_state['gen']),
]:
    # All population (below band)
    below = ~band
    ax.scatter(x[below], y[below], alpha=0.2, s=12, c='steelblue', label='Population')

    # Breeding band
    band_only = band & ~env
    ax.scatter(x[band_only], y[band_only], alpha=0.5, s=20, c='mediumpurple',
               label=f'Breeding Band ({band.sum()} pts)')

    # Envelope line — sorted by x
    px, py = x[env], y[env]
    order = np.argsort(px)
    px, py = px[order], py[order]
    ax.plot(px, py, color='red', linewidth=2.5, zorder=5)
    ax.scatter(px, py, c='red', s=45, zorder=6, edgecolors='darkred',
               linewidths=0.5, label=f'Pareto Front ({env.sum()} pts)')

    ax.scatter([0], [global_max], c='gold', s=200, marker='*', edgecolors='black',
               linewidths=1.5, zorder=10, label='Historical Best')
    ax.axhline(y=score_floor, color='gray', linestyle='--', alpha=0.5,
               label=f'10% floor ({score_floor:.2f})')

    ax.set_title(f'Gen {gen}', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('Genetic Distance from Historical Best', fontsize=12)

ax1.set_ylabel('Raw Score', fontsize=12)

fig.suptitle('Alpha Shape Pareto Front + Breeding Band (top 15%)', fontsize=16)
plt.tight_layout()
plt.savefig(str(checkpoint_dir.parent / 'pareto_front.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved to {checkpoint_dir.parent / 'pareto_front.png'}")
plt.show()
