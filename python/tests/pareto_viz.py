"""
2D genome-space visualization: raw score vs genetic distance from historical best.

═══════════════════════════════════════════════════════════════════════════
COORDINATE SYSTEM & SEMANTICS  (read this before touching anything)
═══════════════════════════════════════════════════════════════════════════

  x-axis : genetic distance from the historical best genome
           → MORE distance = MORE exploration (diverged further from best)
  y-axis : raw score (fitness)
           → MORE score = MORE exploitation (better at the task)

  BOTH axes point in the "good" direction.  A genome in the upper-right
  corner is both high-scoring AND genetically diverse — the ideal.

  The "upper boundary" / "rag on top" we want to visualise is the CEILING
  of the point cloud — imagine draping a cloth from above onto the scatter.
  It traces the best score achievable at each level of exploration.

  Floor: only genomes scoring >= 10% of the historical best are plotted.
         This removes noise / dead genomes from the bottom of the cloud.

Three upper-boundary methods shown side by side:
  1. Alpha Shape Envelope  – concave hull upper boundary ("bendy" — more
     flexible than convex hull, hugs the top fold of the cloud)
  2. Regularized Envelope  – bin-max + Gaussian smoothing (smooth, human-
     like curve that never dips below data)
  3. Pareto Front          – true non-dominated set for MAX x, MAX y
     (a point is non-dominated iff nothing else beats it on BOTH axes)

Breeding band (top 15% per distance slice) highlighted on all three.
═══════════════════════════════════════════════════════════════════════════
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import Delaunay, ConvexHull
from scipy.ndimage import gaussian_filter1d
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
        scores = hover_scorer_headless(
            drones, config['width'], config['height'],
            config['meters_to_pixels'], limit=limit)
    return np.array(scores)


def compute_distances(pop, ref):
    dists = []
    for g in pop:
        try:
            dists.append(distance(g, ref))
        except Exception:
            dists.append(np.nan)
    return np.array(dists)


# ══════════════════════════════════════════════════════════════════════
#  METHOD 1: Alpha Shape Upper Envelope
# ══════════════════════════════════════════════════════════════════════
# NOTE: we use the alpha shape instead of convex hull because convex
# hull is too rigid — it stretches a straight line across concavities.
# Alpha shape follows the actual "bendiness" of the cloud boundary.
# We then extract just the TOP fold (the "rag draped on top").
#
# Approach: compute the full alpha-shape boundary, then extract the
# upper fold by binning boundary points along x and keeping only the
# highest-y point per bin.  This avoids the fragile chain-walk /
# left-right path split that breaks on non-convex or disconnected
# boundaries.

def alpha_upper_envelope(x_vals, y_vals, alpha=1.0, n_bins=50):
    """
    Compute the upper fold of the alpha shape boundary.

    1. Delaunay triangulate, filter simplices by circumradius (alpha test).
    2. Boundary edges = edges appearing in exactly one surviving simplex.
    3. Collect all boundary vertex indices.
    4. Bin boundary vertices along x, keep only the highest-y per bin.
    5. Sort left-to-right for line plotting.

    Returns (mask, upper_indices):
      mask           – bool array over all points, True for upper boundary pts
      upper_indices  – ordered indices (left→right) for direct line plotting
    """
    points = np.column_stack([x_vals, y_vals])
    n = len(points)

    # ── Fallback for tiny populations ──
    if n < 4:
        hull = ConvexHull(points)
        mask = np.zeros(n, dtype=bool)
        mask[hull.vertices] = True
        hv = hull.vertices[np.argsort(x_vals[hull.vertices])]
        return mask, hv

    # ── Alpha shape boundary ──
    tri = Delaunay(points)
    edge_count = {}
    for simplex in tri.simplices:
        pts = points[simplex]
        a = np.linalg.norm(pts[0] - pts[1])
        b = np.linalg.norm(pts[1] - pts[2])
        c = np.linalg.norm(pts[2] - pts[0])
        s = (a + b + c) / 2
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
        if area < 1e-12:
            continue
        circumradius = (a * b * c) / (4 * area)
        if circumradius < 1.0 / alpha:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_edges = {e for e, cnt in edge_count.items() if cnt == 1}

    if len(boundary_edges) < 3:
        # Alpha too tight — fall back to convex hull
        hull = ConvexHull(points)
        boundary_verts = set(hull.vertices)
    else:
        boundary_verts = set()
        for a, b in boundary_edges:
            boundary_verts.add(a)
            boundary_verts.add(b)

    boundary_verts = np.array(sorted(boundary_verts))
    bx = x_vals[boundary_verts]
    by = y_vals[boundary_verts]

    # ── Extract upper fold: highest-y boundary point per x-bin ──
    edges = np.linspace(bx.min(), bx.max() + 1e-9, n_bins + 1)
    upper_idxs = []   # indices into the original x_vals/y_vals arrays
    for i in range(n_bins):
        in_bin = (bx >= edges[i]) & (bx < edges[i + 1])
        if not in_bin.any():
            continue
        # Among boundary verts in this bin, pick the one with highest y
        bin_local = np.where(in_bin)[0]
        best_local = bin_local[np.argmax(by[bin_local])]
        upper_idxs.append(boundary_verts[best_local])

    upper_idxs = np.array(upper_idxs)

    # Sort left-to-right by x for clean line plotting
    order = np.argsort(x_vals[upper_idxs])
    upper_idxs = upper_idxs[order]

    mask = np.zeros(n, dtype=bool)
    mask[upper_idxs] = True
    return mask, upper_idxs


# ══════════════════════════════════════════════════════════════════════
#  METHOD 2: Regularized (Smooth Quantile) Envelope
# ══════════════════════════════════════════════════════════════════════

def regularized_envelope(x_vals, y_vals, n_bins=40, quantile=0.92, sigma=2.5):
    """
    Human-like smooth upper boundary.

    Algorithm:
      1. Divide x-range into n_bins equal-width bins.
      2. In each bin, take the `quantile`-th percentile of y (e.g. 92nd).
         Using a high quantile rather than max makes it robust to outliers
         while still tracing the genuine upper fold of the cloud.
      3. Smooth the resulting curve with a Gaussian filter (sigma controls
         how glassy/stiff the curve is — higher = smoother).
      4. Mark each genome as "on the envelope" if it is the closest genome
         to the smoothed curve in its bin (gives one representative point
         per bin for scatter highlighting).

    Returns
    -------
    env_x, env_y : 1-D arrays
        The smooth envelope curve (for line plotting).
    mask : bool array (len == len(x_vals))
        True for genomes that sit on / very near the envelope.
    """
    x_min, x_max = x_vals.min(), x_vals.max()
    edges = np.linspace(x_min, x_max + 1e-9, n_bins + 1)
    bin_cx = 0.5 * (edges[:-1] + edges[1:])   # bin centres

    raw_max = np.full(n_bins, np.nan)   # actual per-bin ceiling
    for i in range(n_bins):
        in_bin = (x_vals >= edges[i]) & (x_vals < edges[i + 1])
        if in_bin.any():
            raw_max[i] = y_vals[in_bin].max()

    valid = ~np.isnan(raw_max)
    if valid.sum() < 2:
        return bin_cx, raw_max, np.zeros(len(x_vals), dtype=bool)

    # Interpolate gaps, smooth, then clip back UP so the curve never dips
    # below the actual per-bin maximum.  This prevents the Gaussian from
    # dragging the envelope into the interior of the cloud in sparse regions.
    raw_filled  = np.interp(bin_cx, bin_cx[valid], raw_max[valid])
    smooth_q    = gaussian_filter1d(raw_filled, sigma=sigma)
    smooth_q    = np.maximum(smooth_q, raw_filled)   # ← key fix: no dipping below data

    # Mark a genome as "near-ceiling" if within 5% of the envelope at its x
    env_at_x = np.interp(x_vals, bin_cx, smooth_q)
    rel_gap  = (env_at_x - y_vals) / (np.abs(env_at_x) + 1e-9)
    mask     = rel_gap < 0.05

    return bin_cx[valid], smooth_q[valid], mask


# ══════════════════════════════════════════════════════════════════════
#  METHOD 3: True Pareto Front  (MAX x, MAX y)
# ══════════════════════════════════════════════════════════════════════
# NOTE: both axes are "good" in the positive direction.
#   x = genetic distance = exploration
#   y = raw score         = exploitation
# A genome is Pareto-optimal iff NO other genome has both x >= xi AND
# y >= yi with at least one strict inequality.  The front traces the
# UPPER-RIGHT boundary of the cloud — stepping from the highest-y
# point (pure exploitation) rightward/downward to the highest-x point
# (pure exploration).
#
# Algorithm: sort by DESCENDING x (ties by descending y), sweep keeping
# a running max of y.  A point survives iff y >= running_max — meaning
# nothing to its RIGHT (higher x) also has higher y.

def pareto_front(x_vals, y_vals):
    """
    Non-dominated set for max-x, max-y objectives.

    A point (xi, yi) is Pareto-optimal iff no other point has
    x >= xi AND y >= yi with at least one strict inequality.

    Sweep from right to left (descending x).  A point survives iff its
    y is >= the best y seen so far among points to its right.  This
    gives the upper-right staircase: descending y as x decreases.

    Returns a boolean mask (len == len(x_vals)).
    """
    n = len(x_vals)
    # Sort by descending x; ties broken by descending y so that among
    # same-x points only the highest-y survives.
    order = np.lexsort((-y_vals, -x_vals))

    mask = np.zeros(n, dtype=bool)
    best_y = -np.inf
    for idx in order:
        if y_vals[idx] >= best_y:
            mask[idx] = True
            best_y = y_vals[idx]

    return mask


def pareto_staircase(x_vals, y_vals, mask):
    """
    Return (px, py) sorted by ascending distance for step-plot rendering.
    pareto_front() already computed the non-dominated set; just sort it.
    """
    px = x_vals[mask]
    py = y_vals[mask]
    order = np.argsort(px)
    return px[order], py[order]


# ══════════════════════════════════════════════════════════════════════
#  Breeding band (shared across methods)
# ══════════════════════════════════════════════════════════════════════

def breeding_band(x_vals, y_vals, n_bins=40, band_pct=0.15):
    """
    Mark genomes in the top `band_pct` fraction of score within each
    x-distance bin.  Independent of which envelope method is used.
    """
    edges = np.linspace(x_vals.min(), x_vals.max() + 1e-9, n_bins + 1)
    band_mask = np.zeros(len(x_vals), dtype=bool)
    for i in range(n_bins):
        in_bin = (x_vals >= edges[i]) & (x_vals < edges[i + 1])
        if not in_bin.any():
            continue
        threshold = np.percentile(y_vals[in_bin], (1 - band_pct) * 100)
        band_mask |= in_bin & (y_vals >= threshold)
    return band_mask


# ── Compute ───────────────────────────────────────────────────────────
print(f"\nScoring gen {best_state['gen']}...")
scores_best = score_population(pop_best, best_state)
dists_best  = compute_distances(pop_best, ref_genome)

print(f"Scoring gen {save_state['gen']}...")
scores_save = score_population(pop_save, save_state)
dists_save  = compute_distances(pop_save, ref_genome)

global_max  = max(scores_best.max(), scores_save.max())
score_floor = 0.10 * global_max
print(f"Global best: {global_max:.4f}, 10% floor: {score_floor:.4f}")


def filter_data(dists, scores, floor):
    valid = ~np.isnan(dists) & (scores >= floor)
    return dists[valid], scores[valid]


x1, y1 = filter_data(dists_best,  scores_best,  score_floor)
x2, y2 = filter_data(dists_save,  scores_save,  score_floor)

# Alpha envelope parameters
spread1 = max(x1.max() - x1.min(), y1.max() - y1.min())
spread2 = max(x2.max() - x2.min(), y2.max() - y2.min())
alpha1, alpha2 = 2.0 / spread1, 2.0 / spread2

# Compute all three methods for both generations
env_alpha1, alpha_path1 = alpha_upper_envelope(x1, y1, alpha=alpha1)
env_alpha2, alpha_path2 = alpha_upper_envelope(x2, y2, alpha=alpha2)

reg_bx1, reg_by1, env_reg1 = regularized_envelope(x1, y1)
reg_bx2, reg_by2, env_reg2 = regularized_envelope(x2, y2)

env_pareto1 = pareto_front(x1, y1)
env_pareto2 = pareto_front(x2, y2)
stair_x1, stair_y1 = pareto_staircase(x1, y1, env_pareto1)
stair_x2, stair_y2 = pareto_staircase(x2, y2, env_pareto2)

band1 = breeding_band(x1, y1)
band2 = breeding_band(x2, y2)

print(f"\nGen {best_state['gen']}:")
print(f"  Alpha envelope: {env_alpha1.sum()} pts")
print(f"  Regularized:    {env_reg1.sum()} pts near ceiling")
print(f"  Pareto front:   {env_pareto1.sum()} pts")
print(f"  Breeding band:  {band1.sum()} pts ({band1.sum()/len(x1)*100:.0f}%)")

print(f"\nGen {save_state['gen']}:")
print(f"  Alpha envelope: {env_alpha2.sum()} pts")
print(f"  Regularized:    {env_reg2.sum()} pts near ceiling")
print(f"  Pareto front:   {env_pareto2.sum()} pts")
print(f"  Breeding band:  {band2.sum()} pts ({band2.sum()/len(x2)*100:.0f}%)")


# ── Plot ──────────────────────────────────────────────────────────────
#
#  Layout: 3 methods × 2 generations = 6 panels (3 rows, 2 cols)
#
METHODS = ['Alpha Shape', 'Regularized (Smooth Max)', 'True Pareto Front']
COLORS  = {
    'pop':    ('steelblue',    0.18, 10),  # colour, alpha, s
    'band':   ('mediumpurple', 0.50, 18),
    'env':    ('red',          0.90, 40),
    'curve':  'crimson',
    'pareto': 'darkorange',
    'best':   'gold',
    'floor':  'gray',
}

fig, axes = plt.subplots(3, 2, figsize=(18, 20), sharey=True, sharex=False)
fig.patch.set_facecolor('#0f0f12')

x_max = max(x1.max(), x2.max()) * 1.05
y_max = max(y1.max(), y2.max()) * 1.12

gen_data = [
    (x1, y1, best_state['gen'],
     env_alpha1, alpha_path1, reg_bx1, reg_by1, env_reg1,
     env_pareto1, stair_x1, stair_y1, band1, alpha1),
    (x2, y2, save_state['gen'],
     env_alpha2, alpha_path2, reg_bx2, reg_by2, env_reg2,
     env_pareto2, stair_x2, stair_y2, band2, alpha2),
]

for col, (x, y, gen,
          env_a, apath, rbx, rby, env_r,
          env_p, sx, sy, band, alph) in enumerate(gen_data):

    for row, method in enumerate(METHODS):
        ax = axes[row][col]
        ax.set_facecolor('#1a1a2e')

        # ── Background population ──────────────────────────────────────
        below = ~band
        ax.scatter(x[below], y[below],
                   alpha=COLORS['pop'][1], s=COLORS['pop'][2],
                   c=COLORS['pop'][0], rasterized=True, label='Population')

        # ── Breeding band ──────────────────────────────────────────────
        band_only = band
        ax.scatter(x[band_only], y[band_only],
                   alpha=COLORS['band'][1], s=COLORS['band'][2],
                   c=COLORS['band'][0], rasterized=True,
                   label=f'Breeding Band ({band.sum()} pts, top 15%)')

        # ── Method-specific envelope ───────────────────────────────────
        if method == 'Alpha Shape':
            # Use the walk-ordered upper path for the line
            px, py = x[apath], y[apath]
            ax.plot(px, py, color=COLORS['curve'], linewidth=2.2,
                    zorder=5, label=f'Alpha Envelope (α={alph:.3f})')
            ax.scatter(px, py, c=COLORS['curve'], s=COLORS['env'][2],
                       zorder=6, edgecolors='darkred', linewidths=0.5,
                       label=f'Envelope pts ({env_a.sum()})')

        elif method == 'Regularized (Smooth Max)':
            ax.plot(rbx, rby, color=COLORS['curve'], linewidth=2.5,
                    zorder=5, label='Smooth Max Envelope (per-bin max + Gaussian, σ=2.5)')
            ax.scatter(x[env_r], y[env_r],
                       c=COLORS['curve'], s=COLORS['env'][2],
                       zorder=6, edgecolors='darkred', linewidths=0.5,
                       label=f'Near-ceiling pts ({env_r.sum()})')

        elif method == 'True Pareto Front':
            # Staircase line
            ax.step(sx, sy, where='post', color=COLORS['pareto'],
                    linewidth=2.5, zorder=5,
                    label=f'Pareto Staircase ({env_p.sum()} pts)')
            ax.scatter(x[env_p], y[env_p],
                       c=COLORS['pareto'], s=COLORS['env'][2],
                       zorder=6, edgecolors='saddlebrown', linewidths=0.5,
                       label='Non-dominated pts')

        # ── Historical best marker ─────────────────────────────────────
        ax.scatter([0], [global_max],
                   c=COLORS['best'], s=220, marker='*',
                   edgecolors='black', linewidths=1.5,
                   zorder=10, label='Historical Best')

        # ── 10% floor line ─────────────────────────────────────────────
        ax.axhline(y=score_floor, color=COLORS['floor'],
                   linestyle='--', alpha=0.45, linewidth=1.2,
                   label=f'10% floor ({score_floor:.2f})')

        # ── Cosmetics ──────────────────────────────────────────────────
        title = f'{method}  —  Gen {gen}'
        ax.set_title(title, fontsize=12, color='white', pad=7)
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.15, color='#4444aa')
        ax.legend(fontsize=8.5, loc='upper right',
                  facecolor='#1a1a2e', labelcolor='white',
                  edgecolor='#333366')

        if col == 0:
            ax.set_ylabel('Raw Score', fontsize=11, color='#cccccc')
        if row == 2:
            ax.set_xlabel('Genetic Distance from Historical Best',
                          fontsize=11, color='#cccccc')

fig.suptitle(
    'Genome Space — Upper Boundary Methods Comparison\n'
    'Alpha Shape  ·  Regularized Smooth Max  ·  True Pareto Front',
    fontsize=15, color='white', y=1.005
)
plt.tight_layout(h_pad=3.0, w_pad=2.5)

out_path = checkpoint_dir.parent / 'pareto_front.png'
plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"\nSaved → {out_path}")
plt.show()