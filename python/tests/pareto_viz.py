"""
Pareto rank visualisation — successive fronts (F1, F2, ...) per checkpoint.

Each checkpoint gets its own panel. Within each panel, successive Pareto
fronts are peeled recursively:
  F1 = non-dominated set (best trade-off)
  F2 = non-dominated set after removing F1
  F3 = non-dominated set after removing F1+F2  ... etc.

x-axis : genetic distance from historical best  (more = more exploration)
y-axis : raw score / fitness                    (more = better)
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from breeding_prototype import distance

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

checkpoints = [
    (pop_best, best_state, f"Gen {best_state['gen']} (best)"),
    (pop_save, save_state, f"Gen {save_state['gen']} (save)"),
]

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
        _, scores, *_ = stage1_vmax_test(
            pop, config['width'], config['height'],
            config['meters_to_pixels'], limit=limit, diff=difficulty)
    else:
        _, scores, *_ = hover_scorer_headless(
            pop, config['width'], config['height'],
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


def pareto_rank(x_vals, y_vals, max_fronts=8):
    """
    Recursively peel Pareto fronts (max-x, max-y).
    Returns list of boolean masks, one per front.
    """
    remaining = np.ones(len(x_vals), dtype=bool)
    fronts = []
    while remaining.any() and len(fronts) < max_fronts:
        rx, ry = x_vals[remaining], y_vals[remaining]
        local_mask = np.zeros(rx.shape[0], dtype=bool)
        order = np.lexsort((-ry, -rx))
        best_y = -np.inf
        for idx in order:
            if ry[idx] >= best_y:
                local_mask[idx] = True
                best_y = ry[idx]
        global_mask = np.zeros(len(x_vals), dtype=bool)
        rem_indices = np.where(remaining)[0]
        global_mask[rem_indices[local_mask]] = True
        fronts.append(global_mask)
        remaining &= ~global_mask
    return fronts


# ── Score each checkpoint independently ──────────────────────────────
all_scores_max = 0.0
results = []

for pop, state, label in checkpoints:
    print(f"\nScoring {label}...")
    scores = score_population(pop, state)
    dists  = compute_distances(pop, ref_genome)
    all_scores_max = max(all_scores_max, scores.max())
    results.append((dists, scores, label, state))

score_floor = 0.10 * all_scores_max
print(f"\nGlobal best: {all_scores_max:.4f}  |  10% floor: {score_floor:.4f}")

# ── Plot: one panel per checkpoint ───────────────────────────────────
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(13 * n, 9), sharey=False)
if n == 1:
    axes = [axes]

fig.patch.set_facecolor('#0f0f12')
cmap = plt.cm.plasma

for ax, (dists, scores, label, state) in zip(axes, results):
    ax.set_facecolor('#1a1a2e')

    valid = ~np.isnan(dists) & (scores >= score_floor)
    x, y  = dists[valid], scores[valid]
    print(f"\n{label}: {valid.sum()} pts above floor")

    fronts = pareto_rank(x, y, max_fronts=8)
    print(f"  {len(fronts)} fronts computed")
    for i, m in enumerate(fronts):
        print(f"    F{i+1}: {m.sum()} pts")

    front_colors = [cmap(1.0 - i / max(len(fronts) - 1, 1)) for i in range(len(fronts))]

    # Draw back-to-front so F1 is on top
    for i in range(len(fronts) - 1, -1, -1):
        mask  = fronts[i]
        color = front_colors[i]
        px, py = x[mask], y[mask]
        order  = np.argsort(px)
        px, py = px[order], py[order]

        lw    = max(2.5 - i * 0.22, 0.8)
        alpha = max(0.95 - i * 0.09, 0.25)

        ax.scatter(px, py, c=[color], s=28, zorder=5 + (len(fronts) - i),
                   edgecolors='black', linewidths=0.5, alpha=alpha)
        ax.plot(px, py, color=color, linewidth=lw, zorder=4 + (len(fronts) - i),
                alpha=alpha, label=f'F{i+1}  ({mask.sum()} pts)')

    # Historical best marker
    ax.scatter([0], [all_scores_max], c='gold', s=280, marker='*',
               edgecolors='black', linewidths=1.5, zorder=20, label='Historical Best')

    ax.axhline(y=score_floor, color='gray', linestyle='--', alpha=0.35,
               linewidth=1.0, label=f'10% floor ({score_floor:.2f})')

    ax.set_xlim(0, x.max() * 1.05)
    ax.set_ylim(0, y.max() * 1.12)
    ax.set_xlabel('Genetic Distance from Historical Best', fontsize=12, color='#cccccc')
    ax.set_ylabel('Raw Score', fontsize=12, color='#cccccc')
    ax.set_title(label, fontsize=13, color='white', pad=10)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.grid(True, alpha=0.15, color='#4444aa')
    ax.legend(fontsize=9, loc='upper right',
              facecolor='#1a1a2e', labelcolor='white', edgecolor='#333366')

fig.suptitle(
    'Pareto Front Ranking  —  F1 = non-dominated  ·  F2 = non-dominated after removing F1  ·  ...',
    fontsize=13, color='white', y=1.01
)
plt.tight_layout()

out_path = checkpoint_dir.parent / 'pareto_front.png'
plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"\nSaved -> {out_path}")
plt.show()
