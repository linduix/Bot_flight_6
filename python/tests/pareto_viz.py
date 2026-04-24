"""
Pareto rank visualisation — successive fronts (F1, F2, ...) per checkpoint.

Each checkpoint gets its own panel. Within each panel, successive Pareto
fronts are peeled recursively:
  F1 = non-dominated set (best trade-off)
  F2 = non-dominated set after removing F1
  F3 = non-dominated set after removing F1+F2  ... etc.

x-axis : KNN genetic diversity (avg distance to sqrt(n) nearest neighbors)
y-axis : raw score / fitness                    (more = better)
"""
import sys
import math
import os
import signal
import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from breeding_prototype import distance


def _pool_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

checkpoint_dir = Path(__file__).parent.parent.parent / "data" / "checkpoints"

# ── Load checkpoints ──────────────────────────────────────────────────
best_path = checkpoint_dir / "prototype_best.pkl"
save_path = checkpoint_dir / "prototype_save.pkl"

with open(best_path, 'rb') as f:
    best_state = pickle.load(f)
with open(save_path, 'rb') as f:
    save_state = pickle.load(f)

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


def score_population(pop, state, pool, num_workers):
    stage = state.get('stage', 1)
    difficulty = state.get('difficulty', 15)
    seed = state.get('pool_seed', state.get('validation_seed', 42))
    W, H, M = config['width'], config['height'], config['meters_to_pixels']

    chunk_size = math.ceil(len(pop) / num_workers)
    chunks = [pop[i:i + chunk_size] for i in range(0, len(pop), chunk_size)]

    if stage >= 2:
        raw = pool.starmap(stage2_vmax_test,
            [(c, W, H, M, limit, difficulty, seed) for c in chunks])
    elif stage == 1:
        raw = pool.starmap(stage1_vmax_test,
            [(c, W, H, M, limit, difficulty) for c in chunks])
    else:
        raw = pool.starmap(hover_scorer_headless,
            [(c, W, H, M, limit) for c in chunks])

    scores = np.concatenate([np.array(r[1]) for r in raw])
    return scores


def compute_knn_diversity(pop):
    """Average distance to sqrt(n) nearest neighbors (mirrors breeding_prototype)."""
    n = len(pop)
    k = max(1, int(n ** 0.5))
    gen_dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            try:
                d = distance(pop[i], pop[j])
            except Exception:
                d = np.nan
            gen_dists[i][j] = d
            gen_dists[j][i] = d
    obj_dists = np.full(n, np.nan)
    for i in range(n):
        row = gen_dists[i]
        valid = row[~np.isnan(row)]
        sorted_d = np.sort(valid)  # self = 0.0 at index 0
        obj_dists[i] = sorted_d[1:k + 1].mean() if len(sorted_d) > 1 else np.nan
    return obj_dists


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


if __name__ == '__main__':
    # ── Score each checkpoint independently ──────────────────────────────
    num_workers = max(1, os.cpu_count())
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    print(f"Using {num_workers} workers")
    scoring_pool = mp.Pool(processes=num_workers, initializer=_pool_init)
    try:
        all_scores_max = 0.0
        results = []

        for pop, state, label in checkpoints:
            print(f"\nScoring {label}...")
            scores = score_population(pop, state, scoring_pool, num_workers)
            dists  = compute_knn_diversity(pop)
            all_scores_max = max(all_scores_max, scores.max())
            results.append((dists, scores, label, state))
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers.")
        scoring_pool.terminate()
        scoring_pool.join()
        sys.exit(0)
    else:
        scoring_pool.close()
        scoring_pool.join()

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

        ax.axhline(y=score_floor, color='gray', linestyle='--', alpha=0.35,
                   linewidth=1.0, label=f'10% floor ({score_floor:.2f})')

        ax.set_xlim(0, x.max() * 1.05)
        ax.set_ylim(0, y.max() * 1.12)
        ax.set_xlabel('KNN Genetic Diversity (avg dist to √n neighbors)', fontsize=12, color='#cccccc')
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
