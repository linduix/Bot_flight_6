# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BotFlight6 is a NEAT (NeuroEvolution of Augmenting Topologies) driven 2D drone simulator. Python implements the NEAT engine (population, speciation, mutation, crossover, checkpoints) and visualization (Pygame). A Rust physics/scoring backend (PyO3 FFI) exists as a stub for future optimization.

## Running

```bash
# Headless training (primary workflow)
python python/tests/headless_trainer_prototype.py

# Visual training with Pygame rendering
python python/tests/visual_trainer_prototype.py

# Interactive showcase of best genome
python python/tests/showcase_prototype.py

# Manual drone control
python python/tests/player_prototype.py
```

No build step for Python. Rust scorer (`rust_scorer/`) is not yet integrated.

## Dependencies

pygame, numpy, matplotlib, networkx, requests, python-dotenv

## Architecture

The active code lives in `python/tests/*_prototype.py`. The top-level `python/*.py` files are empty stubs for a future refactor.

### Core pipeline

1. **headless_trainer_prototype.py** — Main training loop. Uses `multiprocessing.Pool` (one worker per CPU core, SIGINT-ignored) to score genomes in parallel chunks. Manages stage progression, checkpoints every 100 gens, 50-gen rolling metric buffer with Discord logging.
2. **drone_prototype.py** — 2D rigid body physics. Two independent thrusters with angle/thrust control. Provides 8 sensor inputs (bias, delta_x, delta_y, angle, vel_x, vel_y, angular_vel, thruster angles) to 4 outputs (thruster turns + throttles).
3. **network_prototype.py** — NEAT neural network. Feed-forward evaluation with topological sort. Supports hidden nodes added via mutation.
4. **genome_prototype.py** — Genome representation: node genes + connection genes with innovation numbers.
5. **breeding_prototype.py** — Speciation (compatibility distance), crossover, elitism, species stagnation tracking. Adaptive distance threshold (0.1–10+). Three-tier species immunity prevents stagnation-killing: (1) best average recent performer, (2) historical best species, (3) species containing global best genome.
6. **mutation_prototype.py** — Weight perturbation (80%), add connection (5%), add node (3%).
7. **scoring_prototype.py** — Fitness evaluation. Stage 0: hover in place. Stage 1: target acquisition. Rewards progress toward target, penalizes lateral motion, bonuses for precision hover and completion.
8. **prototype_stage1.py** — Stage 1 directional training. Tests 8 compass directions (N, NE, E, SE, S, SW, W, NW) simultaneously via `stage1_vmax_test`. Per-direction difficulty tracking with gap-squared weighting to favor weakest directions.
9. **util_prototype.py** — Discord webhook logging (batched with configurable interval), checkpoint save/load with custom filenames.

### Target behavior

The end goal is "guided munition" drone behavior for mouse-cursor chasing:

- Immediate full-speed commitment on a direct attack vector toward the target
- Maximum safe closing speed held throughout approach with no hesitation
- Hard committed deceleration at the physics-dictated braking threshold
- Precise stop at target with minimum residual velocity
- Straight-line paths always, no arcing or lateral deviation
- Smooth committed thrust with no oscillation
- When target moves (mouse cursor), instant re-orientation and re-commitment

Core scoring signal: `v_ratio = v_par / safe_v` where `safe_v = sqrt(2 * a_eff * d)` and `a_eff` accounts for gravity projected onto the braking direction. Asymmetric parabola rewards v_ratio=1.0 (at physics-max safe speed), steep penalty above 1.0 (can't stop in time).

### Training stages

- **Stage 0**: Learn to hover (stationary target at spawn)
- **Stage 1** (current): Navigate to targets in 8 compass directions with per-direction difficulty scaling. Spawn distance increases based on completion rates. All 8 directions tested simultaneously each evaluation.
- **Stage 1** (planned): Sequential random waypoints — touch-and-go intermediate waypoints with hover-hold on final waypoint. Fitness = waypoint_score + hover_score. Random placement with minimum distance floor, fly-throughs allowed. Difficulty knob: waypoint count in fixed time window, scaled by completion rate.

### Persistence

- Two checkpoint files in `data/checkpoints/`:
  - `prototype_save.pkl` — periodic save every 100 generations
  - `prototype_best.pkl` — saved whenever a new all-time best score is achieved
- State includes: generation, population, innovation tracker, species, best genome, historical scores
- Discord webhook notifications configured via env vars: `DISCORD_WEBHOOK`, `NAME`, `LOGGING` (ON/OFF)

## Conventions

- All tunable parameters should come from `data/simulation.toml` config, never hard-coded (aspiration — current prototypes still use inline config dicts)
- Population size: 500 drones per generation
- Persist run artifacts under `data/`
- Input vectors use logarithmic encoding for normalization
