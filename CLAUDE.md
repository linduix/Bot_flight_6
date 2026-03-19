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

1. **headless_trainer_prototype.py** — Main training loop. Runs generations, manages stage progression, checkpoints every 100 gens, Discord logging.
2. **drone_prototype.py** — 2D rigid body physics. Two independent thrusters with angle/thrust control. Provides 8 sensor inputs (bias, delta_x, delta_y, angle, vel_x, vel_y, angular_vel, thruster angles) to 4 outputs (thruster turns + throttles).
3. **network_prototype.py** — NEAT neural network. Feed-forward evaluation with topological sort. Supports hidden nodes added via mutation.
4. **genome_prototype.py** — Genome representation: node genes + connection genes with innovation numbers.
5. **breeding_prototype.py** — Speciation (compatibility distance), crossover, elitism, species stagnation tracking. Adaptive distance threshold (0.1–10+).
6. **mutation_prototype.py** — Weight perturbation (80%), add connection (5%), add node (3%).
7. **scoring_prototype.py** — Fitness evaluation. Stage 0: hover in place. Stage 1: target acquisition. Rewards progress toward target, penalizes lateral motion, bonuses for precision hover and completion.
8. **util_prototype.py** — Discord webhook logging, config helpers.

### Training stages

- **Stage 0**: Learn to hover (stationary target at spawn)
- **Stage 1**: Navigate to target with increasing spawn distance based on completion rates

### Persistence

- Checkpoints saved as pickle to `data/checkpoints/` every 100 generations
- State includes: generation, population, innovation tracker, species, best genome, historical scores
- Discord webhook notifications configured via `.env` (DISCORD_WEBHOOK_URL, DISCORD_LOGGING_ENABLED, INSTANCE_NAME)

## Conventions

- All tunable parameters should come from `data/simulation.toml` config, never hard-coded (aspiration — current prototypes still have inline constants)
- Population size: 1000 drones per generation
- Persist run artifacts under `data/`
- Input vectors use logarithmic encoding for normalization
