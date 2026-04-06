"""
AgentManager — the adaptive 'brain' of Agentic-Q-Tuner.

Drives the QEA in discrete chunks of N generations, analyzes the fitness
trend after every chunk, and autonomously tunes the Quantum Rotation Angle
when progress stalls.

Adaptation rule
---------------
After each chunk, compute:
    delta = best_fitness_at_chunk_end - best_fitness_at_chunk_start

If delta < STALL_THRESHOLD (1 % absolute improvement):
    → STALL detected: multiply rotation_angle by ANGLE_SCALE_UP (capped at MAX_ANGLE)
      Rationale: a larger angle forces broader exploration of the 5,000-feature space.

If delta >= STALL_THRESHOLD and rotation_angle > DEFAULT_ANGLE:
    → Progress resumed: step rotation_angle back down toward the default.
      Rationale: fine-grained exploitation is again appropriate.

Every decision—including the reason and numeric values—is written to a
timestamped log file in agent_logs/.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Re-use the QEA primitives without reimplementing them
sys.path.insert(0, str(Path(__file__).parent))
from q_algorithm import (
    Individual,
    BestSolution,
    evaluate_fitness,
    apply_rotation_gate,
    mutate,
    rank_features_by_activity,
    POPULATION_SIZE,
    ROTATION_ANGLE,
    MUTATION_RATE,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Agent hyper-parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE      = 5                   # generations between each analysis pass
TOTAL_GEN       = 60                  # total generations to run

STALL_THRESHOLD = 0.01                # < 1 % absolute improvement → stall
ANGLE_SCALE_UP  = 1.5                 # multiply angle by this on stall
ANGLE_STEP_DOWN = 0.85                # multiply angle by this on recovery
DEFAULT_ANGLE   = ROTATION_ANGLE      # 0.05π  (baseline)
MAX_ANGLE       = 0.25 * np.pi        # hard ceiling to prevent chaos

LOG_DIR = Path(__file__).parent.parent / "agent_logs"


# ---------------------------------------------------------------------------
# AgentManager
# ---------------------------------------------------------------------------
class AgentManager:
    """
    Adaptive controller that wraps the QEA evolution loop.

    Parameters
    ----------
    X               : feature matrix  (n_samples × n_features)
    y_raw           : class labels (string or int array-like)
    population_size : Q-population size
    rotation_angle  : starting |Δθ| for the Quantum Rotation Gate
    mutation_rate   : per-Q-bit swap-mutation probability
    random_seed     : master RNG seed
    log_dir         : directory for decision logs (created if absent)
    chunk_size      : generations per analysis window
    total_generations : total number of generations to evolve
    """

    def __init__(
        self,
        X:               pd.DataFrame,
        y_raw,
        population_size: int   = POPULATION_SIZE,
        rotation_angle:  float = DEFAULT_ANGLE,
        mutation_rate:   float = MUTATION_RATE,
        random_seed:     int   = RANDOM_SEED,
        log_dir:         Path  = LOG_DIR,
        chunk_size:      int   = CHUNK_SIZE,
        total_generations: int = TOTAL_GEN,
    ) -> None:
        self.X              = X
        self.n_features     = X.shape[1]
        self.mutation_rate  = mutation_rate
        self.chunk_size     = chunk_size
        self.total_gen      = total_generations

        # Encode labels once
        self._le = LabelEncoder()
        self.y   = self._le.fit_transform(y_raw)

        # Mutable state that the agent controls
        self.rotation_angle = rotation_angle
        self.rng            = np.random.default_rng(random_seed)

        # Q-population
        self.population: list[Individual] = [
            Individual(self.n_features) for _ in range(population_size)
        ]
        self.best: Optional[BestSolution] = None

        # Telemetry
        self.history:           list[float] = []   # global_best per generation
        self.chunk_start_bests: list[float] = []   # best at start of each chunk
        self.angle_history:     list[float] = []   # rotation_angle at each chunk

        # Logging
        self._log_dir  = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        run_ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = self._log_dir / f"run_{run_ts}.log"
        self._log_file = open(self._log_path, "w", buffering=1)   # line-buffered

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, message: str, also_print: bool = True) -> None:
        """Stamp and persist a decision or status message."""
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line  = f"[{ts}] {message}"
        self._log_file.write(line + "\n")
        if also_print:
            print(line)

    # ------------------------------------------------------------------
    # Single-generation step
    # ------------------------------------------------------------------
    def _step_generation(self) -> tuple[float, float, float]:
        """
        Run one generation of the QEA.

        Returns
        -------
        (global_best_fitness, gen_best_fitness, gen_mean_fitness)
        """
        fitnesses: list[float] = []

        for ind in self.population:
            chromosome = ind.observe(self.rng)
            fitness    = evaluate_fitness(
                chromosome, self.X, self.y,
                rng_seed=int(self.rng.integers(1_000_000)),
            )
            fitnesses.append(fitness)

            if self.best is None or fitness > self.best.fitness:
                self.best = BestSolution(
                    chromosome=chromosome.copy(), fitness=fitness
                )

        for ind, fit in zip(self.population, fitnesses):
            apply_rotation_gate(ind, self.best, fit, delta_theta=self.rotation_angle)
            mutate(ind, self.rng, rate=self.mutation_rate)

        self.history.append(self.best.fitness)
        return self.best.fitness, max(fitnesses), float(np.mean(fitnesses))

    # ------------------------------------------------------------------
    # Chunk execution
    # ------------------------------------------------------------------
    def _run_chunk(self, chunk_idx: int, start_gen: int) -> list[float]:
        """
        Run self.chunk_size generations.

        Returns the global-best fitness recorded after each generation
        within this chunk.
        """
        end_gen = min(start_gen + self.chunk_size, self.total_gen)
        n_gens  = end_gen - start_gen

        chunk_bests: list[float] = []

        for local_gen in range(n_gens):
            gen_abs = start_gen + local_gen + 1
            global_best, gen_best, gen_mean = self._step_generation()
            chunk_bests.append(global_best)

            self._log(
                f"  Gen {gen_abs:03d}/{self.total_gen} | "
                f"global_best={global_best:.4f}  "
                f"gen_best={gen_best:.4f}  "
                f"gen_mean={gen_mean:.4f}  |  "
                f"Δθ={self.rotation_angle/np.pi:.4f}π  "
                f"features_in_best={int(self.best.chromosome.sum())}",
            )

        return chunk_bests

    # ------------------------------------------------------------------
    # Analysis and adaptation
    # ------------------------------------------------------------------
    def _analyze_and_adapt(
        self,
        chunk_idx:    int,
        chunk_bests:  list[float],
        chunk_start:  float,
    ) -> None:
        """
        Inspect the fitness trend over the last chunk and adjust Δθ.

        Decision tree
        -------------
        1. delta < STALL_THRESHOLD  →  STALL: increase rotation angle.
        2. delta >= STALL_THRESHOLD and angle > default  →  RECOVERING: decrease angle.
        3. delta >= STALL_THRESHOLD and angle == default →  HEALTHY: no change.
        """
        chunk_end = chunk_bests[-1]
        delta     = chunk_end - chunk_start

        self._log(
            f"ANALYSIS [Chunk {chunk_idx}] "
            f"Fitness: {chunk_start:.4f} → {chunk_end:.4f}  "
            f"(Δ={delta:+.4f}  threshold={STALL_THRESHOLD:.2f})",
        )

        old_angle = self.rotation_angle

        if delta < STALL_THRESHOLD:
            # ── STALL ──────────────────────────────────────────────────
            new_angle = min(self.rotation_angle * ANGLE_SCALE_UP, MAX_ANGLE)
            self.rotation_angle = new_angle

            self._log(
                f"DECISION [Chunk {chunk_idx}] "
                f"Fitness stalled at {chunk_end:.4f} "
                f"(Δ={delta:+.4f} < {STALL_THRESHOLD:.0%}), "
                f"increasing rotation angle: "
                f"{old_angle/np.pi:.4f}π → {new_angle/np.pi:.4f}π  "
                f"[scale ×{ANGLE_SCALE_UP}]",
            )

        elif self.rotation_angle > DEFAULT_ANGLE + 1e-9:
            # ── RECOVERING ─────────────────────────────────────────────
            new_angle = max(self.rotation_angle * ANGLE_STEP_DOWN, DEFAULT_ANGLE)
            self.rotation_angle = new_angle

            self._log(
                f"DECISION [Chunk {chunk_idx}] "
                f"Fitness improving (Δ={delta:+.4f} ≥ {STALL_THRESHOLD:.0%}), "
                f"stepping rotation angle back down: "
                f"{old_angle/np.pi:.4f}π → {new_angle/np.pi:.4f}π  "
                f"[scale ×{ANGLE_STEP_DOWN}]",
            )

        else:
            # ── HEALTHY ────────────────────────────────────────────────
            self._log(
                f"DECISION [Chunk {chunk_idx}] "
                f"Fitness improving (Δ={delta:+.4f} ≥ {STALL_THRESHOLD:.0%}). "
                f"Rotation angle unchanged: {self.rotation_angle/np.pi:.4f}π",
            )

        self.angle_history.append(self.rotation_angle)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Execute the full adaptive QEA run.

        Returns
        -------
        dict matching the shape returned by q_algorithm.run_qea(), plus:
            'angle_history'      : list[float], Δθ at the end of each chunk
            'chunk_start_bests'  : list[float], global_best at each chunk boundary
            'log_path'           : Path, location of the decision log
        """
        n_chunks = (self.total_gen + self.chunk_size - 1) // self.chunk_size

        self._log("=" * 70)
        self._log("Agentic Q-Tuner — Adaptive QEA Run")
        self._log("=" * 70)
        self._log(
            f"Dataset   : {self.X.shape[0]} samples × {self.X.shape[1]} features"
        )
        self._log(
            f"Population: {len(self.population)}  |  "
            f"Rotation angle: {self.rotation_angle/np.pi:.4f}π  |  "
            f"Mutation rate: {self.mutation_rate:.4f}"
        )
        self._log(
            f"Schedule  : {self.total_gen} total generations, "
            f"{self.chunk_size} per chunk → {n_chunks} analysis windows"
        )
        self._log(
            f"Stall rule: Δfitness < {STALL_THRESHOLD:.0%} → multiply Δθ by {ANGLE_SCALE_UP}"
        )
        self._log("-" * 70)

        current_gen = 0

        for chunk_idx in range(1, n_chunks + 1):
            gens_this_chunk = min(self.chunk_size, self.total_gen - current_gen)
            chunk_end_gen   = current_gen + gens_this_chunk

            self._log(
                f"\n--- Chunk {chunk_idx}/{n_chunks}  "
                f"(Gen {current_gen + 1}–{chunk_end_gen})  "
                f"Δθ={self.rotation_angle/np.pi:.4f}π ---"
            )

            # Record the fitness *before* this chunk begins
            chunk_start_best = self.best.fitness if self.best is not None else 0.0
            self.chunk_start_bests.append(chunk_start_best)

            chunk_bests = self._run_chunk(chunk_idx, current_gen)
            current_gen += gens_this_chunk

            self._analyze_and_adapt(chunk_idx, chunk_bests, chunk_start_best)

        # ------------------------------------------------------------------
        # Finalise
        # ------------------------------------------------------------------
        mean_probs    = np.stack([ind.selection_probs for ind in self.population]).mean(axis=0)
        best_features = list(self.X.columns[self.best.chromosome.astype(bool)])

        self._log("\n" + "=" * 70)
        self._log("Run complete.")
        self._log(f"Best balanced accuracy : {self.best.fitness:.4f}")
        self._log(f"Features in best chr.  : {len(best_features)}")
        self._log(f"Log written to         : {self._log_path}")
        self._log("=" * 70)

        self._log_file.close()

        return {
            "best_chromosome":   self.best.chromosome,
            "best_fitness":      self.best.fitness,
            "best_features":     best_features,
            "selection_probs":   mean_probs,
            "history":           self.history,
            "angle_history":     self.angle_history,
            "chunk_start_bests": self.chunk_start_bests,
            "log_path":          self._log_path,
        }
