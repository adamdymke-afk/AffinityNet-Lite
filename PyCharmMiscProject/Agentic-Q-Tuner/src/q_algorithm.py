"""
Quantum-Inspired Evolutionary Algorithm (QEA) for omics feature selection.

Q-bit representation
--------------------
Each feature i has two real amplitudes (alpha_i, beta_i) satisfying:
    alpha_i² + beta_i² = 1

Interpretation:
    P(feature i is selected) = beta_i²

Initialisation: alpha = beta = 1/√2  →  50 % prior selection probability.

Algorithm outline (per generation)
-----------------------------------
1. Observe:  collapse each Q-individual's Q-bits into a binary chromosome
             by sampling Bernoulli(beta²) for each feature.
2. Evaluate: score each chromosome with 3-fold CV balanced accuracy using
             a RandomForestClassifier on the selected feature subset.
3. Update best: track the globally best chromosome seen so far.
4. Rotate:   apply the Quantum Rotation Gate to nudge Q-bits toward the
             best solution wherever the current chromosome is inferior.
5. Mutate:   flip Q-bits stochastically to escape premature convergence.

Reference: Han, K.-H. & Kim, J.-H. (2002). Quantum-inspired evolutionary
           algorithm for a class of combinatorial optimization.
           IEEE Trans. Evol. Comput., 6(6), 580-593.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
POPULATION_SIZE = 20
N_GENERATIONS   = 60
ROTATION_ANGLE  = 0.05 * np.pi   # |Δθ| per gate application
MIN_FEATURES    = 5               # floor: prevents degenerate all-zero chromosomes
MAX_FEATURES    = 500             # ceiling: keeps 3-fold CV tractable
MUTATION_RATE   = 0.005           # per-Q-bit probability of a swap mutation
RANDOM_SEED     = 42


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------
@dataclass
class Individual:
    """
    One member of the Q-population.

    Stores n_features Q-bits as two amplitude vectors (alpha, beta) and
    the most recently observed binary chromosome.
    """
    n_features: int
    alpha: np.ndarray = field(init=False)   # shape (n_features,)
    beta:  np.ndarray = field(init=False)   # shape (n_features,)
    chromosome: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Initialise all Q-bits at 45°: equal probability of being 0 or 1
        angle = np.full(self.n_features, np.pi / 4)
        self.alpha = np.cos(angle)   # = 1/√2 for each feature
        self.beta  = np.sin(angle)   # = 1/√2 for each feature

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def selection_probs(self) -> np.ndarray:
        """P(feature i selected) = beta_i²  ∈ [0, 1]."""
        return self.beta ** 2

    # ------------------------------------------------------------------
    # Observation (wavefunction collapse)
    # ------------------------------------------------------------------
    def observe(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a binary chromosome from the current Q-bit distribution.

        Each bit is drawn independently: x_i ~ Bernoulli(beta_i²).
        A clamping step keeps the number of selected features in
        [MIN_FEATURES, MAX_FEATURES] so the fitness call is always
        meaningful and fast.
        """
        probs      = self.selection_probs
        chromosome = (rng.random(self.n_features) < probs).astype(np.int8)
        n_selected = chromosome.sum()

        # Too few selected → force-enable the highest-probability OFF features
        if n_selected < MIN_FEATURES:
            off_idx = np.where(chromosome == 0)[0]
            top_off = off_idx[np.argsort(probs[off_idx])[::-1]]
            chromosome[top_off[: MIN_FEATURES - n_selected]] = 1

        # Too many selected → disable the lowest-probability ON features
        if n_selected > MAX_FEATURES:
            on_idx  = np.where(chromosome == 1)[0]
            weak_on = on_idx[np.argsort(probs[on_idx])]          # weakest first
            chromosome[weak_on[: n_selected - MAX_FEATURES]] = 0

        self.chromosome = chromosome
        return chromosome


# ---------------------------------------------------------------------------
# Best solution tracker
# ---------------------------------------------------------------------------
@dataclass
class BestSolution:
    chromosome: np.ndarray
    fitness:    float


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------
SPARSITY_PENALTY = 0.01   # maximum accuracy penalty at full feature selection


def evaluate_fitness(
    chromosome: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    rng_seed: int = 0,
) -> float:
    """
    Score a binary chromosome using 3-fold CV balanced accuracy minus a
    sparsity penalty proportional to the fraction of features selected.

    fitness = balanced_accuracy - SPARSITY_PENALTY * (n_selected / n_total)

    The penalty is at most SPARSITY_PENALTY (0.01) when every feature is
    selected, and 0 when only one feature is chosen.  This creates gradient
    pressure toward smaller gene sets: two chromosomes with equal accuracy
    will always rank the sparser one higher, so the QEA progressively
    discards noise features rather than carrying them for free.

    Parameters
    ----------
    chromosome : int8 array of length n_features (1 = selected, 0 = ignored)
    X          : full feature matrix  (n_samples × n_features)
    y          : integer-encoded class labels
    rng_seed   : seed forwarded to RandomForestClassifier for reproducibility

    Returns
    -------
    Penalised fitness score.  Returns 0.0 if no features are selected.
    """
    selected_cols = X.columns[chromosome.astype(bool)]
    n_selected    = len(selected_cols)
    if n_selected == 0:
        return 0.0

    clf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=rng_seed,
        n_jobs=-1,
    )
    scores   = cross_val_score(
        clf, X[selected_cols].values, y, cv=3, scoring="balanced_accuracy"
    )
    accuracy = float(scores.mean())
    penalty  = SPARSITY_PENALTY * (n_selected / len(chromosome))
    return accuracy - penalty


# ---------------------------------------------------------------------------
# Quantum Rotation Gate
# ---------------------------------------------------------------------------
def apply_rotation_gate(
    individual:       Individual,
    best:             BestSolution,
    current_fitness:  float,
    delta_theta:      float = ROTATION_ANGLE,
) -> None:
    """
    Update Q-bits in-place using the Han & Kim (2002) lookup table.

    The gate rotates (alpha_i, beta_i) by a signed angle Δθ_i determined by:
        - Whether the current bit x_i differs from the best bit b_i
        - The sign of alpha_i · beta_i
        - Whether current_fitness < best.fitness

    Rotation matrix:
        | cos(Δθ_i)  -sin(Δθ_i) |   | alpha_i |
        | sin(Δθ_i)   cos(Δθ_i) | × | beta_i  |

    The rotation direction is chosen so that beta_i² moves toward b_i,
    i.e. the Q-bit is steered to make the best solution more probable.
    """
    # No update if the current individual is already at least as good
    if current_fitness >= best.fitness:
        return

    x  = individual.chromosome   # current observed bits, shape (n_features,)
    b  = best.chromosome          # best known bits,      shape (n_features,)
    ab = individual.alpha * individual.beta  # element-wise product

    differ = x != b               # boolean mask: bits where rotation may help
    if not differ.any():
        return

    # Build signed rotation angles -------------------------------------------
    signs = np.zeros(individual.n_features, dtype=np.float64)

    # Case: x_i = 0, b_i = 1  → want beta_i to grow  → Δθ < 0 when αβ > 0
    mask_01 = differ & (x == 0)
    signs[mask_01 & (ab > 0)] = -1.0
    signs[mask_01 & (ab < 0)] =  1.0

    # Case: x_i = 1, b_i = 0  → want beta_i to shrink → Δθ > 0 when αβ > 0
    mask_10 = differ & (x == 1)
    signs[mask_10 & (ab > 0)] =  1.0
    signs[mask_10 & (ab < 0)] = -1.0

    theta   = signs * delta_theta
    cos_t   = np.cos(theta)
    sin_t   = np.sin(theta)

    new_alpha = cos_t * individual.alpha - sin_t * individual.beta
    new_beta  = sin_t * individual.alpha + cos_t * individual.beta

    # Re-normalise to guard against floating-point drift
    norm = np.sqrt(new_alpha ** 2 + new_beta ** 2)
    norm = np.where(norm == 0, 1.0, norm)
    individual.alpha = new_alpha / norm
    individual.beta  = new_beta  / norm


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------
def mutate(
    individual: Individual,
    rng:        np.random.Generator,
    rate:       float = MUTATION_RATE,
) -> None:
    """
    Swap-mutation: interchange alpha_i ↔ beta_i for randomly chosen Q-bits.

    This is equivalent to a ±π/4 rotation and inverts the current
    selection probability of the affected feature (p → 1−p).
    It prevents premature convergence when Q-bits have fully collapsed.
    """
    mask = rng.random(individual.n_features) < rate
    if not mask.any():
        return
    individual.alpha[mask], individual.beta[mask] = (
        individual.beta[mask].copy(),
        individual.alpha[mask].copy(),
    )


# ---------------------------------------------------------------------------
# Main QEA loop
# ---------------------------------------------------------------------------
def run_qea(
    X:               pd.DataFrame,
    y_raw,           # array-like of string or integer labels
    population_size: int   = POPULATION_SIZE,
    n_generations:   int   = N_GENERATIONS,
    rotation_angle:  float = ROTATION_ANGLE,
    mutation_rate:   float = MUTATION_RATE,
    random_seed:     int   = RANDOM_SEED,
    verbose:         bool  = True,
) -> dict:
    """
    Run QEA and return the best-found feature subset plus population statistics.

    Parameters
    ----------
    X               : feature matrix (n_samples × n_features)
    y_raw           : class labels (string or int)
    population_size : number of Q-individuals
    n_generations   : number of evolutionary generations
    rotation_angle  : |Δθ| for the Quantum Rotation Gate
    mutation_rate   : per-Q-bit swap-mutation probability
    random_seed     : master RNG seed
    verbose         : print per-generation progress

    Returns
    -------
    dict
        'best_chromosome'  : int8 array (length n_features), 1 = selected
        'best_fitness'     : float, balanced accuracy of the best solution
        'best_features'    : list[str], names of selected features
        'selection_probs'  : float array (length n_features), mean beta²
                             across the final population — higher = more
                             likely to be genuinely informative
        'history'          : list[float], best fitness seen up to each generation
    """
    rng        = np.random.default_rng(random_seed)
    n_features = X.shape[1]

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    # --- Initialise Q-population ---
    population: list[Individual] = [Individual(n_features) for _ in range(population_size)]
    best: Optional[BestSolution] = None
    history: list[float]         = []

    for gen in range(n_generations):
        fitnesses: list[float] = []

        # --- Step 1 & 2: Observe and evaluate each individual ---
        for ind in population:
            chromosome = ind.observe(rng)
            fitness    = evaluate_fitness(
                chromosome, X, y, rng_seed=int(rng.integers(1_000_000))
            )
            fitnesses.append(fitness)

            # --- Step 3: Update global best ---
            if best is None or fitness > best.fitness:
                best = BestSolution(chromosome=chromosome.copy(), fitness=fitness)

        gen_best = max(fitnesses)
        gen_mean = float(np.mean(fitnesses))
        history.append(best.fitness)   # best seen *up to* this generation

        if verbose:
            print(
                f"Gen {gen + 1:03d}/{n_generations} | "
                f"global_best={best.fitness:.4f}  "
                f"gen_best={gen_best:.4f}  "
                f"gen_mean={gen_mean:.4f}  |  "
                f"features_in_best={int(best.chromosome.sum())}"
            )

        # --- Step 4 & 5: Rotate and mutate ---
        for ind, fit in zip(population, fitnesses):
            apply_rotation_gate(ind, best, fit, delta_theta=rotation_angle)
            mutate(ind, rng, rate=mutation_rate)

    # --- Aggregate selection probabilities across final population ---
    mean_probs   = np.stack([ind.selection_probs for ind in population]).mean(axis=0)
    best_features = list(X.columns[best.chromosome.astype(bool)])

    return {
        "best_chromosome":  best.chromosome,
        "best_fitness":     best.fitness,
        "best_features":    best_features,
        "selection_probs":  mean_probs,
        "history":          history,
    }


# ---------------------------------------------------------------------------
# Post-run analysis
# ---------------------------------------------------------------------------
def rank_features_by_activity(
    selection_probs: np.ndarray,
    feature_names,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Rank features by mean Q-population selection probability (beta²).

    A high value means the population's Q-bits have collectively converged
    toward selecting that feature, indicating likely biological relevance.

    Parameters
    ----------
    selection_probs : mean beta² per feature from run_qea()
    feature_names   : column names of the original X DataFrame
    top_n           : how many top features to return

    Returns
    -------
    DataFrame with columns: feature, selection_probability
    """
    return (
        pd.DataFrame({"feature": feature_names, "selection_probability": selection_probs})
        .sort_values("selection_probability", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import generate_omics_dataset, get_ground_truth_features

    print("=" * 65)
    print("Quantum-Inspired Evolutionary Algorithm — Feature Selection")
    print("=" * 65)

    print("\n[1/3] Generating synthetic omics dataset …")
    X, y = generate_omics_dataset()
    ground_truth = set(get_ground_truth_features())
    print(f"      {X.shape[0]} samples × {X.shape[1]} features")
    print(f"      {len(ground_truth)} ground-truth informative features\n")

    print("[2/3] Running QEA …\n")
    results = run_qea(X, y, verbose=True)

    print("\n[3/3] Results")
    print("-" * 65)
    print(f"Best balanced accuracy : {results['best_fitness']:.4f}")
    print(f"Features in best chromosome : {len(results['best_features'])}")

    recovered    = set(results["best_features"]) & ground_truth
    false_pos    = set(results["best_features"]) - ground_truth
    print(f"\nGround-truth recovery  : {len(recovered)}/{len(ground_truth)}")
    print(f"  Recovered : {sorted(recovered)}")
    print(f"False positives (noise): {len(false_pos)}")

    print("\nTop 30 features by Q-population activity (beta² mean):")
    ranking = rank_features_by_activity(results["selection_probs"], X.columns, top_n=30)
    ranking["is_ground_truth"] = ranking["feature"].isin(ground_truth)
    print(ranking.to_string(index=False))
