"""
Synthetic high-dimensional omics dataset generator.

Produces 50 samples x 5000 features, where exactly 20 features are
mathematically tied to a binary target ('Disease' vs 'Healthy').
All other 4980 features are pure noise.
"""

import numpy as np
import pandas as pd

# Dataset dimensions
N_SAMPLES = 50
N_FEATURES = 5000
N_INFORMATIVE = 20
RANDOM_SEED = 42

# Indices of the 20 ground-truth informative features (first 20)
INFORMATIVE_FEATURE_INDICES = list(range(N_INFORMATIVE))


def generate_omics_dataset(
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    n_informative: int = N_INFORMATIVE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a synthetic omics dataset with a known ground truth.

    Parameters
    ----------
    n_samples : int
        Number of observations (samples/patients).
    n_features : int
        Total number of features (genes/proteins).
    n_informative : int
        Number of features that are mathematically tied to the target.
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    X : pd.DataFrame, shape (n_samples, n_features)
        Feature matrix. Columns are named 'gene_0000' … 'gene_4999'.
        Columns 'gene_0000' … 'gene_0019' are the informative features.
    y : pd.Series, shape (n_samples,)
        Binary target: 0 = 'Healthy', 1 = 'Disease'.
    """
    rng = np.random.default_rng(random_seed)

    # Balanced binary labels: first half Healthy, second half Disease
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    rng.shuffle(labels)

    # --- Noise features (columns n_informative … n_features-1) ---
    # Drawn from N(0, 1); no relationship with labels.
    X_noise = rng.standard_normal((n_samples, n_features - n_informative))

    # --- Informative features (columns 0 … n_informative-1) ---
    # Effect sizes are drawn from U(0.4, 0.8) — deliberately small so the
    # class means overlap substantially with the within-class spread.
    # Within-class std is 1.5 (vs 1.0 for noise), further compressing the
    # signal-to-noise ratio: Cohen's d per feature ≈ 0.4–0.8σ / 1.5 ≈ 0.27–0.53.
    # This prevents trivial early convergence and forces the QEA to
    # discriminate genuine signal from noise over many generations.
    effect_sizes  = rng.uniform(0.4, 0.8, size=n_informative)
    X_informative = rng.standard_normal((n_samples, n_informative)) * 1.5
    shifts = np.where(labels[:, np.newaxis] == 1, effect_sizes, -effect_sizes)
    X_informative += shifts

    # Concatenate: informative first, then noise
    X_raw = np.hstack([X_informative, X_noise])

    # Build named column index
    col_names = [f"gene_{i:04d}" for i in range(n_features)]
    X = pd.DataFrame(X_raw, columns=col_names)
    y = pd.Series(
        labels,
        name="target",
        dtype="int8",
    ).map({0: "Healthy", 1: "Disease"})

    return X, y


def get_ground_truth_features(n_informative: int = N_INFORMATIVE) -> list[str]:
    """Return the names of the known informative features."""
    return [f"gene_{i:04d}" for i in range(n_informative)]


if __name__ == "__main__":
    X, y = generate_omics_dataset()

    print(f"X shape : {X.shape}")
    print(f"y distribution:\n{y.value_counts()}")
    print(f"\nInformative features : {get_ground_truth_features()[:5]} … (first 20)")
    print(f"\nFirst 3 informative features (Disease mean vs Healthy mean):")
    for gene in get_ground_truth_features()[:3]:
        disease_mean = X.loc[y == "Disease", gene].mean()
        healthy_mean = X.loc[y == "Healthy", gene].mean()
        print(f"  {gene}: Disease={disease_mean:.3f}, Healthy={healthy_mean:.3f}")
