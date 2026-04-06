"""
Agentic-Q-Tuner — pipeline entry point.

Run with:
    python main.py
"""

import sys
from pathlib import Path

# Make src/ importable regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import generate_omics_dataset, get_ground_truth_features
from agent_manager import AgentManager
from q_algorithm import rank_features_by_activity
from visualize_results import plot_convergence, plot_manhattan

DIVIDER = "=" * 70


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print(DIVIDER)
    print("STEP 1/4 — Generate synthetic omics dataset")
    print(DIVIDER)

    X, y = generate_omics_dataset()
    ground_truth = set(get_ground_truth_features())

    print(f"  Samples  : {X.shape[0]}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Classes  : {y.value_counts().to_dict()}")
    print(f"  Ground-truth informative features: {len(ground_truth)}")

    # ------------------------------------------------------------------
    # 2. Adaptive QEA via AgentManager
    # ------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("STEP 2/4 — Run Adaptive QEA (AgentManager)")
    print(DIVIDER + "\n")

    agent = AgentManager(
        X                 = X,
        y_raw             = y,
        population_size   = 20,
        rotation_angle    = 0.05 * 3.141592653589793,   # 0.05π starting angle
        mutation_rate     = 0.005,
        random_seed       = 42,
        chunk_size        = 5,
        total_generations = 60,
    )

    results = agent.run()

    # ------------------------------------------------------------------
    # 3. Results
    # ------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("STEP 3/4 — Results")
    print(DIVIDER)

    best_set  = set(results["best_features"])
    recovered = best_set & ground_truth
    false_pos = best_set - ground_truth
    missed    = ground_truth - best_set

    print(f"\n  Best balanced accuracy : {results['best_fitness']:.4f}")
    print(f"  Features in best chr.  : {len(results['best_features'])}")

    print(f"\n  Ground-truth recovery  : {len(recovered)}/{len(ground_truth)}")
    if recovered:
        print(f"    Recovered : {sorted(recovered)}")
    if missed:
        print(f"    Missed    : {sorted(missed)}")
    print(f"  False positives (noise): {len(false_pos)}")

    print(f"\n  Rotation-angle trace (end of each chunk):")
    for i, angle in enumerate(results["angle_history"], 1):
        marker = " ← increased (stall)" if (
            i > 1 and angle > results["angle_history"][i - 2] + 1e-9
        ) else ""
        print(f"    Chunk {i:02d}: {angle/3.141592653589793:.4f}π{marker}")

    print(f"\n  Top 20 features by Q-population activity (mean β²):")
    ranking = rank_features_by_activity(results["selection_probs"], X.columns, top_n=20)
    ranking["ground_truth"] = ranking["feature"].isin(ground_truth)
    print(ranking.to_string(index=False))

    print(f"\n  Full decision log : {results['log_path']}")

    # ------------------------------------------------------------------
    # 4. Visualisations
    # ------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("STEP 4/4 — Generate plots")
    print(DIVIDER)

    plot_convergence(
        history       = results["history"],
        angle_history = results["angle_history"],
        chunk_size    = 5,
    )
    plot_manhattan(
        selection_probs = results["selection_probs"],
        feature_names   = X.columns,
        ground_truth    = ground_truth,
    )

    print(DIVIDER)
    print("Pipeline complete. Outputs:")
    print(f"  Plots    → plots/")
    print(f"  Decision log → {results['log_path']}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
