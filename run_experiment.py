"""Run the complete experiment and generate all figures.

Usage:
    python run_experiment.py              # full paper configuration (~30 s)
    python run_experiment.py --quick      # reduced repetitions, smoke test
    python run_experiment.py --plot-only  # redraw figures from saved results

Results and figures are written to results/ (results/quick/ for --quick).
"""

import argparse

import config
from src.experiment import load_results, run_experiment, save_results
from src.plotting.plot_geometry import plot_geometry
from src.plotting.plot_risk import plot_excess_risk


def main() -> None:
    """Run the experiment and generate the paper figures."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduced number of Monte-Carlo repetitions (not the paper configuration)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="only redraw the figures from previously saved results",
    )
    args = parser.parse_args()

    output_dir = config.RESULTS_DIR / "quick" if args.quick else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / config.RESULTS_FILE

    if args.plot_only:
        print(f"Loading results from {results_path}")
        results = load_results(results_path)
    else:
        print("Running experiment...")
        results = run_experiment(quick=args.quick)
        save_results(results, results_path)
        print(f"Results saved to {results_path}")

    print("Generating geometry figure...")
    plot_geometry(results, output_dir / config.GEOMETRY_FIGURE)

    print("Generating excess-risk figure...")
    plot_excess_risk(results, output_dir / config.EXCESS_RISK_FIGURE)

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
