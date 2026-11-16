"""Scientific computations for the misspecified identification experiment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config

# =============================================================================
# Result containers
# =============================================================================


@dataclass
class RiskExperimentResult:
    """Results of the excess-risk experiment for one data-generation mode."""

    mode: str
    covariance: np.ndarray
    hypothesis_risks: np.ndarray
    target_index: int
    target_risk: float
    mean_risk: dict[str, np.ndarray]


@dataclass
class GeometryExperimentResult:
    """Results of the estimator-geometry experiment for one data mode."""

    mode: str
    estimates: dict[str, np.ndarray]


@dataclass
class ExperimentResult:
    """Complete result of the experiment."""

    risk: dict[str, RiskExperimentResult]
    geometry: dict[str, GeometryExperimentResult]


# =============================================================================
# Data generation
# =============================================================================


def generate_data(
    n: int,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n samples (regressor z = [x; u], output y) of the true system.

    Args:
        n: number of samples.
        mode: "iid" (x ~ N(0, I) drawn independently for every sample) or
            "trajectory" (one trajectory of length n started at x_0 = 0).
        rng: random generator; consumed in the order noise, inputs, states.

    Returns:
        regressors: shape (n, n_x + n_u), rows z_t = [x_t, u_t].
        outputs: shape (n, n_x), rows y_t = A x_t + B u_t + w_t.
    """

    if mode not in config.MODES:
        raise ValueError(f"Unknown mode: {mode}")

    noise = config.SIGMA_W * rng.standard_normal((n, config.A_TRUE.shape[0]))

    inputs = config.SIGMA_U * rng.standard_normal((n, config.B_TRUE.shape[1]))

    if mode == "iid":
        states = rng.standard_normal((n, config.A_TRUE.shape[0]))

        outputs = states @ config.A_TRUE.T + inputs @ config.B_TRUE.T + noise

        regressors = np.hstack([states, inputs])
        return regressors, outputs

    # Single trajectory
    trajectory = np.zeros((n + 1, config.A_TRUE.shape[0]))

    for t in range(n):
        trajectory[t + 1] = (
            config.A_TRUE @ trajectory[t] + config.B_TRUE @ inputs[t] + noise[t]
        )

    regressors = np.hstack([trajectory[:-1], inputs])
    outputs = trajectory[1:]

    return regressors, outputs


# =============================================================================
# Risk
# =============================================================================


def prediction_excess_risk(
    theta: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Prediction excess risk tr((theta - theta_*) Sigma (theta - theta_*)^T).

    Args:
        theta: parameter matrix [A, B], shape (n_x, n_x + n_u).
        covariance: regressor second-moment matrix Sigma, shape (n_z, n_z).
    """

    error = theta - config.THETA_TRUE

    return np.trace(error @ covariance @ error.T)


def compute_hypothesis_risks(
    covariance: np.ndarray,
) -> np.ndarray:
    """Compute the prediction excess risk of every hypothesis."""

    return np.array(
        [prediction_excess_risk(theta, covariance) for theta in config.HYPOTHESES]
    )


# =============================================================================
# Estimators
# =============================================================================


def ols_estimator(
    regressors: np.ndarray,
    outputs: np.ndarray,
) -> np.ndarray:
    """Ordinary least-squares estimator."""

    theta = np.linalg.lstsq(
        regressors,
        outputs,
        rcond=None,
    )[0]

    return theta.T


def exp_weight_estimator(
    regressors: np.ndarray,
    outputs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exponential-weighting estimator over the finite hypothesis class.

    Samples one hypothesis with probability proportional to
    exp(-ETA * sum of squared prediction errors).
    """

    errors = np.array(
        [np.sum((outputs - regressors @ theta.T) ** 2) for theta in config.HYPOTHESES]
    )

    # Subtract the minimum for numerical stability.
    weights = np.exp(-config.ETA * (errors - np.min(errors)))

    weights /= np.sum(weights)

    index = rng.choice(
        len(config.HYPOTHESES),
        p=weights,
    )

    return config.HYPOTHESES[index]


# =============================================================================
# Population covariance
# =============================================================================


def regressor_covariance(
    mode: str,
) -> np.ndarray:
    """Estimate the population covariance of the regressors by Monte Carlo.

    Uses N_COVARIANCE_SAMPLES samples with a fixed seed. For the trajectory
    mode this approximates the stationary covariance.
    """

    rng = np.random.default_rng(config.SEED + 1000)

    regressors, _ = generate_data(
        config.N_COVARIANCE_SAMPLES,
        mode,
        rng,
    )

    return regressors.T @ regressors / len(regressors)


# =============================================================================
# Risk experiment
# =============================================================================


def run_risk_experiment(
    mode: str,
    rng: np.random.Generator,
    n_reps: int = config.N_REPS,
) -> RiskExperimentResult:
    """Run the excess-risk Monte Carlo experiment (n_reps per sample size)."""

    covariance = regressor_covariance(mode)

    hypothesis_risks = compute_hypothesis_risks(covariance)

    target_index = int(np.argmin(hypothesis_risks))
    target_risk = hypothesis_risks[target_index]

    mean_risk = {
        "OLS": np.zeros(len(config.SAMPLE_SIZES)),
        "Exp_Weight": np.zeros(len(config.SAMPLE_SIZES)),
    }

    for sample_index, n in enumerate(config.SAMPLE_SIZES):

        risks_ols = np.zeros(n_reps)
        risks_exp_weight = np.zeros(n_reps)

        for repetition in range(n_reps):

            regressors, outputs = generate_data(
                n,
                mode,
                rng,
            )

            theta_ols = ols_estimator(
                regressors,
                outputs,
            )

            theta_exp_weight = exp_weight_estimator(
                regressors,
                outputs,
                rng,
            )

            risks_ols[repetition] = prediction_excess_risk(
                theta_ols,
                covariance,
            )

            risks_exp_weight[repetition] = prediction_excess_risk(
                theta_exp_weight,
                covariance,
            )

        mean_risk["OLS"][sample_index] = np.mean(risks_ols)

        mean_risk["Exp_Weight"][sample_index] = np.mean(risks_exp_weight)

    return RiskExperimentResult(
        mode=mode,
        covariance=covariance,
        hypothesis_risks=hypothesis_risks,
        target_index=target_index,
        target_risk=target_risk,
        mean_risk=mean_risk,
    )


# =============================================================================
# Geometry experiment
# =============================================================================


def run_geometry_experiment(
    mode: str,
    rng: np.random.Generator,
    n_reps: int = config.N_GEOMETRY_REPS,
) -> GeometryExperimentResult:
    """Generate n_reps estimator samples for the geometry plot.

    Each repetition uses a fresh data set of N_GEOMETRY_SAMPLES samples.
    """

    estimates = {
        "OLS": np.zeros((n_reps, *config.THETA_TRUE.shape)),
        "Exp_Weight": np.zeros((n_reps, *config.THETA_TRUE.shape)),
    }

    for repetition in range(n_reps):

        regressors, outputs = generate_data(
            config.N_GEOMETRY_SAMPLES,
            mode,
            rng,
        )

        estimates["OLS"][repetition] = ols_estimator(
            regressors,
            outputs,
        )

        estimates["Exp_Weight"][repetition] = exp_weight_estimator(
            regressors,
            outputs,
            rng,
        )

    return GeometryExperimentResult(
        mode=mode,
        estimates=estimates,
    )


# =============================================================================
# Complete experiment
# =============================================================================


def run_experiment(quick: bool = False) -> ExperimentResult:
    """Run the complete experiment for all data-generation modes.

    Args:
        quick: use the reduced repetition counts QUICK_N_REPS and
            QUICK_N_GEOMETRY_REPS (smoke test; not the paper configuration).
    """

    n_reps = config.QUICK_N_REPS if quick else config.N_REPS
    n_geometry_reps = config.QUICK_N_GEOMETRY_REPS if quick else config.N_GEOMETRY_REPS

    risk_results = {}
    geometry_results = {}

    for mode_index, mode in enumerate(config.MODES):

        # Independent random stream for each mode and experiment.
        risk_rng = np.random.default_rng(config.SEED + 10 * mode_index + 1)

        geometry_rng = np.random.default_rng(config.SEED + 10 * mode_index + 2)

        risk_results[mode] = run_risk_experiment(mode, risk_rng, n_reps)

        geometry_results[mode] = run_geometry_experiment(
            mode, geometry_rng, n_geometry_reps
        )

    return ExperimentResult(
        risk=risk_results,
        geometry=geometry_results,
    )


# =============================================================================
# Saving and loading results
# =============================================================================


def save_results(results: ExperimentResult, path: Path) -> None:
    """Save all experiment results to a single .npz file."""

    arrays = {}

    for mode, risk in results.risk.items():
        arrays[f"{mode}/covariance"] = risk.covariance
        arrays[f"{mode}/hypothesis_risks"] = risk.hypothesis_risks
        arrays[f"{mode}/target_index"] = np.array(risk.target_index)
        arrays[f"{mode}/target_risk"] = np.array(risk.target_risk)
        for estimator, values in risk.mean_risk.items():
            arrays[f"{mode}/mean_risk/{estimator}"] = values

    for mode, geometry in results.geometry.items():
        for estimator, values in geometry.estimates.items():
            arrays[f"{mode}/estimates/{estimator}"] = values

    np.savez(path, **arrays)


def load_results(path: Path) -> ExperimentResult:
    """Load results written by `save_results`."""

    with np.load(path) as data:
        modes = sorted(
            {key.split("/")[0] for key in data.files}, key=config.MODES.index
        )

        def sub(mode: str, group: str) -> dict[str, np.ndarray]:
            prefix = f"{mode}/{group}/"
            return {
                key[len(prefix) :]: data[key]
                for key in data.files
                if key.startswith(prefix)
            }

        risk = {
            mode: RiskExperimentResult(
                mode=mode,
                covariance=data[f"{mode}/covariance"],
                hypothesis_risks=data[f"{mode}/hypothesis_risks"],
                target_index=int(data[f"{mode}/target_index"]),
                target_risk=float(data[f"{mode}/target_risk"]),
                mean_risk=sub(mode, "mean_risk"),
            )
            for mode in modes
        }
        geometry = {
            mode: GeometryExperimentResult(
                mode=mode,
                estimates=sub(mode, "estimates"),
            )
            for mode in modes
        }

    return ExperimentResult(risk=risk, geometry=geometry)
