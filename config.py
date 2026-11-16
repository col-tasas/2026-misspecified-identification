"""Configuration for the misspecified finite-class identification experiment.

All parameters of the experiment, the figures and the output locations are
collected here.
"""

from pathlib import Path

import numpy as np

# =============================================================================
# True system  x_{t+1} = A x_t + B u_t + w_t
# =============================================================================

A_TRUE = np.array(
    [
        [0.99, 0.20],
        [0.00, 0.10],
    ]
)

B_TRUE = np.array(
    [
        [0.50],
        [0.20],
    ]
)

# theta = [A, B], shape (n_x, n_x + n_u)
THETA_TRUE = np.hstack([A_TRUE, B_TRUE])

N_X, N_U = B_TRUE.shape

# =============================================================================
# Hypothesis class
# =============================================================================

# The true system is intentionally not contained in the hypothesis class.
HYPOTHESES = [
    np.array(
        [
            [0.90, 0.20, 0.50],
            [0.00, 0.10, 0.20],
        ]
    ),
    np.array(
        [
            [0.99, -0.05, 0.50],
            [0.00, 0.10, 0.20],
        ]
    ),
    np.array(
        [
            [1.25, 0.55, 0.50],
            [0.00, 0.10, 0.20],
        ]
    ),
    np.array(
        [
            [0.60, 0.00, 0.50],
            [0.00, 0.10, 0.20],
        ]
    ),
]

# Entries (row, column) of theta in which the hypotheses differ.
# Used for the two-dimensional geometry plot.
SLICE = [
    (0, 0),  # a_11
    (0, 1),  # a_12
]

# =============================================================================
# Noise
# =============================================================================

SIGMA_W = 1 / np.sqrt(2)  # process-noise standard deviation
SIGMA_U = 1.0  # input standard deviation

# =============================================================================
# Experiment
# =============================================================================

MODES = ("iid", "trajectory")

SAMPLE_SIZES = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])

# Monte-Carlo repetitions per sample size (excess-risk experiment).
N_REPS = 2000

# Samples per data set and repetitions for the geometry experiment.
N_GEOMETRY_SAMPLES = 250
N_GEOMETRY_REPS = 2000

# Number of samples used to estimate the population covariance of the
# regressors by Monte Carlo.
N_COVARIANCE_SAMPLES = 200_000

# Reduced repetitions for `run_experiment.py --quick` (smoke test only).
QUICK_N_REPS = 200
QUICK_N_GEOMETRY_REPS = 200

# =============================================================================
# Reproducibility
# =============================================================================

SEED = 0

# =============================================================================
# Estimator parameters
# =============================================================================

# Exponential-weighting learning rate.
ETA = 0.25

# =============================================================================
# Plotting
# =============================================================================

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Colors assigned to hypotheses H1, ..., H4.
HYPOTHESIS_COLORS = [
    OKABE_ITO["bluish_green"],
    OKABE_ITO["yellow"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["blue"],
]

COLOR_ESTIMATOR = {
    "OLS": OKABE_ITO["reddish_purple"],
    "Exp_Weight": OKABE_ITO["orange"],
}
COLOR_TARGET = "#FFFFFF"  # true system (drawn with a black edge)
COLOR_REFERENCE = OKABE_ITO["black"]

LABEL_TARGET_HYPOTHESIS = "KL-optimal hypothesis"

# Geometry plot: plotted window and resolution of the region grid.
GEOMETRY_XLIM = (0.45, 1.35)
GEOMETRY_YLIM = (-0.10, 0.65)
REGION_GRID_X = (0.3, 1.5)
REGION_GRID_Y = (-0.4, 0.9)
REGION_GRID_RESOLUTION = 1000

# =============================================================================
# Output
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = "results.npz"

# Figure file names without extension; one file per format is written.
GEOMETRY_FIGURE = "misspecified_geometry"
EXCESS_RISK_FIGURE = "misspecified_excess_risk"
FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 600  # resolution of PNG output
