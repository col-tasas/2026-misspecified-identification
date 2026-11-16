<!-- PROJECT SHIELDS -->
[![arXiv][arxiv-shield]][arxiv-url]
[![MIT License][license-shield]][license-url]
[![Webpage][webpage-shield-NC]][webpage-url-NC]

# Finite-Sample LTI System Identification from a Misspecified Finite Model Set

This repository contains the code from our paper "Finite-Sample LTI System Identification from a Misspecified Finite Model Set" which can be acessed [here](https://arxiv.org/abs/2610.40387). 


If you use this project for academic work, please consider citing our publication 

    Chatzikiriakos, N., & Iannelli A. (2026). 
    Finite-Sample LTI System Identification from a Misspecified Finite Model Set
    Preprint:arXiv:2610.40387

## Running the experiment

### Requirements
Install the dependencies (preferably in a virtual environment):

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### Reproducing the figures
From any directory:

```bash
python run_experiment.py              # full configuration used in the paper 
python run_experiment.py --quick      # reduced number of repetitions, quick test 
python run_experiment.py --plot-only  # redraw the figures from previously saved results
```

All outputs are written to `results/` (`results/quick/` for `--quick`):

| File | Content |
|---|---|
| `results.npz` | all computed results (covariances, hypothesis risks, mean excess-risk curves, geometry estimates) |
| `misspecified_excess_risk.{pdf,png}` | excess risk vs. number of samples (left: i.i.d. data, right: single trajectory) |
| `misspecified_geometry.{pdf,png}` | estimators and risk regions in the (a₁₁, a₁₂)-plane |

### Adapting the setting
All parameters are in `config.py`:

- **True system:** `A_TRUE`, `B_TRUE` (system $x_{t+1} = A x_t + B u_t + w_t$).
- **Hypothesis class:** `HYPOTHESES`, a list of matrices $\theta = [A, B]$. Give `HYPOTHESIS_COLORS` one color per hypothesis.
- **Noise:** `SIGMA_W` (process noise), `SIGMA_U` (input excitation).
- **Experiment size:** `SAMPLE_SIZES`, `N_REPS` (Monte-Carlo repetitions per sample size), `N_GEOMETRY_SAMPLES` and `N_GEOMETRY_REPS` (geometry plot), `QUICK_N_REPS` and `QUICK_N_GEOMETRY_REPS` (`--quick`).
- **Estimator:** `ETA`, the learning rate of the exponential-weights estimator.
- **Figures:** `FIGURE_FORMATS`, `FIGURE_DPI`, and colors and labels.

When changing the hypothesis class or the true system, keep in mind:

- The geometry plot shows the two entries of $\theta$ listed in `SLICE`. It is exact only if all hypotheses agree with the true system in every other entry. Choose `SLICE` as the entries in which the hypotheses differ.
- Adjust the plotted window (`GEOMETRY_XLIM`, `GEOMETRY_YLIM`) and the grid for the shaded regions (`REGION_GRID_X`, `REGION_GRID_Y`) so that they contain the hypotheses.
- The figures expect both data modes in `MODES` (`"iid"` and `"trajectory"`).

## Contact
🧑‍💻 Nicolas Chatzikiriakos - [nicolas.chatzikiriakos@ist.uni-stuttgart.de](mailto:nicolas.chatzikiriakos@ist.uni-stuttgart.de)


[license-shield]: https://img.shields.io/badge/License-MIT-T?style=flat&color=blue
[license-url]: https://github.com/col-tasas/2026-stochastic-SME/blob/main/LICENSE
[webpage-shield-NC]: https://img.shields.io/badge/Webpage-Nicolas%20Chatzikiriakos-T?style=flat&logo=codementor&color=green
[webpage-url-NC]: https://nchatzikiriakos.github.io
[webpage-shield-FB]: https://img.shields.io/badge/Webpage-Felix%20Brändle-T?style=flat&logo=codementor&color=green
[webpage-url-FB]: https://www.ist.uni-stuttgart.de/institute/team/Braendle-00003/
[arxiv-shield]: https://img.shields.io/badge/arXiv-2610.40387-t?style=flat&logo=arxiv&logoColor=white&color=red
[arxiv-url]: https://arxiv.org/abs/2610.40387
[researchgate-shield-NC]: https://img.shields.io/badge/ResearchGate-Nicolas%20Chatzikiriakos-T?style=flat&logo=researchgate&color=darkgreen
[researchgate-url-NC]: https://www.researchgate.net/profile/Nicolas-Chatzikiriakos


