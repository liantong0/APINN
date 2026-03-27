# APINN Simulation and Real-World Evaluation Suite

This repository contains the simulation and evaluation assets used to analyze **APINN** against three baseline families under increasing sensor-noise complexity and environmental uncertainty:

- Classical analytic localization
- Pure data-driven learning
- Fixed-weight PINN

The benchmark is designed to answer four core questions:

1. How robust is each method under time-varying non-Gaussian noise?
2. How well does each method generalize with sparse or biased calibration?
3. How effective is the adaptive weighting mechanism in APINN?
4. How do methods perform in real-world deployment trajectories?

## Method Overview

APINN combines physics-guided constraints with data-driven function approximation and adaptive objective balancing.

- `Physics component`: Encodes geometric/optical consistency and localization constraints.
- `Data component`: Learns residual nonlinear mappings from sensor observations.
- `Adaptive weighting`: Dynamically rebalances loss terms during optimization to improve stability across changing noise and turbidity conditions.

For fair comparison, all methods are evaluated under aligned trajectory settings, shared calibration assumptions per experiment group, and matched reporting metrics.

## Experiment Groups

The experimental protocol follows the paper-level grouping (Table 1 in manuscript context):

| Group | Folder | Main Focus | Variable Factors | Targeted Claim |
|---|---|---|---|---|
| 1A | `Simulation/1A/` | Core performance under standard setting | Reference Gaussian-like condition | Baseline localization capability |
| 1B | `Simulation/1B/` | Robustness to non-Gaussian noise | Alpha, GMM, Middleton-A noise models | APINN robustness under heavy-tailed/time-varying disturbances |
| 2A | `Simulation/2A/` | Generalization with sparse calibration | Reduced calibration support | APINN extrapolation under limited supervision |
| 2B | `Simulation/2B/` | Generalization with biased calibration | Calibration bias/mismatch | Method resilience to calibration shift |
| 3A | `Simulation/3A/` | Adaptive weighting validation (setting A) | Turbidity-dependent uncertainty | Benefit of adaptive weighting over fixed weighting |
| 3B | `Simulation/3B/` | Adaptive weighting validation (setting B) | Alternate turbidity regime | Consistency of adaptive mechanism gains |
| Real test | `Experiment/` | Deployment-level performance | Two physical trajectories | Practical effectiveness in real environments |

## Repository Layout

```text
Final code/
  Experiment/
    Track1/
      Actual_environment_trajectory_1.mat
    Track2/
      Actual_environment_trajectory_2.mat

  Simulation/
    1A/
      Gaussian_noise_main.m
      data/
      src/
        localization/
        losses/
        nn/
        optics/
        viz/

    1B/
      Alpha/
      GMM/
      MiddletonA/

    2A/
      Near_distance.m
      data/
      src/

    2B/
      Fixed_distance.m

    3A/
      Smooth_Test1.m

    3B/
      Smooth_Test2.m
      setup_paths.m
      src/
        data_generation/
        localization/
        losses/
        nn/
        optics/
        viz/
```

## Reproducibility Guide

### Environment

- MATLAB (recommended: R2021b or newer)
- No external toolbox is required beyond standard numerical and plotting capabilities used in scripts

### Quick Start

1. Open MATLAB and set working directory to the desired experiment folder, for example:
   - `Simulation/1A/`
   - `Simulation/1B/Alpha/`
   - `Simulation/3B/`
2. Run the corresponding main script:
   - `Gaussian_noise_main.m` for Group 1A
   - `Alpha_noise.m`, `GMM_noise.m`, `middletona_noise.m` for Group 1B
   - `Near_distance.m` for Group 2A
   - `Fixed_distance.m` for Group 2B
   - `Smooth_Test1.m` / `Smooth_Test2.m` for Groups 3A / 3B
3. If needed, run `setup_paths.m` first (or verify it is called by the main script).
4. Inspect outputs in each group's `data/`, `evaluation/`, or generated figures.

### Implementation Notes

- `src/localization/`: Analytic and hybrid localization utilities.
- `src/losses/`: Objective functions and gradient computation for different training modes.
- `src/nn/`: Lightweight neural-network forward/training utilities.
- `src/optics/`: Optical-model transformations and calibration-related mappings.
- `src/viz/` or `src/plotting/`: Smoothing, metrics, and visualization helpers.

## Data Availability and Confidentiality

The `Experiment/` folder provides **post-processed trajectory results** from two real-environment tests.

- Raw real-world data are not included due to confidentiality and privacy constraints.
- Only the minimum sharable artifacts required for result-level validation are released.

## Scope and Intended Use

- This repository is intended for academic verification, methodological comparison, and reproducibility support.
- Commercial reuse requires explicit authorization from the data/code owners.

## Citation

If you use this codebase or derived results in academic work, please cite the corresponding paper/manuscript and acknowledge this repository.
