# dislocation-equilibrium-simulation

# Dislocation Equilibrium in the Presence of a Grain Boundary and Precipitates

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the computational Python framework accompanying the paper:
> **"Dislocation equilibrium positions on either side of a grain boundary in the presence of precipitates"** > *Anantha Lakshmi Prasanna Tatavarty, Sushil Mishra, Amit Singh* > *Department of Mechanical Engineering, Indian Institute of Technology Bombay*

## 📖 Overview

The fine precipitation microstructure in age-hardenable alloys heavily influences their mechanical properties. This codebase determines the equilibrium positions of edge dislocations in a bicrystal under the combined influence of:
1. **A Grain Boundary:** Represented by a disclination dipole array.
2. **Inter-dislocation Interactions:** Forces exerted between dislocations.
3. **Misfitting Elliptical Precipitates:** Modeled mathematically as Eshelby inclusions.

Using the **superposition principle of linear elasticity**, the script calculates the total stress field. The equilibrium states are then obtained by identifying the minima of the total elastic energy ($\Delta \bar{E}_{el}$) using the **Peach-Koehler (PK) framework**.

The code is designed to evaluate multiple spatial arrangements, specifically noting **co-quadrant** and **opposite-quadrant** precipitate configurations, to predict whether the dislocation reaches a near-field or far-field stable equilibrium.

## ⚙️ Mathematical & Computational Framework

To ensure high performance across large parameter sweeps (sweeping through various misorientation angles $\omega$, eigenstrains $\epsilon^*$, and initial dislocation positions $\bar{d}$), the code includes several optimizations:
* **Gauss-Legendre Quadrature:** The complex area-under-the-curve (AUC) interaction integrals are approximated using multi-point Gaussian quadrature rather than continuous integration.
* **Symmetry Caching:** Numerical integration of the auxiliary tensor caches mathematically identical tensor components to drastically reduce solver time.
* **Global Tensor Pre-computation:** The isotropic stiffness tensor ($C_{ijkl}$) is initialized globally.
* **Parallel Processing:** Calculates combinations across all available CPU cores using Python's `multiprocessing` pool.

## 🛠️ Prerequisites & Installation

The simulation is built in Python 3. You will need the following libraries to run the script:

- `numpy`
- `scipy`
- `matplotlib` (for plotting)
- `tqdm` (for parallel computation progress bars)

You can install the dependencies via pip:
```bash
pip install numpy scipy matplotlib tqdm

## 🚀 Usage

The main script is set up for headless execution (no GUI popups), making it ideal for both local machines and high-performance computing (HPC) clusters.

1. Clone the repository to your local machine.
2. Modify the geometric or material parameters at the top of the script if necessary (defaults are set for typical Aluminum alloys, e.g., $\mu = 26.97$ GPa, $\nu = 0.33$).
3. Adjust your arrays (`d_bar_values`, `eigen_values`, `omega_values`).
4. Run the script:

    python dislocation_equilibrium.py

## 📊 Output Data

The script streams the computed results directly to a CSV file (e.g., `Data.csv`). For every combination of $\bar{d}$, $\epsilon^*$, and $\omega$, it outputs:
* `P_BAR`: The array of investigated vertical dislocation glide positions ($\bar{p}$).
* `Delta_Eel`: The calculated change in total elastic energy.
* `Delta_Eel_no_last_term`: The baseline analytical energy excluding the precipitate interaction integral.

## 🎓 Authors

* **Anantha Lakshmi Prasanna Tatavarty** (email: prasannatatavarty@gmail.com)
* **Sushil Mishra** (email: sushil.mishra@iitb.ac.in)
* **Amit Singh** (email: amit.k.singh@iitb.ac.in)

*Department of Mechanical Engineering, Indian Institute of Technology Bombay, Powai, Mumbai, Maharashtra, India.*

## 📄 Citation

If you utilize this code or our methodology in your research, please cite our corresponding paper:

    @article{tatavarty2026dislocation,
      title={Dislocation equilibrium positions on either side of a grain boundary in the presence of precipitates},
      author={Tatavarty, Anantha Lakshmi Prasanna and Mishra, Sushil and Singh, Amit},
      journal={Journal of Applied Mechanics},
      year={2026}
    }
