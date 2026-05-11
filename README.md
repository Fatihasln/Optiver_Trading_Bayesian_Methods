# Optiver_Trading_Bayesian_Methods

Bayesian Inference for High-Frequency Market Microstructure
This repository contains the complete implementation and analysis for the paper "Bayesian Inference for High‑Frequency Market Microstructure" (Ciftaslan, 2026). The project develops a hierarchical Bayesian pipeline to predict 60‑second forward relative returns during the NASDAQ closing auction using the Optiver Trading at the Close dataset.

Problem Statement
The NASDAQ closing auction (15:50–16:00 EST) is a critical 10‑minute window where market participants submit orders that are matched at a single clearing price. The goal is to predict the 60‑second forward return of each stock relative to the synthetic market index (basis points). This relative return imposes a structural zero‑sum constraint across stocks at each moment.

Standard machine learning approaches produce point estimates without uncertainty quantification. This project uses Bayesian methods to obtain full predictive distributions, enabling calibrated confidence intervals and risk‑adjusted position sizing.

Key Contributions
Hierarchical Bayesian pipeline combining three modelling layers:

Bayesian Linear Regression (BLR) – analytically exact uncertainty for long‑range linear signals.

Monte Carlo Dropout Bayesian Neural Network (BNN) – approximate posterior for non‑linear short‑term dynamics.

Inverse‑variance ensemble (Bayesian Model Averaging).

Post‑processing improvements:

Zero‑sum adjustment to enforce the cross‑sectional constraint.

Extended lag and rolling features to capture autocorrelation.

Within‑fold uncertainty calibration (corrects systematic overconfidence).

Hidden Markov Model (HMM) for latent market regime detection, with a Mixture of Experts (MoE) that fails to improve MAE but is honestly reported.

Walk‑forward validation (purged) to prevent look‑ahead and data leakage.

Feature engineering in five groups (microstructure, price dynamics, temporal pressure, lags, cross‑sectional) – all groupby operations are strictly per‑stock and per‑day to avoid leakage.

Results (from the paper)
Model	Mean MAE (bp)	std/MAE (raw)	std/MAE (calibrated)
BLR (312 days)	2.4222	1.736	–
BNN (56 days)	2.4370	0.338	–
Ensemble (raw)	2.4223	0.317	–
Ensemble + Cal	2.3997	0.317	1.009
MoE + Cal	2.4883	0.491	1.154
Key findings:

Calibration is the most important practical contribution – std/MAE improves from 0.317 to 1.009 with no change in MAE.

BNN adds negligible MAE improvement over BLR; the feature engineering linearises most of the signal.

Cross‑sectional and lag features are the primary alpha sources (MAE+ZS / MAE = 1.04x in the final pipeline).

HMM identifies 5 interpretable market regimes with high persistence (e.g., coordinated buying regime has self‑transition probability 0.993).

Mixture of Experts yields worse MAE due to data scarcity for rare regimes in some folds – an honest negative result.

Repository Structure
text
.
├── Bayesian_Analysis_Ciftaslan.pdf   # Full paper with detailed derivations
├── Bayesian_Inference_Optiver.ipynb  # Complete runnable notebook
├── README.md                         # READ.md

Requirements
The code is written in Python 3 and uses the following main libraries:

numpy, pandas

scikit-learn (BayesianRidge, StandardScaler)

torch (PyTorch for BNN)

hmmlearn (for HMM regime detection)

matplotlib, seaborn (visualisation)

pymc, arviz (imported but not used in the final pipeline)

Install with:

bash
pip install numpy pandas scikit-learn torch hmmlearn matplotlib seaborn pymc arviz
How to Run
Download the dataset – The Optiver Trading at the Close dataset (CSV file) is required. Place it as train.csv in the working directory or adjust the path in the notebook.

Run the notebook – Open Bayesian_Inference_Optiver.ipynb in Jupyter or VS Code and execute cells in order. The notebook is fully self‑contained:

Data loading with memory‑optimised dtypes.

Exploratory data analysis.

Feature engineering (28 candidate features, then ARD‑style selection via BayesianRidge).

Walk‑forward cross‑validation configuration.

BLR, BNN, ensemble, calibration, and MoE implementations.

HMM regime analysis.

Visualisation of results (MAE, calibration, scale factors).

Expected output – The notebook prints:

MAE and std/MAE for each fold and overall.

Calibration diagnostic plots.

Final summary table (similar to the one above).

Key Implementation Notes
No data leakage: All shift, diff, rolling operations are performed inside groupby(['stock_id', 'date_id']). Cross‑sectional features use groupby(['date_id', 'seconds_in_bucket']).

Walk‑forward folds: Automatically configured based on the number of days (481 days → 4 folds).

Calibration: Computed on the last 9 days of each training set, then applied to the validation predictions. Does not affect MAE.

HMM Mixture of Experts: Uses a single global scaler and a minimum sample threshold for BNN training (8,400 rows per regime). Regimes with insufficient samples fall back to BLR only.

Citation
If you use this code or the ideas from the paper, please cite:

bibtex
@mastersthesis{Ciftaslan2026Bayesian,
  author = {Mehmet Fatih Çiftaslan},
  title  = {Bayesian Inference for High-Frequency Market Microstructure},
  school = {University of Milan},
  year   = {2026},
  type   = {Master's thesis}
}
License
This project is for academic and research purposes. Please refer to the original Optiver competition rules for dataset usage.

Author: Mehmet Fatih Çiftaslan
Course: Data Science for Economics, University of Milan, 2026
