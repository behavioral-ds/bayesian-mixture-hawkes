# Bayesian Mixture Hawkes

Code accompanying the paper "# What Drives Online Popularity: Author, Content or Sharers? Estimating Spread Dynamics with Bayesian Mixture Hawkes" [[Calderon P, Rizoiu MA]](https://arxiv.org/pdf/2406.03390).

## Description

This repo contains data and the Stan and Python implementation of the Bayesian Mixture Hawkes (BMH) model in the paper. The repo is divided into two parts: `model_inference/` and `headline_style_profiling/`.

### Model Inference
`model_inference/` contains the Stan and Python implementation of the BMH-P (popularity) and BMH-K (kernel) models. There are three subdirectories:
* `bmhp/` contains the BMH-P implementation and RNIX/CNIX cascade sizes.
* `bmhk_rnix/` contains the BMH-K implementation and RNIX inter-arrival distribution data.
* `bmhk_cnix/` contains the BMH-K implementation and CNIX inter-arrival distribution data.

Each `model_inference/bmh*/` folder consists of the following folders:
* `data/` contains the pertinent data files for fitting
* `stan_files/` contains the Stan code for the BMH model variants
* `log/` holds log files upon running
* `output/` holds (fitting) output files from running `run_bayesianfit.sh`
* `preds` holds (prediction) output files from running `run_bayesianpred.sh`
* `tracker/` stores tracker files when running `run_bayesianfit.sh` on PBS
* `tracker_pred/`stores tracker files when running `run_bayesianpred.sh` on PBS
* `metrics_mean/` and `preds_mean/`contain holdout likelihood and cascade size predictions for BMH-K and BMH-P, resp.

Pipeline to run inference and prediction:
1. Compile the Stan models using `compile_models.py` .
2. Run BMH fitting on data with `multiple_run_bayesianfit.sh`
3. Obtain BMH predictions with `multiple_run_bayesianpred.sh`
4. Run `collect_mean.py` to get mean metrics.
5. (BMH-P) Run `collect_pred.py` to get cascade size predictions.

### Headline Style Profiling
`headline_style_profiling/`contains the Python resources for the headline style profiling case study in the paper. There are two subdirectories.
- `inflammatory_fakenix/` contains resources for the CNIX analysis.
- `inflammatory_rnix/`  contains resources for the RNIX analysis.

Raw headline data is in `inflammatory_fakenix/data/`.
Code to generate the graphs in the case study are in `inflammatory_rnix/theta/combined_analysis.ipynb`.

## License

Both dataset and code are distributed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license](https://creativecommons.org/licenses/by-nc/4.0/). If you require a different license, please contact us at <piogabrielle.b.calderon@student.uts.edu.au> or <Marian-Andrei@rizoiu.eu>.

