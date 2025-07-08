# Differentiable Vocal Effects Model
[![arXiv](https://img.shields.io/badge/arXiv-2504.14735-b31b1b.svg)](https://arxiv.org/abs/2504.14735)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/yoyolicoris/diffvox)

The accompanying code for the paper [DiffVox: A Differentiable Model for Capturing and Analysing Professional Effects Distributions](https://arxiv.org/abs/2504.14735) (accepted at DAFx25).


## Table of contents
- [Requirements](#requirements)
- [Environment setup](#environment-setup)
- [Retrieving effect parameters on paired data](#retrieving-effect-parameters-on-paired-data)
- [Quick start](#quick-start)
    - [Examples](#examples)
- [Processing on multiple tracks](#processing-on-multiple-tracks)
    - [MedleyDB vocals](#medleydb-vocals)
- [Collecting presets from multiple training runs](#collecting-presets-from-multiple-training-runs)
- [Evaluation](#evaluation)
- [Features for PCA analysis](#features-for-pca-analysis)
- [Preset datasets](#preset-datasets)
- [Analysis notebooks](#analysis-notebooks)
- [Additional materials](#additional-materials)


## Environment setup

```bash
pip install -r requirements.txt
```

## Retrieving effect parameters on paired data

This step is to train the proposed effects chain on given vocal tracks.

### Quick start

The following command will run the retrieval process on one track:

```bash
python main.py
```

Editing the file [`cfg/config.yaml`](cfg/config.yaml) or passing the arguments will allow the user to change the parameters of the retrieval process. 
For details on configuring the yaml file and passing arguments, please refer to the documentation of [hydra](https://hydra.cc/docs/intro/).

#### Examples

```bash
python main.py data_dir=/data/medley1/v1/Audio/AimeeNorwich_Child --dataset=medley_vocal --log_dir=~/medley_vocal_log
```
What this command does is:
- Run the retrieval process on every track in `AnimeeNorwich_Child` that:
  1. is a vocal track and 
  2. has a one-to-one mapping with a stem track.
- The training logs, best checkpoints for the lowest loss will be saved in the folder `~/medley_vocal_log`.
- Repeat running the process on the same track will create new logs subfolders `run_0`, `run_1`, etc.

The command will check if it's a valid vocal track and stop if it's not.


### Processing on multiple tracks

#### MedleyDB vocals

```bash
./scripts/train_medley.sh ./scripts/medley_vocals_list.txt ~/processed/medley_vocal_log
```
The `medley_vocals_list.txt` was created from the script `scripts/get_medley_vocal_txt.py`. 
Please edit the file paths in the txt file to point to the correct location of the MedleyDB dataset on your machine.
The results are stored in `~/processed/medley_vocal_log`.


## Collecting presets from multiple training runs

This step is to collect the retrieval parameters from multiple training runs into single file/folder for further processing.

```bash
python scripts/get_params.py --loss-thresh 4 --fluc-thresh 0.2 ~/processed/medley_vocal_log selected-runs/medley_vox_0919-0926/ 2024-09-17:00 --end_date 2024-09-23:00
```
This command will collect the retrieval parameters from the training logs in the folder `~/processed/medley_vocal_log` that have:
- minimum loss less than 4
- fluctuation `median(abs(diff(losses))` less than 0.2
- the initial loss isn't the lowest loss
- the logs are created between `2024-09-17:00` and `2024-09-23:00`

If multiple runs of the same track are found, the script will choose the latest run.
The script create a folder `selected-runs/medley_vox_0919-0926/` and store two files, `info.json` and `raw_params.npy` in it. The former contains the information of the selected runs and the latter contains the raw logits of the parameters from the selected runs with shape `(num_runs, num_params)`, and the order of the first dimension is the same as file order in the `info.json`.

> **_Note:_**
> - The `--end_date` argument is optional. If not provided, the script will collect the logs up to the current date.
> - The script assume the collected training runs in the specified time range have the same effect configuration (run with the same `config.yaml`). Please make sure the training runs are consistent.

## Evaluation

The below command will compute the individual training losses of each presets on the corresponding tracks and save the results in `scores.csv`.
```bash
python loss_eval.py selected-runs/medley_vox_0919-0926/ scores.csv
```
Optional flags:
- `--fx-config`: Manually specify the effect configuration. By default the script will use the effect configuration in the first training run in the folder. [presets/fx_config.yaml](presets/fx_config.yaml) is the one used in the paper. [presets/rt_config.yaml](presets/rt_config.yaml) replaces the FDN and delay with real-time version implemented in Numba (on CPU).
- `--cpu`: Use CPU for the evaluation. By default the script assume GPU is available.
- `--clip-Q`: This flag will clip the Q factor of the low-pass filter to 0.707 in the delay for numerical stability. This descrepancy is due to the FS approximation that does not reveal potential instability in the delay. 
- `--no-process`: This flag will skip the processing of the audio files and directly compare the raw audio with the target audio. 

## Features for PCA analysis

```bash
python scripts/pca_trsfm.py selected-runs/medley_vox_0919-0926/
```

This script will compute the mean and covariance of the parameter logits in`selected-runs/medley_vox_0919-0926/` and store two files, `feature_mask.npy` and `gaussian.npz` in the same folder.
The former contains a 1D mask $\lbrace0, 1\rbrace^{152}$ used to select the minimum set of parameters to reproduce the effect ($\mathbb{R}^{152} \to \mathbb{R}^{130}$).
The unused dimensions are the unilossless matrix `U` in the FDN due to the parameterisation and the surrogate variable $\eta$ in the Ping-Pong delay.
The latter contains the sample mean $\mathbb{R}^{130}$ and covariance $\mathbb{R}^{130 \times 130}$ of the parameters in the form of a Gaussian distribution.

## Preset datasets

The preset datasets, **Internal** and **MedleyDB**, are stored in the folder [`presets`](presets/).
Both folders contain the files computed by the previous steps [collecting presets from multiple training runs](#collecting-presets-from-multiple-training-runs) and [features for PCA analysis](#features-for-pca-analysis).
The **Internal** folder contains one more numpy file `train_index.npy` which contains a 1D array of the indices $\mathbb{Z}^{365}$ of the training samples for the PCA we used in the paper.

## Analysis notebooks

- [visualisation.ipynb](visualisation.ipynb): This notebook contains scripts to reproduce the figures and analysis in the paper.

## Additional materials
- [Evaluation raw data](https://docs.google.com/spreadsheets/d/1ksSylBki1151pLR4-GebQBUYForlUKAy20fUdlyADhA/edit?usp=sharing): 
  - The raw data of the evaluation metrics per track for the paper and the spearman correlation coefficients between the parameters on the two datasets.

## Citation
 ```bibtex
@inproceedings{ycy2025diffvox,
      title={DiffVox: A Differentiable Model for Capturing and Analysing Professional Effects Distributions}, 
      author={Chin-Yun Yu and Marco A. Martínez-Ramírez and Junghyun Koo and Ben Hayes and Wei-Hsiang Liao and György Fazekas and Yuki Mitsufuji},
      year={2025},
      booktitle={Proc. Digital Audio Effects (DAFx-25)},
}
```
