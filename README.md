# Differentiable Vocal Effects Model
[![arXiv](https://img.shields.io/badge/arXiv-2504.14735-b31b1b.svg)](https://arxiv.org/abs/2504.14735)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/yoyolicoris/diffvox)

The accompanying code for the paper *DiffVox: A Differentiable Model for Capturing and Analysing Professional Effects Distributions* (under review).


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


## Vocal effects style transfer evaluation

The section describe how to perform the same experiments as in the paper [Improving Inference-Time Optimisation for Vocal Effects Style Transfer with a Gaussian Prior](https://arxiv.org/abs/2505.11315).
Please make sure you have done previous steps up until [Collecting presets from multiple training runs](#collecting-presets-from-multiple-training-runs).

We use the script `ito.py` to do vocal effects style transfer on the MedleyDB vocals dataset.
It would create a folder that stores the evaluation results with each track as a subfolder.

### Oracle

The oracle model uses the presets derived in the previous step [Collecting presets from multiple training runs](#collecting-presets-from-multiple-training-runs) to process the corresponding tracks.
It serves as an upper bound for the performance of the ITO methods.
The evaluation results are the same as in the [Evaluation](#evaluation) step.

```bash
python -W ignore ito.py selected-runs/medley_vox_0919-0926/ presets/internal/ output_dir/ --config presets/fx_config.yaml --method oracle
```

### Mean

This baseline uses the mean of the parameters in the internal datset to process every target track.

```bash
python -W ignore ito.py selected-runs/medley_vox_0919-0926/ presets/internal/ output_dir/ --config presets/fx_config.yaml --method mean
```

### Nearest neighbour in parameter space (**NN-$\theta$**)

This baseline picks a preset from the internal dataset that is most similar to the target presest.

```bash
python -W ignore ito.py selected-runs/medley_vox_0919-0926/ presets/internal/ output_dir/ --config presets/fx_config.yaml --method nn_param
```

### Nearest neighbour in embedding space (**NN-***)

The following command evaluate the nearest neighbour baselines with different embeddings.

```bash
python -W ignore ito.py selected-runs/medley_vox_0919-0926/ presets/internal/ output_dir/ --config presets/fx_config.yaml --method nn_emb --encoder [encoder_type]
```

`encoder_type` can be one of the following:
- `afx-rep`: Corresponds to **NN-AFx-Rep** in the paper.
- `mfcc`: Corresponds to **NN-MFCC** in the paper.
- `mir`: Corresponds to **NN-MIR** in the paper.

### Regression

We trained a simple CNN model on the internal dataset to predict the parameters from a given processed audio.
The pre-trained weights are in the folder [`reg-ckpts`](reg-ckpts/).

```bash
python -W ignore ito.py selected-runs/medley_vox_0919-0926/ presets/internal/ output_dir/ --config presets/fx_config.yaml --method regression --ckpt-dir reg-ckpts/
```

### ITO with Gaussian prior

This method uses the Gaussian prior computed from the internal vocal presets to regularise the inference-time optimisation (ITO) process.

```bash
python -W ignore ito.py selected-runs/medley_vox_0919-0926/ presets/internal/ output_dir/ --config presets/fx_config.yaml --method ito --encoder [encoder_type] --weight 0.1
```

The `--weight` argument corresponds to $\alpha$ in the paper, which is the weight of the Gaussian prior.
Other arguments that can be passed to the script are:
- `--lr`: The learning rate for the Adam optimiser. Default is `0.01`.
- `--steps`: The number of optimisation steps. Default is `1000`.

### Gather results

Once you have folders of the evaluation results for each method, you can gather the results into a single CSV file for later analysis.

```bash
python scripts/gather_scores.py model_A/ model_B/ ... model_N/ -o results.csv
```

> **_Note:_**
> - The evaluation results of oracle, mean, and NN-$`\theta`$ baselines may contain tracks that are not evaluated by the other methods, as they do not need to split the audio into segments for the evaluation setting we described in the paper. The statistics we report in the paper are computed on the common tracks that all methods evaluated.

### Additional information on listening test

The following table list the track combinations used in the listening test for the paper.


| Raw Vocal Track | Processed Vocal Track (Reference) |
|:----------------:|:---------------------------------:|
| LizNelson_ImComingHome | Torres_NewSkin |
| MusicDelta_Disco | StevenClark_Bounty |
| StevenClark_Bounty | HopAlong_SisterCities |
| HopAlong_SisterCities | MusicDelta_Grunge |
| MusicDelta_Grunge | MusicDelta_Gospel |
| MusicDelta_Gospel | LizNelson_Rainfall |
| LizNelson_Rainfall | BrandonWebster_YesSirICanFly |
| BrandonWebster_YesSirICanFly | MusicDelta_Rockabilly |
| MusicDelta_Rockabilly | PortStWillow_StayEven |
| PortStWillow_StayEven | CatMartino_IPromise |
| ClaraBerryAndWooldog_WaltzForMyVictims | MusicDelta_80sRock |
| MusicDelta_80sRock | TheScarletBrand_LesFleursDuMal |
| TheScarletBrand_LesFleursDuMal | BigTroubles_Phantom |
| BigTroubles_Phantom | StrandOfOaks_Spacestation |
| StrandOfOaks_Spacestation | MusicDelta_Beatles |
| MusicDelta_Beatles | MutualBenefit_NotForNothing |
| MutualBenefit_NotForNothing | MidnightBlue_StarsAreScreaming |
| LizNelson_Coldwar | MusicDelta_Britpop |
| MusicDelta_Britpop | AClassicEducation_NightOwl |


## Additional materials

- [visualisation.ipynb](visualisation.ipynb): This notebook contains scripts to reproduce the figures and analysis in the DAFx paper.
- [Evaluation raw data](https://docs.google.com/spreadsheets/d/1ksSylBki1151pLR4-GebQBUYForlUKAy20fUdlyADhA/edit?usp=sharing): The raw data of the evaluation metrics per track for the DAFx paper and the spearman correlation coefficients between the parameters on the two datasets.
- [Listening test website](https://yoyolicoris.github.io/vocal-fx-mushra/): The website for the listening test of the vocal effects style transfer methods.
- [Evaluation raw scores (WASPAA)](/eval_data/raw_scores.csv): The data we used to compute the evaluation metrics in the WASPAA paper, processed from the step [Gather results](#gather-results).


## Citation
 ```bibtex
@misc{ycy2025diffvox,
      title={DiffVox: A Differentiable Model for Capturing and Analysing Professional Effects Distributions}, 
      author={Chin-Yun Yu and Marco A. Martínez-Ramírez and Junghyun Koo and Ben Hayes and Wei-Hsiang Liao and György Fazekas and Yuki Mitsufuji},
      year={2025},
      eprint={2504.14735},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2504.14735}, 
}
```
