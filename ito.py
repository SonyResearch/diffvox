import torch
import numpy as np
import torchaudio
import torch.nn.functional as F
import argparse
from pathlib import Path
import yaml
from typing import Callable, Tuple, Optional
import json
from hydra.utils import instantiate
from tqdm import tqdm
from functools import reduce
import math
import pyloudnorm as pyln
from functools import partial
from auraloss.freq import MultiResolutionSTFTLoss, SumAndDifferenceSTFTLoss

from modules.utils import chain_functions, get_chunks, vec2statedict
from st_ito.utils import (
    load_param_model,
    get_param_embeds,
    get_feature_embeds,
    load_mfcc_feature_extractor,
    load_mir_feature_extractor,
)
from utils import remove_window_fn, jsonparse2hydra
from loss.ldr import MLDRLoss


def get_reference_query_chunks(dry_audio, wet_audio, chunk_size, sr):
    dry = dry_audio.unfold(1, chunk_size, chunk_size).transpose(0, 1)
    wet = wet_audio.unfold(1, chunk_size, chunk_size).transpose(0, 1)

    max_filtered = F.max_pool1d(wet.mean(1).abs(), int(sr * 0.05), stride=1)
    active_mask = torch.quantile(max_filtered, 0.5, dim=1) > 0.001  # -60 dB
    if not active_mask.any():
        raise ValueError("No active frames")
    elif active_mask.count_nonzero() < 2:
        raise ValueError("Too few active frames")

    dry = dry[active_mask]
    wet = wet[active_mask]

    ref_audio = wet[::2].contiguous()
    raw_audio = dry[1::2].contiguous()
    return ref_audio, raw_audio


def logp_y_given_x(y, mu, std):
    cos_dist = torch.arccos(y @ mu)
    return -0.5 * (cos_dist / std).pow(2) - 0.5 * math.log(2 * math.pi) - std.log()


def one_evaluation(
    fx: torch.nn.Module,
    # afx_rep: torch.nn.Module,
    mid_side_embeds_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    to_fx_state_dict: Callable[[torch.Tensor], dict],
    logp_x: Callable[[torch.Tensor], torch.Tensor],
    init_vec: torch.Tensor,
    ref_audio: torch.Tensor,
    raw_audio: torch.Tensor,
    # sr: int,
    # chunk_size: int,
    lr: float,
    steps: int,
    weight: float,
) -> torch.Tensor:
    # ref_audio, raw_audio = get_reference_query_chunks(
    #     dry_audio, wet_audio, chunk_size, sr
    # )

    peak_scaler = 1 / ref_audio.abs().max()
    ref_audio = ref_audio * peak_scaler

    print(ref_audio.shape, raw_audio.shape)

    param_logits = torch.nn.Parameter(init_vec.clone())
    optimiser = torch.optim.Adam([param_logits], lr=lr)

    with torch.no_grad():
        # ref_mid_embs, ref_side_embs = get_param_embeds(ref_audio, afx_rep, sr)
        ref_mid_embs, ref_side_embs = mid_side_embeds_fn(ref_audio)

    with tqdm(range(steps), disable=True) as pbar:
        for i in pbar:
            cur_state_dict = to_fx_state_dict(param_logits)
            preds = (
                torch.func.functional_call(fx, cur_state_dict, raw_audio) * peak_scaler
            )
            mid_embs_pred, side_embs_pred = mid_side_embeds_fn(preds)
            # mid_embs_pred, side_embs_pred = get_param_embeds(preds, afx_rep, sr)
            # mid_mu = mid_embs_pred.mean(0)
            # mid_mu = mid_mu / mid_mu.norm()
            # side_mu = side_embs_pred.mean(0)
            # side_mu = side_mu / side_mu.norm()

            mid_cos = torch.arccos(mid_embs_pred @ ref_mid_embs.T)
            side_cos = torch.arccos(side_embs_pred @ ref_side_embs.T)

            mid_std = mid_cos.square().mean().sqrt()
            side_std = side_cos.square().mean().sqrt()

            y_x_ll = (
                logp_y_given_x(ref_mid_embs, mid_embs_pred.T, mid_std).mean()
                + logp_y_given_x(ref_side_embs, side_embs_pred.T, side_std).mean()
            )

            if weight > 0:
                x_ll = logp_x(param_logits)
                loss = -y_x_ll - x_ll * weight
            else:
                x_ll = y_x_ll.new_zeros(1)
                loss = -y_x_ll
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            postfix_dict = {
                "y_x_ll": y_x_ll.item(),
                "x_ll": x_ll.item(),
                "loss": loss.item(),
                "mid_std": mid_std.item() / math.pi * 180,
                "side_std": side_std.item() / math.pi * 180,
            }

            pbar.set_postfix(
                **postfix_dict,
            )

    print(y_x_ll.item(), x_ll.item(), loss.item())
    print(mid_std.item() / math.pi * 180, side_std.item() / math.pi * 180)
    return param_logits.detach()


@torch.no_grad()
def find_closest_training_sample(
    fx: torch.nn.Module,
    mid_side_embeds_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    to_fx_state_dict: Callable[[torch.Tensor], dict],
    training_samples: torch.Tensor,
    ref_audio: torch.Tensor,
    raw_audio: torch.Tensor,
    # sr: int,
    # chunk_size: int,
) -> torch.Tensor:
    # ref_audio, raw_audio = get_reference_query_chunks(
    #     dry_audio, wet_audio, chunk_size, sr
    # )

    peak_scaler = 1 / ref_audio.abs().max()
    ref_audio = ref_audio * peak_scaler

    print(ref_audio.shape, raw_audio.shape)

    ref_mid_embs, ref_side_embs = mid_side_embeds_fn(ref_audio)

    def reduce_closure(
        x: Tuple[float, torch.Tensor], next_param: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        cur_best_logp, cur_best_param = x
        cur_state_dict = to_fx_state_dict(next_param)
        preds = torch.func.functional_call(fx, cur_state_dict, raw_audio) * peak_scaler
        mid_embs_pred, side_embs_pred = mid_side_embeds_fn(preds)

        mid_cos = torch.arccos(mid_embs_pred @ ref_mid_embs.T)
        side_cos = torch.arccos(side_embs_pred @ ref_side_embs.T)

        mid_std = mid_cos.square().mean().sqrt()
        side_std = side_cos.square().mean().sqrt()

        y_x_ll = (
            logp_y_given_x(ref_mid_embs, mid_embs_pred.T, mid_std).mean()
            + logp_y_given_x(ref_side_embs, side_embs_pred.T, side_std).mean()
        ).item()

        return (
            (cur_best_logp, cur_best_param)
            if y_x_ll < cur_best_logp
            else (y_x_ll, next_param)
        )

    best_logp, best_param = reduce(
        reduce_closure, training_samples.unbind(0), (-float("inf"), torch.tensor([]))
    )
    print(f"Best log-likelihood: {best_logp}")
    return best_param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_analysis_dir", type=str)
    parser.add_argument("train_analysis_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--config", type=str, help="Path to fx config file")
    parser.add_argument("--chunk-duration", type=float, default=11.0)
    parser.add_argument("--weight", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--method",
        type=str,
        choices=["ito", "gt", "nn_param", "nn_emb", "mean", "regression"],
        default="ito",
    )
    parser.add_argument(
        "--encoder", type=str, default="afx-rep", choices=["afx-rep", "mfcc", "mir"]
    )
    parser.add_argument("--save-pred", action="store_true")
    parser.add_argument("--ckpt-dir", type=str)

    args = parser.parse_args()

    # load PCA
    train_analysis_folder = Path(args.train_analysis_dir).resolve()
    eval_analysis_folder = Path(args.eval_analysis_dir).resolve()
    # skops_file = train_analysis_folder / "pca.skops"
    # unknown_types = sio.get_untrusted_types(file=skops_file)
    # pca: PCA = sio.load(skops_file, trusted=unknown_types)
    # # baseline_vec = torch.tensor(pca.mean_).cuda()
    # if pca.whiten:
    #     components = np.sqrt(pca.explained_variance_[:, np.newaxis]) * pca.components_
    # else:
    #     components = pca.components_
    # components = torch.tensor(components).cuda()

    gauss_data = np.load(train_analysis_folder / "gaussian.npz")
    baseline_vec = torch.tensor(gauss_data["mean"]).cuda()
    cov = torch.tensor(gauss_data["cov"]).cuda()
    cov_logdet = cov.logdet()

    def logp_x(x):
        diff = x - baseline_vec
        b = torch.linalg.solve(cov, diff)
        norm = diff @ b
        return -0.5 * (
            norm + cov_logdet + baseline_vec.shape[0] * math.log(2 * math.pi)
        )

    print(f"Baseline logp: {logp_x(baseline_vec).item()}")

    with open(eval_analysis_folder / "info.json") as f:
        info = json.load(f)

    param_keys = info["params_keys"]
    original_shapes = list(
        map(lambda lst: lst if len(lst) else [1], info["params_original_shapes"])
    )

    *vec2dict_args, dimensions_not_need = get_chunks(param_keys, original_shapes)
    vec2dict_args = [param_keys, original_shapes] + vec2dict_args
    vec2dict = partial(
        vec2statedict,
        **dict(
            zip(
                [
                    "keys",
                    "original_shapes",
                    "selected_chunks",
                    "position",
                    "U_matrix_shape",
                ],
                vec2dict_args,
            )
        ),
    )

    if args.config is not None:
        config_path = Path(args.config).resolve()
    else:
        config_path = Path(info["runs"][0]) / "config.yaml"

    with open(config_path) as fp:
        fx_config = yaml.safe_load(fp)
    fx = instantiate(fx_config["model"])
    fx = fx.cuda()
    fx.eval()

    fx.load_state_dict(vec2dict(baseline_vec), strict=False)
    # print(fx)

    ndim_dict = {k: v.ndim for k, v in fx.state_dict().items()}
    to_fx_state_dict = lambda x: {
        k: v[0] if ndim_dict[k] == 0 else v for k, v in vec2dict(x).items()
    }

    if args.method == "regression":
        ckpt_dir = Path(args.ckpt_dir)
        with open(ckpt_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        model_config = config["model"]
        data_config = config["data"]

        checkpoints = (ckpt_dir / "checkpoints").glob("*val_loss*.ckpt")
        lowest_checkpoint = min(checkpoints, key=lambda x: float(x.stem.split("=")[-1]))
        print(f"Loading checkpoint: {lowest_checkpoint}")
        last_ckpt = torch.load(lowest_checkpoint, map_location="cpu")
        model = chain_functions(remove_window_fn, jsonparse2hydra, instantiate)(
            model_config
        )
        model.load_state_dict(last_ckpt["state_dict"])

        model = model.cuda()
        model.eval()

        train_root = Path(data_config["init_args"]["train_root"])
        try:
            param_stats = torch.load(train_root / "param_stats.pt")
        except FileNotFoundError:
            param_stats = torch.load(ckpt_dir / "param_stats.pt")

        param_mu, param_std = (
            param_stats["mu"].float().cuda(),
            param_stats["std"].float().cuda(),
        )

        regressor = lambda wet: model(wet, dry=None) * param_std + param_mu
        mid_side_embeds_fn = lambda x: (x, x)
    else:
        match args.encoder:
            case "afx-rep":
                afx_rep = load_param_model().cuda()
                mid_side_embeds_fn = lambda x: get_param_embeds(x, afx_rep, 44100)
            case "mfcc":
                mfcc = load_mfcc_feature_extractor().cuda()
                mid_side_embeds_fn = lambda x: get_feature_embeds(x, mfcc)
            case "mir":
                mir = load_mir_feature_extractor().cuda()
                mid_side_embeds_fn = lambda x: get_feature_embeds(x, mir)
            case _:
                raise ValueError(f"Unknown encoder: {args.encoder}")

    loss_fns = {
        "mss_lr": MultiResolutionSTFTLoss(
            [128, 512, 2048],
            [32, 128, 512],
            [128, 512, 2048],
            sample_rate=44100,
            perceptual_weighting=True,
        ).cuda(),
        "mss_ms": SumAndDifferenceSTFTLoss(
            [128, 512, 2048],
            [32, 128, 512],
            [128, 512, 2048],
            sample_rate=44100,
            perceptual_weighting=True,
        ),
        "mldr_lr": MLDRLoss(
            sr=44100,
            s_taus=[50, 100],
            l_taus=[1000, 2000],
        ).cuda(),
        "mldr_ms": MLDRLoss(
            sr=44100,
            s_taus=[50, 100],
            l_taus=[1000, 2000],
            mid_side=True,
        ).cuda(),
    }

    raw_params = np.load(eval_analysis_folder / "raw_params.npy")
    feature_mask = np.load(train_analysis_folder / "feature_mask.npy")
    gt_params = raw_params[:, feature_mask]

    train_params = np.load(train_analysis_folder / "raw_params.npy")
    train_index = np.load(train_analysis_folder / "train_index.npy")
    train_params = torch.from_numpy(train_params[train_index][:, feature_mask]).cuda()

    output_root = Path(args.output_dir)

    weights = []
    losses = []

    for dry_file, wet_file, shifts, gt_param in zip(
        info["dry_files"], info["wet_files"], info["alignment_shifts"], gt_params
    ):
        dry, sr = torchaudio.load(dry_file)
        wet, _ = torchaudio.load(wet_file)
        assert sr == _

        dry = dry[:, : wet.shape[1]]
        wet = wet[:, : dry.shape[1]]

        dry = torch.roll(dry, shifts=int(shifts), dims=1)
        print(shifts, dry.shape, dry_file)

        dry = dry.mean(0, keepdim=True)

        meter = pyln.Meter(sr)
        normaliser = lambda x: pyln.normalize.loudness(
            x, meter.integrated_loudness(x), -18.0
        )
        dry = torch.from_numpy(normaliser(dry.numpy().T).T).float().cuda()
        wet = torch.from_numpy(normaliser(wet.numpy().T).T).float().cuda()
        gt_param = torch.tensor(gt_param).cuda()

        match args.method:
            case "ito":
                try:
                    ref_audio, raw_audio = get_reference_query_chunks(
                        dry, wet, int(sr * args.chunk_duration), sr
                    )
                except ValueError as e:
                    print(f"Skipping {dry_file}: {e}")
                    continue
                pred_param = one_evaluation(
                    fx,
                    mid_side_embeds_fn,
                    to_fx_state_dict,
                    logp_x,
                    baseline_vec,
                    ref_audio,
                    raw_audio,
                    # wet,
                    # dry,
                    # sr,
                    # int(sr * args.chunk_duration),
                    lr=args.lr,
                    steps=args.steps,
                    weight=args.weight,
                )
            case "gt":
                pred_param = gt_param
            case "nn_param":
                pred_param = train_params[
                    torch.argmin((train_params - gt_param).square().mean(1))
                ]
            case "nn_emb":
                try:
                    ref_audio, raw_audio = get_reference_query_chunks(
                        dry, wet, int(sr * args.chunk_duration), sr
                    )
                except ValueError as e:
                    print(f"Skipping {dry_file}: {e}")
                    continue
                pred_param = find_closest_training_sample(
                    fx,
                    mid_side_embeds_fn,
                    to_fx_state_dict,
                    train_params,
                    ref_audio,
                    raw_audio,
                    # wet,
                    # dry,
                    # sr,
                    # int(sr * args.chunk_duration),
                )
            case "mean":
                pred_param = baseline_vec
            case "regression":
                try:
                    ref_audio, _ = get_reference_query_chunks(
                        dry, wet, int(sr * args.chunk_duration), sr
                    )
                except ValueError as e:
                    print(f"Skipping {dry_file}: {e}")
                    continue
                with torch.no_grad():
                    pred_param = regressor(ref_audio).mean(0)
            case _:
                raise ValueError(f"Unknown method: {args.method}")

        fx.load_state_dict(vec2dict(pred_param), strict=False)
        with torch.no_grad():
            rendered = fx(dry.unsqueeze(0)).squeeze()

        # mss_loss = mss(rendered.unsqueeze(0), wet.unsqueeze(0)).item()
        # mldr_loss = mldr(rendered, wet).item()
        loss = {
            k: f(rendered.unsqueeze(0), wet.unsqueeze(0)).item()
            for k, f in loss_fns.items()
        }
        param_mse_loss = F.mse_loss(pred_param, gt_param).item()
        loss["param_mse"] = param_mse_loss
        print(", ".join([f"{k}: {v}" for k, v in loss.items()]))
        # print(
        #     f"MSS Loss: {mss_loss}, MDR Loss: {mldr_loss}, Param MSE Loss: {param_mse_loss}"
        # )

        losses.append(loss)
        weights.append(wet.shape[1])

        dry_file = Path(dry_file)
        out_dir = output_root / dry_file.parts[-2] / dry_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "metrics.yaml", "w") as fp:
            yaml.safe_dump(
                loss,
                fp,
            )

        torch.save(pred_param.cpu(), out_dir / "pred_param.pth")

        with open(out_dir / "meta.yaml", "w") as fp:
            yaml.safe_dump(
                {
                    "model": fx_config["model"],
                    "params_keys": param_keys,
                    "params_original_shapes": original_shapes,
                    "alignment_shift": shifts,
                },
                fp,
            )

        # symbolic link
        original_wet = out_dir / "wet.wav"
        original_dry = out_dir / "dry.wav"
        if not original_wet.exists():
            original_wet.symlink_to(wet_file)
        if not original_dry.exists():
            original_dry.symlink_to(dry_file)

        if args.save_pred:
            torchaudio.save(out_dir / "pred.wav", rendered.cpu(), sr)

    weights = np.array(weights)
    weights = weights / weights.sum()

    print({k: np.array([l[k] for l in losses]) @ weights for k in losses[0].keys()})


if __name__ == "__main__":
    main()
