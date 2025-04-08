import torch
import torchaudio
import argparse
from pathlib import Path
import yaml
import json
from operator import mul
from functools import reduce
from hydra.utils import instantiate
from tqdm import tqdm
from functools import partial
from auraloss.freq import MultiResolutionSTFTLoss, SumAndDifferenceSTFTLoss
import pyloudnorm as pyln
import numpy as np

from modules.fx import clip_delay_eq_Q
from loss.ldr import MLDRLoss


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=str)
    parser.add_argument("out_csv", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--fx-config", type=str)
    parser.add_argument("--clip-Q", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-process", action="store_true")

    args = parser.parse_args()

    analysis_folder = Path(args.analysis_dir)
    with open(analysis_folder / "info.json") as f:
        info = json.load(f)

    dry_files = info["dry_files"]
    wet_files = info["wet_files"]
    alignment_shifts = info["alignment_shifts"]
    raw_params = torch.from_numpy(np.load(analysis_folder / "raw_params.npy")).cuda()
    if args.subset is not None:
        with open(Path(args.subset) / "info.json") as f:
            subset_info = json.load(f)
        subset_files = subset_info["dry_files"]
        indexes, dry_files, wet_files, alignment_shifts = zip(
            *[
                (i, d, w, s)
                for i, (d, w, s) in enumerate(
                    zip(dry_files, wet_files, alignment_shifts)
                )
                if d in subset_files
            ]
        )
        print(indexes)
        raw_params = raw_params[list(indexes)]

    param_keys = info["params_keys"]
    original_shapes = list(
        map(lambda lst: lst if len(lst) else [1], info["params_original_shapes"])
    )
    chunks = list(map(partial(reduce, mul), original_shapes))
    vect2dict = lambda x: dict(
        zip(
            param_keys,
            map(
                lambda x, shape: x.reshape(*shape),
                torch.split(x, chunks),
                original_shapes,
            ),
        )
    )

    if args.no_process:
        fx = lambda x: torch.cat([x, x], 1)
    else:
        if args.fx_config is not None:
            config_path = Path(args.fx_config)
        else:
            config_path = Path(info["runs"][0]) / "config.yaml"

        with open(config_path) as fp:
            fx_config = yaml.safe_load(fp)
        fx = instantiate(fx_config["model"])
        if args.clip_Q:
            fx.apply(partial(clip_delay_eq_Q, Q=0.707))
        if not args.cpu:
            fx = fx.cuda()
        fx.eval()

    # loss_config = fx_config["loss_fn"]["loss_fns"]
    metrics = {
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

    losses = []
    for dry_file, wet_file, shifts, params in tqdm(
        zip(dry_files, wet_files, alignment_shifts, raw_params.unbind(0))
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
        dry = torch.from_numpy(normaliser(dry.numpy().T).T).float()
        wet = torch.from_numpy(normaliser(wet.numpy().T).T).float().cuda()

        if not args.cpu:
            dry = dry.cuda()
        else:
            params = params.cpu()

        state_dict = vect2dict(params)

        if not args.no_process:
            fx.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            rendered = fx(dry.unsqueeze(0)).cuda()

        if torch.any(torch.isnan(rendered)):
            print("NAN", dry_file)
            continue

        if torch.any(torch.isinf(rendered)):
            print("INF", dry_file)
            continue

        loss = {k: f(rendered, wet.unsqueeze(0)).item() for k, f in metrics.items()}
        losses.append(loss)

    with open(args.out_csv, "w") as f:
        keys = list(metrics.keys())
        f.write(f"track_id,stem_id,{','.join(keys)}\n")
        for dry, metrics in zip(map(Path, dry_files), losses):
            d = dry.parent.name
            p = dry.stem
            f.write(f"{d},{p},{','.join(map(str, [metrics[k] for k in keys]))}\n")


if __name__ == "__main__":
    main()
