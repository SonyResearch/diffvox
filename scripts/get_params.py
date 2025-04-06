import argparse
import json
import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import groupby
from functools import reduce
from tqdm import tqdm
import math
from hydra.utils import instantiate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("start_date", type=str)
    parser.add_argument("--end_date", type=str)
    parser.add_argument("--loss-thresh", default=4.0, type=float)
    parser.add_argument(
        "--fluc-thresh",
        default=math.inf,
        type=float,
    )
    parser.add_argument("--include-all", action="store_true")

    args = parser.parse_args()

    date_format = "%Y-%m-%d:%H"

    start_date = datetime.strptime(args.start_date, date_format)
    end_date = (
        datetime.strptime(args.end_date, date_format)
        if args.end_date is not None
        else datetime.now()
    )

    logs = Path(args.logs)

    runs = sorted(
        [
            reduce(
                lambda x, y: (
                    x
                    if (x[0] > int(y.stem.split("_")[1]))
                    else (int(y.stem.split("_")[1]), y)
                ),
                v,
                (-1, None),
            )[1]
            for _, v in groupby(
                filter(
                    lambda x: (datetime.fromtimestamp(x.stat().st_mtime) > start_date)
                    and (datetime.fromtimestamp(x.stat().st_mtime) < end_date),
                    logs.glob("*/**/run_*"),
                ),
                key=lambda p: str(p.parent),
            )
        ],
        key=lambda x: x.parents[1].stem,
    )
    print(len(runs))

    terminated = []
    loss_above_thresh = []
    not_converged = []
    too_fluctuated = []

    valid_runs = []
    model_config = None
    all_flat_params = []
    dry_wet_pairs = []
    for run in tqdm(runs):
        with open(run / "meta.json") as f:
            meta = json.load(f)

        relative_path = str(run.relative_to(logs))
        losses = meta["losses"]
        fluctuations = np.median(np.abs(np.diff(losses)))

        if not args.include_all:
            if meta.get("terminated_by", None) is not None:
                terminated.append((relative_path, meta["terminated_by"]))
                continue
            elif min(losses) > args.loss_thresh:
                loss_above_thresh.append((relative_path, min(losses)))
                continue
            elif losses[0] < min(losses[1:]):
                not_converged.append((relative_path, losses[0], min(losses[1:])))
                continue
            elif fluctuations > args.fluc_thresh:
                too_fluctuated.append((relative_path, fluctuations))
                continue

        valid_runs.append(run)

        with open(run / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        if model_config is None:
            model_config = cfg["model"]
        else:
            assert model_config == cfg["model"]

        m = instantiate(model_config)
        ckpt = torch.load(
            run / "checkpoint.ckpt", map_location="cpu", weights_only=True
        )
        m.load_state_dict(ckpt["best_model"], strict=True)

        m.eval()

        interested_params = {
            k: v
            for k, v in filter(lambda x: "params" in x[0], m.state_dict().items())
            if (k.split(".")[-2] == "params" or k[-8:] == "original")
        }

        param_shapes, flatten_params = zip(
            *[(tuple(v.shape), v.flatten()) for v in interested_params.values()]
        )
        all_flat_params.append(torch.cat(flatten_params))

        dry_wet_pairs.append(
            (meta["input_path"], meta["target_path"], meta["alignment_shift"])
        )

    features = torch.stack(all_flat_params).numpy()
    print(features.shape)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "info.json", "w") as fp:
        json.dump(
            {
                "runs": list(map(str, runs)),
                "dry_files": [x for x, *_ in dry_wet_pairs],
                "wet_files": [x for _, x, _ in dry_wet_pairs],
                "alignment_shifts": [x for *_, x in dry_wet_pairs],
                "params_original_shapes": list(param_shapes),
                "params_keys": list(interested_params.keys()),
                "problematic_runs": {
                    "terminated": terminated,
                    f"loss_above_{args.loss_thresh}": loss_above_thresh,
                    "not_converged": not_converged,
                    f"fluctuated_above_{args.fluc_thresh}": too_fluctuated,
                },
            },
            fp,
        )

    np.save(out / "raw_params.npy", features)

    print(
        f"Disgarded {len(terminated) + len(loss_above_thresh) + len(not_converged) + len(too_fluctuated)} files"
    )
    print(f"{len(terminated)} files are terminated before finished.")
    print(
        f"{len(loss_above_thresh)} files have losses not lower than {args.loss_thresh}"
    )
    print(f"{len(not_converged)} files didn't converge.")
    print(f"{len(too_fluctuated)} files has loss fluctuates too much.")


if __name__ == "__main__":
    main()
