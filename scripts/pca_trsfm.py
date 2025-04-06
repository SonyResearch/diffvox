import argparse
import json
import numpy as np
from pathlib import Path
from functools import reduce, partial
from operator import mul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("--exclude", type=str)

    args = parser.parse_args()

    analysis_folder = Path(args.dir)

    with open(analysis_folder / "info.json") as fp:
        info = json.load(fp)
        dry_files = info["dry_files"]

    features = np.load(analysis_folder / "raw_params.npy")
    if args.exclude is not None:
        indexes = [i for i, s in enumerate(dry_files) if args.exclude not in s]
        features = features[indexes]
        print(features.shape, len(indexes))
        np.save(analysis_folder / "train_index.npy", indexes)

    # exclude dimensions in FDN U that are rebundant
    (position, _), *_ = filter(
        lambda i_k: "U.original" in i_k[1], enumerate(info["params_keys"])
    )
    original_shapes = list(
        map(lambda lst: lst if len(lst) else [1], info["params_original_shapes"])
    )
    original_chunks = list(map(partial(reduce, mul), original_shapes))
    U_matrix_shape = original_shapes[position]

    dimensions_not_need = np.ravel_multi_index(
        np.tril_indices(**dict(zip(("n", "m"), U_matrix_shape))), U_matrix_shape
    ) + sum(original_chunks[:position])

    select_mask = np.ones(features.shape[1], dtype=bool)
    select_mask[dimensions_not_need] = False
    selected_features = features[:, select_mask]
    print(selected_features.shape)

    mean = selected_features.mean(axis=0)
    centred = selected_features - mean
    cov = centred.T @ centred / (centred.shape[0] - 1)

    np.save(analysis_folder / "feature_mask.npy", select_mask)
    np.savez(
        analysis_folder / "gaussian.npz",
        mean=mean,
        cov=cov,
    )


if __name__ == "__main__":
    main()
