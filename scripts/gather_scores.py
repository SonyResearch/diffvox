import argparse
from pathlib import Path
from functools import reduce
import yaml
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Compare multiple evaluation results")
    parser.add_argument("dirs", nargs="+", help="Directories to compare")
    parser.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args()

    dirs = list(map(Path, args.dirs))
    if len(dirs) < 2:
        dirs = list(filter(lambda d: d.is_dir(), dirs[0].iterdir()))

    metrics_pools = map(
        lambda d: set(str(p.relative_to(d).parent) for p in d.glob("**/metrics.yaml")),
        dirs,
    )
    intersect = list(reduce(lambda a, b: a & b, metrics_pools))

    read_metrics = map(
        lambda d: map(
            lambda p: yaml.safe_load(open(d / p / "metrics.yaml")), tqdm(intersect)
        ),
        dirs,
    )

    with open(args.output, "w") as f:
        keys = []
        for d, metrics in zip(dirs, read_metrics):
            for p, m in zip(intersect, metrics):
                if len(keys) == 0:
                    keys = list(m.keys())
                    f.write(f"model,track_id,{','.join(keys)}\n")
                f.write(f"{d.name},{p},{','.join(map(str, [m[k] for k in keys]))}\n")
    return


if __name__ == "__main__":
    main()
