from pathlib import Path
import torch
from torchcomp import amp2db
import torchaudio
import pyloudnorm as pyln
import yaml

from .internal import find_time_offset


def medley_vocal(
    song_id, loudness: float = -18.0, return_data=True, side_energy_threshold=-10
):
    song_id = Path(song_id)
    yaml_file = song_id / (song_id.stem + "_METADATA.yaml")
    with open(yaml_file) as f:
        hierachy = yaml.safe_load(f)

    if hierachy["instrumental"] == "yes":
        return []

    results = []
    raw_dir = hierachy["raw_dir"]
    stem_dir = hierachy["stem_dir"]
    for stem_v in hierachy["stems"].values():
        instr_name = stem_v["instrument"]
        if len(stem_v["raw"]) == 1 and any(
            map(
                lambda keyword: keyword in instr_name,
                ("speaker", "singer", "vocal", "voice", "speech"),
            )
        ):
            raw_track, *_ = tuple(stem_v["raw"].values())
            dry_file = song_id / raw_dir / raw_track["filename"]
            wet_file = song_id / stem_dir / stem_v["filename"]

            if not return_data:
                results.append((dry_file, wet_file))
                continue

            dry, sr = torchaudio.load(str(dry_file))
            if dry.size(0) > 1:
                left = dry[0]
                right = dry[1]
                left = left / left.max()
                right = right / right.max()
                side = (left - right) * 0.707
                side_energy = amp2db(side.abs().max()).item()
            else:
                side_energy = -torch.inf

            print(f"Maximum energy of side channel: {side_energy:.4f} dB")
            if side_energy > side_energy_threshold:
                print(f"Skip {dry_file}")
                continue

            wet, _ = torchaudio.load(str(wet_file))
            assert sr == _

            dry = dry[:, : wet.shape[1]]
            wet = wet[:, : dry.shape[1]]

            shifts = find_time_offset(dry.mean(0), wet.mean(0)).item()
            dry = torch.roll(dry, shifts=int(shifts), dims=1)
            print(shifts, dry.shape)

            dry = dry.mean(0, keepdim=True)

            meter = pyln.Meter(sr)
            normaliser = lambda x: pyln.normalize.loudness(
                x, meter.integrated_loudness(x), loudness
            )
            dry = torch.from_numpy(normaliser(dry.numpy().T).T).float()
            wet = torch.from_numpy(normaliser(wet.numpy().T).T).float()

            results.append((dry_file, wet_file, sr, dry, wet, shifts))
    return results
