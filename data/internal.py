import yaml
import torch
import torchaudio
from pathlib import Path
from torchcomp import amp2db
import pyloudnorm as pyln


def find_time_offset(x: torch.Tensor, y: torch.Tensor):
    x = x.double()
    y = y.double()
    N = x.size(-1)
    M = y.size(-1)

    X = torch.fft.rfft(x, n=N + M - 1)
    Y = torch.fft.rfft(y, n=N + M - 1)
    corr = torch.fft.irfft(X.conj() * Y)
    shifts = torch.argmax(corr, dim=-1)
    return torch.where(shifts >= N, shifts - N - M + 1, shifts)


def internal_vocals(
    song_id, loudness: float = -18.0, return_data=True, side_energy_threshold=-10
):
    wet_subdir = "multi"
    dry_subdir = "dry"
    song_id = Path(song_id)

    dict_filt = lambda d, f: {k: v for k, v in d.items() if f(k, v)}

    match_yaml = song_id / "correspondence_pp.yaml"
    results = []
    if match_yaml.exists():
        with open(match_yaml) as f:
            try:
                hierachy = yaml.safe_load(f)
            except:
                print(f"skip {song_id} as the yaml file format is not correct")
                return []
            one2one = dict_filt(hierachy["matched"], lambda _, v: len(v) == 1)
            vocal_only = dict_filt(
                one2one,
                lambda k, _: (
                    "lv" in k.lower()
                    or "ld" in k.lower()
                    or "lead_vocal" in k.lower()
                    or "bv" in k.lower()
                )
                and not (
                    "synth" in k.lower()
                    or "gtr" in k.lower()
                    or "guitar" in k.lower()
                    or "piano" in k.lower()
                    or "strings" in k.lower()
                    or "eg" in k.lower()
                    or "ag" in k.lower()
                ),
            )

            for k, v in vocal_only.items():
                wet_file = song_id / wet_subdir / k.split("/")[-1]
                dry_file = song_id / dry_subdir / v[0].split("/")[-1]

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
