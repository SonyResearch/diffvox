import os
import yaml
import torch
import torchaudio
import pyloudnorm as pyln
from typing import Optional
from importlib import import_module

from modules.encoder import (
    StatisticReduction,
    LogCrest,
    LogRMS,
    LogSpread,
    LogSpectralBandwidth,
    LogSpectralCentroid,
    LogSpectralFlatness,
    Frame,
    MapAndMerge,
)
from modules.fx import hadamard


# ------------------ Normalization  functions ------------------


def apply_fade_in(x: torch.Tensor, num_samples: int = 16384):
    """Apply fade in to the first num_samples of the audio signal.

    Args:
        x (torch.Tensor): Input audio tensor
        num_samples (int, optional): Number of samples to apply fade in. Defaults to 16384.

    Returns:
        torch.Tensor: Audio tensor with fade in applied
    """
    fade = torch.linspace(0, 1, num_samples, device=x.device)
    x[..., :num_samples] = x[..., :num_samples] * fade
    return x


def batch_peak_normalize(x: torch.Tensor):
    peak = torch.max(torch.abs(x), dim=1)[0]
    x = x / peak[:, None].clamp(min=1e-8)
    return x


def batch_loudness_normalize(x: torch.Tensor, meter: pyln.Meter, target_lufs: float):
    for batch_idx in range(x.shape[0]):
        lufs = meter.integrated_loudness(
            x[batch_idx : batch_idx + 1, ...].permute(1, 0).cpu().numpy()
        )
        gain_db = target_lufs - lufs
        gain_lin = 10 ** (gain_db / 20)
        x[batch_idx, :] = gain_lin * x[batch_idx, :]
    return x


# -------- self-supervised parameter estimation model -------- #


def get_param_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: int,
    dropout: float = 0.0,
):

    # if peak_normalize:
    #    x = batch_peak_normalize(x)

    if sample_rate != 48000:
        x = torchaudio.functional.resample(x, sample_rate, 48000)

    seq_len = x.shape[-1]  # update seq_len after resampling
    # if longer than 262144 crop, else repeat pad to 262144
    # if seq_len > 262144:
    #    x = x[:, :, :262144]
    # else:
    #    x = torch.nn.functional.pad(x, (0, 262144 - seq_len), "replicate")

    # peak normalize each batch item
    # for batch_idx in range(bs):
    #     x[batch_idx, ...] /= x[batch_idx, ...].abs().max().clamp(1e-8)
    # x = x / x.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)

    mid_embeddings, side_embeddings = model(x)

    # add dropout
    if dropout > 0.0:
        mid_embeddings = torch.nn.functional.dropout(
            mid_embeddings, p=dropout, training=True
        )
        side_embeddings = torch.nn.functional.dropout(
            side_embeddings, p=dropout, training=True
        )

    # check for nan
    if torch.isnan(mid_embeddings).any():
        print("Warning: NaNs found in mid_embeddings")
        mid_embeddings = torch.nan_to_num(mid_embeddings)
    elif torch.isnan(side_embeddings).any():
        print("Warning: NaNs found in side_embeddings")
        side_embeddings = torch.nan_to_num(side_embeddings)

    # l2 normalize
    mid_embeddings = torch.nn.functional.normalize(mid_embeddings, p=2, dim=-1)
    side_embeddings = torch.nn.functional.normalize(side_embeddings, p=2, dim=-1)

    return mid_embeddings, side_embeddings


def load_param_model(ckpt_path: Optional[str] = None):

    if ckpt_path is None:  # look in tmp direcory
        ckpt_path = os.path.join(os.getcwd(), "tmp", "afx-rep.ckpt")
        os.makedirs("tmp", exist_ok=True)
        if not os.path.isfile(ckpt_path):
            # download from huggingfacehub
            os.system(
                "wget -O tmp/afx-rep.ckpt https://huggingface.co/csteinmetz1/afx-rep/resolve/main/afx-rep.ckpt"
            )
            os.system(
                "wget -O tmp/config.yaml https://huggingface.co/csteinmetz1/afx-rep/resolve/main/config.yaml"
            )

    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    encoder_configs = config["model"]["init_args"]["encoder"]

    module_path, class_name = encoder_configs["class_path"].rsplit(".", 1)
    module_path = module_path.replace("lcap", "st_ito")
    module = import_module(module_path)
    model = getattr(module, class_name)(**encoder_configs["init_args"])

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # load state dicts
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("encoder"):
            state_dict[k.replace("encoder.", "", 1)] = v

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_mfcc_feature_extractor():
    transform = torch.nn.Sequential(
        torchaudio.transforms.MFCC(
            sample_rate=44100,
            n_mfcc=25,
            melkwargs={
                "n_fft": 2048,
                "hop_length": 1024,
                "n_mels": 128,
                "center": False,
            },
        ),
        StatisticReduction(),
        torch.nn.Flatten(-2, -1),
    )
    return transform


def load_mir_feature_extractor():
    transform = torch.nn.Sequential(
        MapAndMerge(
            [
                torch.nn.Sequential(
                    Frame(2048, 1024, center=False),
                    MapAndMerge(
                        [
                            LogRMS(),
                            LogCrest(),
                            LogSpread(),
                        ],
                        dim=-2,
                    ),
                ),
                torch.nn.Sequential(
                    torchaudio.transforms.Spectrogram(
                        n_fft=2048, hop_length=1024, center=False, power=1
                    ),
                    MapAndMerge(
                        [
                            LogSpectralCentroid(),
                            LogSpectralBandwidth(),
                            LogSpectralFlatness(),
                        ],
                        dim=-2,
                    ),
                ),
            ],
            dim=-2,
        ),
        StatisticReduction(),
        torch.nn.Flatten(-2, -1),
    )
    return transform


def get_feature_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
):
    bs, chs, seq_len = x.shape
    assert chs == 2, "MFCC feature extractor expects stereo input"

    x_ms = hadamard(x)

    # Get embeddings
    embeddings = model(x_ms)

    # l2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    return embeddings[:, 0], embeddings[:, 1]
