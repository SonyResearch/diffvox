import torch
import torch.nn.functional as F
import torch.optim.optimizer
import torchaudio
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from itertools import accumulate, chain
from tqdm import tqdm
import logging
from pathlib import Path
from torchcomp import amp2db
import json
from copy import deepcopy
import traceback

nb_logger = logging.getLogger("numba")
nb_logger.setLevel(logging.ERROR)  # only show error

from data.internal import internal_vocal
from data.medley_db import medley_vocal
from modules.utils import remove_fx_parametrisation, chain_functions
from modules.fx import SurrogateDelay, FSSurrogateDelay


move2device = lambda *xs: tuple(x.to(xs[-1]) for x in xs[:-1])


def run(x: torch.Tensor, y: torch.Tensor, sr: int, overlap_size: int, cfg: DictConfig):
    assert sr == cfg.sr
    batch_size = cfg.batch_size
    fx_model: torch.nn.Module = instantiate(cfg.model)
    loss_fn: torch.nn.Module = instantiate(cfg.loss_fn)

    x, y, fx_model, loss_fn = move2device(x, y, fx_model, loss_fn, cfg.device)

    fx_model_copy = deepcopy(fx_model).to(cfg.device)
    fx_model_copy.eval()

    optimiser: torch.optim.Optimizer = instantiate(cfg.optimiser, fx_model.parameters())

    num_params = lambda params: sum([p.numel() for p in params])

    print(f"Number of raw params: {num_params(fx_model.parameters())}")
    print(
        f"Number of fx params: {num_params(chain.from_iterable([m.parameters() for name, m in fx_model.named_modules() if name[-6:] == 'params']))}"
    )

    delay_modules = [
        m
        for m in fx_model.modules()
        if (isinstance(m, SurrogateDelay) or isinstance(m, FSSurrogateDelay))
    ]

    def closure():
        # torch.cuda.empty_cache()
        if batch_size > 0 and x.size(0) > batch_size:
            batch_indexes = torch.randperm(x.size(0))[:batch_size]
            x_batch = x[batch_indexes]
            y_batch = y[batch_indexes]
        else:
            x_batch = x
            y_batch = y
        optimiser.zero_grad()
        y_hat = fx_model(x_batch)[..., overlap_size:]
        loss, raw_losses = loss_fn(
            y_hat.contiguous(),
            y_batch[..., overlap_size : overlap_size + y_hat.shape[-1]].contiguous(),
        )

        if len(delay_modules) and cfg.regularise_delay:
            delay_reg = sum((1 - m.log_damp.exp()).square() for m in delay_modules)
            loss = loss + delay_reg
            raw_losses.append(delay_reg.item())

        loss.backward()
        optimiser.step()
        return loss.item(), raw_losses

    disable_progress_bar = not cfg.enable_progress_bar
    terminate_condition = None
    with tqdm(range(1, cfg.epochs + 1), disable=disable_progress_bar) as pbar:

        losses = []
        lowest_loss = torch.inf
        lowest_epoch = -1
        try:
            for epoch in pbar:
                current_state = {
                    k: v.detach().clone() for k, v in fx_model.named_parameters()
                }
                loss, raw_losses = closure()
                if loss < lowest_loss:
                    fx_model_copy.load_state_dict(current_state, False)
                    lowest_loss = loss
                    lowest_epoch = epoch - 1

                loss_dict = {f"loss_{i}": raw_losses[i] for i in range(len(raw_losses))}

                pbar.set_postfix(
                    lowest_loss=lowest_loss,
                    lowest_epoch=lowest_epoch,
                    loss=loss,
                    **loss_dict,
                )
                losses.append(loss)
        except (
            torch.OutOfMemoryError,
            KeyboardInterrupt,
            AssertionError,
            RuntimeError,
        ) as e:
            traceback.print_exc()
            terminate_condition = str(e)
            # break

    state_dict = {
        "global_step": len(losses),
        "optimiser": optimiser.state_dict(),
        "model": fx_model.state_dict(),
        "best_model": fx_model_copy.state_dict(),
        "lowest_loss": lowest_loss,
    }

    return losses, fx_model_copy, state_dict, terminate_condition


@hydra.main(config_path="cfg", config_name="config")
def train(cfg: DictConfig):
    sr, chunk_dur = cfg.sr, cfg.chunk_duration
    chunk_size = int(sr * chunk_dur)

    chunk_overlap = cfg.chunk_overlap
    overlap_size = int(sr * chunk_overlap)

    if hasattr(cfg, "dataset"):
        match cfg.dataset:
            case "internal_vocal":
                iterator = internal_vocal
            case "medley_vocal":
                iterator = medley_vocal
            case _:
                raise ValueError(cfg.dataset)
    else:
        iterator = medley_vocal

    for dry_file, wet_file, sr, raw_x, raw_y, shifts in iterator(
        cfg.data_dir, loudness=cfg.lufs
    ):
        print(wet_file.stem)

        x = raw_x.unfold(-1, chunk_size, chunk_size - overlap_size).transpose(0, 1)
        y = raw_y.unfold(-1, chunk_size, chunk_size - overlap_size).transpose(0, 1)

        print(f"Number of chunks: {x.size(0)}")

        # filter chunks with very low energy
        threshold = -60
        energies = amp2db(y[..., overlap_size:].abs().amax((1, 2)))
        mask = energies > threshold
        x = x[mask]
        y = y[mask]

        print(f"Dropped {(~mask).count_nonzero()} chunks")

        losses, fx, state_dict, termination = run(x, y, sr, overlap_size, cfg)
        fx = fx.cpu()

        if termination is not None:
            print(termination)

        # remove parameterisation
        remove_fx_parametrisation(fx)

        if cfg.log_dir is not None:
            log_dir = Path(cfg.log_dir)
            song_dir = log_dir / wet_file.stem
            song_dir.mkdir(parents=True, exist_ok=True)

            run_num = 0
            exists_runs = list(song_dir.glob("run_*/"))
            if len(exists_runs):
                run_num = max([int(x.stem.split("_")[1]) for x in exists_runs]) + 1

            log_run_dir = song_dir / f"run_{run_num}/"
            log_run_dir.mkdir()

            with open(log_run_dir / "config.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

            meta_info = {
                "input_path": str(dry_file),
                "target_path": str(wet_file),
                "alignment_shift": shifts,
                "losses": losses,
                "terminated_by": termination,
            }

            with open(log_run_dir / "meta.json", "w") as f:
                json.dump(meta_info, f)

            torch.save(state_dict, log_run_dir / "checkpoint.ckpt")
            torch.save(fx.state_dict(), log_run_dir / "parametrised.pth")

        # print("Rendering...")
        # fx.eval()

        # with torch.no_grad():
        #     pred = fx(raw_x.unsqueeze(0)).squeeze()

        # torchaudio.save("temp.wav", pred, sr)

        # print(fx)


if __name__ == "__main__":
    train()
