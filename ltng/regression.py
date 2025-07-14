import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Tuple, List, Optional


class ParamPrediction(pl.LightningModule):
    def __init__(
        self,
        predictor: nn.Module,
        condition: str = "wet",
        **kwargs,
    ) -> None:
        super().__init__()

        self.predictor = predictor
        self.condition = condition

    def forward(
        self,
        wet: Optional[torch.Tensor] = None,
        dry: Optional[torch.Tensor] = None,
    ):
        match self.condition:
            case "wet":
                return self.predictor(wet)
            case "dry":
                return self.predictor(dry)
            case "both":
                return self.predictor(wet, dry)
            case _:
                raise ValueError(f"Invalid condition: {self.condition}")

    def training_step(self, batch, batch_idx):
        x, cond, dry, rel_path = batch
        pred = self(cond, dry)

        loss = F.mse_loss(pred, x)

        self.log("loss", loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.tmp_val_outputs = []

    def validation_step(self, batch, batch_idx):
        x, cond, dry, *_ = batch

        pred = self(cond, dry)
        loss = F.mse_loss(pred, x)

        values = {
            "loss": loss.item(),
            "N": x.shape[0],
        }
        self.tmp_val_outputs.append(values)
        return loss

    def on_validation_epoch_end(self) -> None:
        outputs = self.tmp_val_outputs
        weights = [x["N"] for x in outputs]
        avg_loss = np.average([x["loss"] for x in outputs], weights=weights)

        self.log_dict(
            {
                "val_loss": avg_loss,
            },
            prog_bar=True,
            sync_dist=True,
        )

        delattr(self, "tmp_val_outputs")

    def on_test_epoch_start(self) -> None:
        self.tmp_test_outputs = []

    def test_step(self, batch, batch_idx):
        x, cond, dry, *_ = batch

        pred = self(cond, dry)
        loss = F.mse_loss(pred, x)

        values = {
            "loss": loss.item(),
            "N": x.shape[0],
        }
        self.tmp_test_outputs.append(values)
        return loss

    def on_test_epoch_end(self) -> None:
        outputs = self.tmp_test_outputs
        weights = [x["N"] for x in outputs]
        avg_loss = np.average([x["loss"] for x in outputs], weights=weights)

        self.log_dict(
            {
                "test_loss": avg_loss,
            },
            prog_bar=True,
            sync_dist=True,
        )

        delattr(self, "tmp_test_outputs")
