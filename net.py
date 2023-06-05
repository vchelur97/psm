from argparse import Namespace
from collections import OrderedDict

import lightning.pytorch as pl
import torch
from metrics import make_figure, weighted_bce_loss, weighted_focal_loss
from models import PSMModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import (
    ROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    JaccardIndex,
    MatthewsCorrCoef,
    MetricCollection,
    Precision,
    PrecisionRecallCurve,
    Recall,
)
from torchmetrics.utilities.compute import auc
from sklearn.metrics import classification_report


class Net(pl.LightningModule):
    def __init__(self, hparams, input_size):
        super().__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        # torch.autograd.set_detect_anomaly(True)
        hparams.input_size = input_size
        self.save_hyperparameters(hparams)

        metrics = MetricCollection(
            [
                MatthewsCorrCoef("binary", threshold=hparams.threshold),
                Accuracy("binary", threshold=hparams.threshold),
                F1Score("binary", threshold=hparams.threshold),
                JaccardIndex("binary", threshold=hparams.threshold),
                Precision("binary", threshold=hparams.threshold),
                Recall("binary", threshold=hparams.threshold),
            ]
        )
        self.valid_metrics = metrics.clone(prefix="v_")
        self.test_metrics = metrics.clone(prefix="t_")

        figure_metrics = MetricCollection(
            [
                ConfusionMatrix("binary", threshold=hparams.threshold),
                ROC("binary"),
                PrecisionRecallCurve("binary"),
            ]
        )
        self.valid_figure_metrics = figure_metrics.clone(prefix="v_")
        self.test_figure_metrics = figure_metrics.clone(prefix="t_")

        self.model = PSMModel(hparams, input_size)
        # TODO: Change loss function
        # if hparams.loss == "focal":
        self.loss_func = weighted_focal_loss
        # self.loss_func = weighted_bce_loss
        # else:
        self.outputs = []

    def forward(self, X, **kwargs):
        output = self.model(X, **kwargs)
        return output

    # def on_after_backward(self):
    #     # example to inspect gradient information in tensorboard
    #     if self.trainer.global_step % 200 == 0:  # don't make the tf file huge
    #         params = self.state_dict()
    #         for k, v in params.items():
    #             grads = v
    #             name = k
    #             self.logger.experiment.add_histogram(
    #                 tag=name, values=grads, global_step=self.trainer.global_step
    #             )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)  # type: ignore
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True),
            "monitor": "v_loss",  # TODO: Change monitor
        }
        return [optimizer], [scheduler]

    # learning rate warm-up
    # def optimizer_step(self, curr_epoch, batch_idx, optim, optimizer_closure):
    #     # warm up lr
    #     warm_up_steps = float(70000 // self.hparams.batch_size)  # type: ignore
    #     if self.trainer.global_step < warm_up_steps:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / warm_up_steps)
    #         for pg in optim.param_groups:
    #             pg["lr"] = lr_scale * self.hparams.lr  # type: ignore

    #     optim.step(closure=optimizer_closure)
    def predict_step(self, batch, batch_idx):
        return self(batch["feature"])

    def training_step(self, batch, batch_idx):
        logits = self.predict_step(batch, batch_idx)
        return self.loss_func(logits, batch["label"].float())

    def val_test_step(self, batch, calc_metrics, calc_figure_metrics):
        logits = self.predict_step(batch, 0)
        y_trues = batch["label"].float()
        y_preds = torch.sigmoid(logits)
        calc_metrics.update(y_preds, y_trues.int())
        calc_figure_metrics.update(y_preds, y_trues.int())
        loss = {
            calc_metrics.prefix + "loss": self.loss_func(logits, y_trues),
        }
        # print(classification_report(y_trues.int().cpu().numpy(), y_preds.int().cpu().numpy()))
        self.outputs.append(loss)
        return loss

    def val_test_epoch_end(self, calc_metrics, calc_figure_metrics):
        metrics = OrderedDict(
            {
                key: torch.stack([el[key] for el in self.outputs]).mean()
                for key in self.outputs[0]
                if not key.startswith("f_")
            }
        )
        metrics.update(calc_metrics.compute())
        calc_metrics.reset()
        for key, val in metrics.items():
            self.try_log(key, val, len(self.outputs))

        figure_metrics = OrderedDict(
            {
                key[2:]: torch.stack(sum([el[key] for el in self.outputs], []))
                for key in self.outputs[0]
                if key.startswith("f_")
            }
        )
        figure_metrics.update(calc_figure_metrics.compute())
        calc_figure_metrics.reset()
        for key, val in figure_metrics.items():
            self.logger.experiment.add_figure(key, make_figure(key, val), self.current_epoch)  # type: ignore
            if key[2:] == "BinaryROC":
                self.try_log("v_auroc", auc(val[0], val[1], reorder=True), len(self.outputs))
            if key[2:] == "BinaryPrecisionRecallCurve":
                self.try_log("v_auprc", auc(val[0], val[1], reorder=True), len(self.outputs))
        self.outputs.clear()

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, self.valid_metrics, self.valid_figure_metrics)

    def on_validation_epoch_end(self):
        self.val_test_epoch_end(self.valid_metrics, self.valid_figure_metrics)

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, self.test_metrics, self.test_figure_metrics)

    def on_test_epoch_end(self):
        self.val_test_epoch_end(self.test_metrics, self.test_figure_metrics)

    def try_log(self, key, value, batch_size):
        try:
            self.log(key, value, prog_bar=True, batch_size=batch_size)
        except Exception as e:
            print(e)
            return

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--lr",
            default=0.01,
            type=float,
            help="Main Net Learning Rate. Default: %(default)f",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout to be applied between layers. Default: %(default)f",
        )
        parser.add_argument(
            "--loss",
            type=str,
            default="bce",
            choices=["bce", "focal"],
            help="Loss function to use. Default: %(default)s",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Threshold to use for binary classification. Default: %(default)f",
        )
        return parser
