import os
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import PSMDataModule
from lightning.pytorch.utilities import move_data_to_device
from metrics import confusion_matrix_figure
from net import Net
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics._plot.precision_recall_curve import PrecisionRecallDisplay
from torchmetrics.utilities.compute import auc
from tqdm import tqdm

colours = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
plt.rcParams.update({"font.size": 20})


def roc_figure(fprs, tprs, areas, labels=None):
    len_values = len(fprs)
    figure = plt.figure(figsize=(12, 12))
    lw = 2
    for i, colour in enumerate(colours[:len_values]):
        if labels:
            label = labels[i]
        else:
            label = "Fold " + str(i + 1)
        tpr = gaussian_filter1d(tprs[i], sigma=5)
        plt.plot(
            fprs[i],
            tpr,
            colour,
            lw=lw,
            label="%s (area = %0.2f)" % (label, areas[i]),
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristics Curve")
    plt.legend(loc="lower right")
    return figure


def pr_figure(precisions, recalls, areas, labels=None):
    len_values = len(precisions)
    figure, ax = plt.subplots(figsize=(12, 12))

    f_scores = np.linspace(0.2, 0.8, num=4)
    _, f_labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    for i, colour in enumerate(colours[:len_values]):
        if labels:
            label = labels[i]
        else:
            label = "Fold " + str(i + 1)
        display = PrecisionRecallDisplay(recall=recalls[i], precision=precisions[i])
        display.plot(ax=ax, name="%s (area = %0.2f)" % (label, areas[i]), color=colour)

    # add the legend for the iso-f1 curves
    handles, f_labels = display.ax_.get_legend_handles_labels()  # type: ignore
    handles.extend([l])    # type: ignore
    f_labels.extend(["Iso-F1 curves"])
    # set the legend and the axes
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(handles=handles, labels=f_labels, loc="best")
    ax.set_title("Precision-Recall Curve")
    return figure


def make_figure(key, values, labels=None):
    if key[2:] == "BinaryConfusionMatrix":
        cm = [[0, 0], [0, 0]]
        for value in values:
            cm[0][0] += value[0][0].numpy()
            cm[0][1] += value[0][1].numpy()
            cm[1][0] += value[1][0].numpy()
            cm[1][1] += value[1][1].numpy()
        return confusion_matrix_figure(np.array(cm), ["False", "True"])
    elif key[2:] == "BinaryROC":
        areas = [auc(value[0], value[1], reorder=True).detach().cpu().numpy() for value in values]
        return roc_figure(
            [value[0].numpy() for value in values],
            [value[1].numpy() for value in values],
            areas,
            labels,
        )
    elif key[2:] == "BinaryPrecisionRecallCurve":
        areas = [auc(value[0], value[1], reorder=True) for value in values]
        return pr_figure(
            [value[0].numpy() for value in values],
            [value[1].numpy() for value in values],
            areas,
            labels,
        )
    else:
        return plt.figure(figsize=(12, 12))


def validate(net, datamodule):
    print("Running model " + datamodule.train_ds.fold + " on its validation set")
    metrics = move_data_to_device(net.valid_metrics.compute(), "cpu")
    figure_metrics = move_data_to_device(net.valid_figure_metrics.compute(), "cpu")
    return metrics, figure_metrics


def predict(nets, data, meta):
    y_preds = []
    for i, net in enumerate(nets):
        y_pred = torch.sigmoid(net(data["feature"]))
        y_preds.append(y_pred)
    return torch.stack(y_preds).sum(dim=0) / len(nets)


def test(nets, datamodule):
    print("Running models on the test set")
    test_dl = datamodule.test_dataloader()
    device = nets[0].device
    for idx, batch in tqdm(enumerate(test_dl)):
        data, meta = move_data_to_device(batch, device)
        y_preds = predict(nets, data, meta)
        y_trues = data["label"]
        nets[0].test_metrics.update(y_preds, y_trues.int())
        nets[0].test_figure_metrics.update(y_preds, y_trues.int())
    metrics = move_data_to_device(nets[0].test_metrics.compute(), "cpu")
    figure_metrics = move_data_to_device(nets[0].test_figure_metrics.compute(), "cpu")
    return [metrics], [figure_metrics]


def get_best_ckpt(folder):
    tmp = [el[:-5].split("-") for el in sorted(os.listdir(folder))]
    tmp = sorted(
        tmp,
        key=lambda x: (
            float(x[1].split("=")[1]),
            float(x[2].split("=")[1]),
            float(x[3].split("=")[1]),
        ),
        reverse=True,
    )
    return os.path.join(folder, "-".join(tmp[0]) + ".ckpt")


def load_nets_frozen(hparams, validate_one=False):
    nets = []
    test = False
    for i in range(10):
        print("Loading model for fold " + str(i))
        ckpt = get_best_ckpt(os.path.join(hparams.ckpt_dir, "fold_" + str(i), "checkpoints"))
        if i == 0:
            test = True
        else:
            if validate_one:
                hparams.validate = False
            test = False
        net = Net.load_from_checkpoint(
            ckpt,
            data_dir=hparams.data_dir,
            run_tests=(not hparams.validate and test),
            load_train_ds=hparams.validate,
            input_size=10000,
        )
        nets.append(net)
        nets[i].freeze()
        nets[i].eval()
        print()
    return nets


def print_metrics(metric):
    print("-------------------------")
    for k, v in metric.items():
        if type(v) == list:
            print(k + ":" + str((sum(v) / len(v))))
        else:
            print(k + ": " + str(v.item()))
    print("-------------------------")


def main(hparams):
    print(hparams)
    nets = load_nets_frozen(hparams)

    if hparams.validate:
        metrics = []
        figure_metrics = []
        for i, net in enumerate(nets):
            datamodule = PSMDataModule(net.hparams)
            metric, figure_metric = validate(net, datamodule)
            metrics.append(metric)
            figure_metrics.append(figure_metric)
            print("Fold " + str(i) + " metrics")
            print_metrics(metric)
    else:
        datamodule = PSMDataModule(nets[0].hparams)
        metrics, figure_metrics = test(nets, datamodule)

    fnl_metrics = defaultdict(list)
    {fnl_metrics[key].append(val) for metric in metrics for key, val in metric.items()}
    fnl_figure_metrics = defaultdict(list)
    {
        fnl_figure_metrics[key].append(val)
        for metric in figure_metrics
        for key, val in metric.items()
    }

    print("Aggregated metrics")
    print_metrics(fnl_metrics)
    if not hparams.validate:
        label = ["Test (Full)"]
    else:
        label = None
    for key, value in fnl_figure_metrics.items():
        make_figure(key, value, label)
    plt.show(block=True)


def parse_arguments():
    parser = ArgumentParser(description="Peptide Spectrum Matching", add_help=True)
    parser.add_argument(
        "--ckpt_dir",
        default="../model",
        type=str,
        help="Checkpoint directory containing checkpoints of all CV folds. Default: %(default)s",
    )
    parser.add_argument(
        "--data-dir",
        default="../data",
        type=str,
        help="Location of data directory. Default: %(default)s",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        help="Run the models on their validation sets. Default: %(default)s",
    )
    parser.set_defaults(validate=False)
    hparams = parser.parse_args()
    hparams.data_dir = os.path.abspath(os.path.expanduser(hparams.data_dir))
    hparams.ckpt_dir = os.path.abspath(os.path.expanduser(hparams.ckpt_dir))
    return hparams


if __name__ == "__main__":
    main(parse_arguments())
