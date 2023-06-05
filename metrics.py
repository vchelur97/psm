import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import get_cmap
from sklearn.metrics import PrecisionRecallDisplay
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.utilities.compute import auc

SMOOTH = 1e-6
plt.rcParams.update({"font.size": 18})


def confusion_matrix_figure(cm, class_names=["False", "True"]):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    cm = cm.astype("int")
    figure = plt.figure(figsize=(8, 8))
    # Normalize the confusion matrix.
    cmap = get_cmap("viridis")
    normalized = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.imshow(normalized, interpolation="nearest", cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use red text if squares are dark; otherwise black.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = cmap(1.0) if normalized[i, j] < 0.5 else cmap(0.0)
        plt.text(
            j,
            i,
            str(normalized[i, j]) + "\n(" + str(cm[i, j]) + ")",
            horizontalalignment="center",
            verticalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def roc_figure(fpr, tpr, area):
    figure = plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % area,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    return figure


def pr_figure(precision, recall, area):
    figure, ax = plt.subplots(figsize=(8, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    _, labels = [], []
    line = None
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (line,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(recall=recall, precision=precision)
    display.plot(ax=ax, name="PR curve (area = %0.2f)" % area, color="gold")

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([line])
    labels.extend(["Iso-F1 curves"])
    # set the legend and the axes
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Precision-Recall curve")

    return figure


def make_figure(key, values):
    if key[2:] == "BinaryConfusionMatrix":
        return confusion_matrix_figure(values.detach().cpu().numpy(), ["False", "True"])
    elif key[2:] == "BinaryROC":
        area = auc(values[0], values[1], reorder=True)
        return roc_figure(values[0].detach().cpu().numpy(), values[1].detach().cpu().numpy(), area)
    elif key[2:] == "BinaryPrecisionRecallCurve":
        area = auc(values[0], values[1], reorder=True)
        return pr_figure(values[0].detach().cpu().numpy(), values[1].detach().cpu().numpy(), area)
    else:
        return plt.figure(figsize=(8, 8))


def weighted_focal_loss(y_pred, y_true, gamma=2.0, pos_weight=[3.0], **kwargs):
    pos_weight = torch.Tensor(pos_weight).to(y_true.device)
    y_pred = torch.clamp(torch.sigmoid(y_pred), SMOOTH, 1.0 - SMOOTH)
    loss = -(pos_weight * y_true * torch.pow(1.0 - y_pred, gamma) * torch.log(y_pred)) - (
        (1 - y_true) * torch.pow(y_pred, gamma) * torch.log(1.0 - y_pred)
    )
    return torch.mean(loss)


def weighted_bce_loss(y_pred, y_true, pos_weight=[3.0], **kwargs):
    pos_weight = torch.Tensor(pos_weight).to(y_true.device)
    return binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=pos_weight, reduction="mean")
