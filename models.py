import torch
import math


class PSMModel(torch.nn.Module):
    def __init__(self, hparams, input_size):
        super().__init__()
        self.input_size = input_size
        layers = []
        for size in hparams.hidden_sizes:
            layers += [torch.nn.Linear(input_size, size)]
            layers += [torch.nn.ReLU()]
            layers += [torch.nn.Dropout(hparams.dropout)]
            input_size = size
        layers += [torch.nn.Linear(input_size, 1)]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(1)

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--hidden-sizes",
            nargs="+",
            type=int,
            default=[1024, 128],
            help="The size of the 1-D convolutional layers. Default: %(default)s",
        )
        return parser
