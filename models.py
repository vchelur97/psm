import torch
import math


class BasicBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[3, 3],
        norm_layer=None,
        downsample=None,
    ):
        # Since we need same length output, we can't have
        # downsampling, dilations or strides
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm1d
        if type(kernel_size) is not list or len(kernel_size) != 2:
            raise ValueError("BasicBlock requires a list of length 2 for kernel_size")
        self.conv1 = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[0],
            padding=kernel_size[0] // 2,
            bias=False,
        )
        self.bn1 = norm_layer(out_channels)
        self.relu1 = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[1],
            padding=kernel_size[1] // 2,
            bias=False,
        )
        self.bn2 = norm_layer(out_channels)
        self.relu2 = torch.nn.PReLU()
        self.downsample = downsample

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.relu2(out)
        # Don't use += will cause inplace operation leading to error
        out = out + identity

        return out


class MakeResNet(torch.nn.Module):
    def __init__(self, layers, kernel_size, input_size, hidden_sizes, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.start_planes = hidden_sizes[0]
        self.conv1 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=self.start_planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_layer(self.start_planes)
        self.relu = torch.nn.PReLU()
        self.depth = len(hidden_sizes)
        # self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.layers = torch.nn.ModuleList([])
        for i in range(self.depth):
            self.layers.append(self._make_layer(hidden_sizes[i], layers[i], kernel_size))

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_planes, blocks, kernel_size):
        norm_layer = self._norm_layer
        downsample = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.start_planes,
                out_channels=out_planes,
                kernel_size=1,
                bias=False,
            ),
            norm_layer(out_planes),
        )
        layers = []
        layers.append(
            BasicBlock(self.start_planes, out_planes, kernel_size, norm_layer, downsample)
        )
        self.start_planes = out_planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.start_planes, out_planes, kernel_size, norm_layer))

        return torch.nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        for i in range(self.depth):
            x = self.layers[i](x)

        return x

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--layers",
            nargs="+",
            type=int,
            default=[2, 2],
            help="The number of basic blocks to be used in each layer. Default: %(default)s",
        )
        parser.add_argument(
            "--hidden-sizes",
            nargs="+",
            type=int,
            default=[512, 128],
            help="The size of the 1-D convolutional layers. Default: %(default)s",
        )
        parser.add_argument(
            "--kernel-sizes",
            nargs="+",
            type=int,
            default=[5, 5],
            help="Kernel sizes of the 2 convolutional layers of the basic block. Default: %(default)s",
        )
        return parser


class Detector(torch.nn.Module):
    def __init__(self, hparams, input_size):
        super().__init__()
        self.input_size = input_size
        layers = []
        for unit in hparams.detector_units:
            layers.append(torch.nn.Linear(input_size, unit))
            layers.append(torch.nn.PReLU())
            layers.append(torch.nn.Dropout(hparams.dropout))
            input_size = unit
        self.detector = torch.nn.Sequential(*layers, torch.nn.Linear(input_size, 1))
        # torch.sigmoid will be done later in the loss function

    def forward(self, X, **kwargs):
        # [Batch]
        return self.detector(X).squeeze(dim=1)

    @staticmethod
    def add_class_specific_args(parser):
        parser.add_argument(
            "--detector-units",
            metavar="UNIT",
            nargs="+",
            type=int,
            default=[32],
            help="The number of units in each layer of the detector. Default: %(default)s",
        )
        return parser


class PSMModel(torch.nn.Module):
    def __init__(self, hparams, input_size):
        super().__init__()
        assert len(hparams.layers) == len(hparams.hidden_sizes)
        self.input_size = input_size
        self.resnet_layer = MakeResNet(
            hparams.layers, hparams.kernel_sizes, self.input_size, hparams.hidden_sizes
        )
        self.detector = Detector(hparams, hparams.hidden_sizes[-1])

    def forward(self, X, **kwargs):
        # [Batch, input_size] -> [Batch, hidden_sizes[-1]]
        out = self.resnet_layer(X.unsqueeze(2))
        # [Batch, hidden_sizes[-1]] -> [Batch]
        out = self.detector(out.squeeze(2))
        return out

    @staticmethod
    def add_class_specific_args(parser):
        parser = MakeResNet.add_class_specific_args(parser)
        parser = Detector.add_class_specific_args(parser)
        return parser


# class PSMModel(torch.nn.Module):
#     def __init__(self, hparams, input_size):
#         super().__init__()
#         self.input_size = input_size
#         layers = []
#         for size in hparams.hidden_sizes:
#             layers += [torch.nn.Linear(input_size, size)]
#             layers += [torch.nn.PReLU()]
#             layers += [torch.nn.Dropout(hparams.dropout)]
#             input_size = size
#         layers += [torch.nn.Linear(input_size, 1)]
#         self.model = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x).squeeze(1)

#     @staticmethod
#     def add_class_specific_args(parser):
#         parser.add_argument(
#             "--hidden-sizes",
#             nargs="+",
#             type=int,
#             default=[512, 128, 32],
#             help="The size of the feedforward layers. Default: %(default)s",
#         )
#         return parser


# class PSMModel(torch.nn.Module):
#     def __init__(self, hparams, input_size):
#         super().__init__()
#         self.input_size = input_size
#         self.fc1 = torch.nn.Linear(input_size, 256)
#         self.transformer = torch.nn.TransformerEncoder(
#             torch.nn.TransformerEncoderLayer(
#                 d_model=256,
#                 nhead=8,
#                 dim_feedforward=512,
#                 dropout=hparams.dropout,
#                 activation="tanh",
#             ),
#             num_layers=2,
#         )
#         self.fc2 = torch.nn.Linear(256, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.transformer(x.unsqueeze(0))
#         x = self.fc2(x.squeeze(0))
#         return x.squeeze(1)

#     @staticmethod
#     def add_class_specific_args(parser):
#         parser.add_argument(
#             "--hidden-sizes",
#             nargs="+",
#             type=int,
#             default=[512, 128, 32],
#             help="The size of the 1-D convolutional layers. Default: %(default)s",
#         )
#         return parser
