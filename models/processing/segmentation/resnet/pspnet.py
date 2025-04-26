import io

import cv2
import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor, nn
from torch.nn import functional as F

from .... import ModelBase
from . import Bottleneck, ResNet


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes]
        )
        self.bottleneck = nn.Conv2d(
            features * (len(sizes) + 1), out_features, kernel_size=1
        )
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(input=stage(feats), size=(h, w), mode="bilinear")
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode="bilinear")
        return self.conv(p)


class PSPNet(ModelBase):
    """PSPNet model.

    Parameters
    ----------
    ModelBase : models.ModelBase
        The model base class.

    Returns
    -------
    Tensor
        The transformed image.
    """

    def __init__(
        self,
        weights: bytes,
    ):
        """Initialize the model.

        Parameters
        ----------
        weights : bytes
            Weights of the model.
        """
        n_classes = 3
        sizes = (1, 2, 3, 6)
        psp_size = 2048
        deep_features_size = 1024
        super().__init__()
        self.feats = ResNet(Bottleneck, [3, 4, 6, 3])
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)
        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1), nn.LogSoftmax()
        )
        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256), nn.ReLU(), nn.Linear(256, n_classes)
        )
        buffer = io.BytesIO(weights)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(buffer, map_location=device)
        self.load_state_dict(state_dict)

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)
        p = self.up_1(p)
        p = self.drop_2(p)
        p = self.up_2(p)
        p = self.drop_2(p)
        p = self.up_3(p)
        p = self.drop_2(p)
        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(
            -1, class_f.size(1)
        )
        return self.final(p), self.classifier(auxiliary)

    @classmethod
    def transform(cls, image: Image) -> Tensor:
        """Transform an image array to a tensor to be used as input to the model.

        Parameters
        ----------
        image_array : Image
            The image array.

        Returns
        -------
        Tensor
            The image tensor to be used as input to the model.
        """
        image_array = np.array(image, dtype=np.uint8)
        transformed_image = cv2.resize(image_array, (512, 256))
        transformed_image = transformed_image[:, :, ::-1]
        transformed_image = transformed_image.astype(np.float64)
        transformed_image = transformed_image.astype(float) / 255.0
        transformed_image = transformed_image.transpose(2, 0, 1)
        tensor = torch.from_numpy(transformed_image).float()
        tensor = tensor.unsqueeze(0).cuda()
        return tensor
