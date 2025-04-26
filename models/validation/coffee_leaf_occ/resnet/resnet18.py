import io

import torch
import torchvision.transforms as transforms
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock

from .... import ModelBase
from . import ResNet


class ResNet18(ResNet, ModelBase):
    """ResNet-18 model for coffee leaf one-class classification.

    Parameters
    ----------
    ResNet : models.coffee_leaf_occ.resnet.ResNet
        ResNet model.
    ModelBase : models.ModelBase
        Base class for models.

    Returns
    -------
    Tensor
        The transformed image.
    """

    _transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
            transforms.Lambda(lambda x: x.cuda()),
        ]
    )

    def __init__(self, weights: bytes):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        num_features = self.fc.in_features  # type: ignore
        self.fc = nn.Linear(num_features, 2)
        buffer = io.BytesIO(weights)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(buffer, map_location=device)
        self.load_state_dict(state_dict)

    @classmethod
    def transform(cls, image) -> Tensor:
        return cls._transform(image)
