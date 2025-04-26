import io

import torch
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms

from .... import ModelBase
from . import Bottleneck, ResNet


class ResNet50(ResNet, ModelBase):
    """ResNet50 model.

    Parameters
    ----------
    ResNet : models.segmentation.resnet.ResNet
        The ResNet class.
    ModelBase : models.ModelBase
        The model base class.

    Returns
    -------
    Tensor
        The transformed image.
    """

    _transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.unsqueeze(0)),
            transforms.Lambda(lambda x: x.cuda()),
        ]
    )

    def __init__(self, weights: bytes):
        """Initialize the model.

        Parameters
        ----------
        weights : bytes
            Weights of the model.
        """
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        buffer = io.BytesIO(weights)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(buffer, map_location=device)
        self.load_state_dict(state_dict)

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
            The image tensor.
        """
        return cls._transform(image)
