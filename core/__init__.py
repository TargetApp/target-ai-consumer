import io
from abc import ABC, abstractmethod
from importlib import import_module

import torch
from PIL import Image

from models import Model, ModelBase, ModelCategory


class ModelWrapper(ABC):
    """Abstract class for model wrappers."""

    def __init__(
        self,
        model: Model,
        weights: bytes,
    ):
        """Initialize the model.

        Parameters
        ----------
        model : Model
            The model dataclass instance.
        weights : bytes
            Weights of the model.
        """
        self._model = self._get_model(model, weights)
        if torch.cuda.is_available():
            self._model.cuda()
            self._device_tensor = lambda x: x.cuda()
        else:
            self._device_tensor = lambda x: x
        self._model.eval()

    def _get_model(self, model: Model, weights: bytes) -> ModelBase:
        """Get the model.

        Parameters
        ----------
        model : Model
            The model dataclass instance.
        weights : bytes
            The model weights.

        Returns
        -------
        ModelBase
            The model.
        """
        ModelCategory.validate_model_category_type(model.category, model.type)
        module = import_module(
            f"models.{model.category.name.lower()}.{model.type.name.lower()}.{model.subtype}.{model.module}"
        )
        return getattr(module, model.class_name)(weights)

    @property
    def model(self):
        return self._model

    def load_image(self, image: bytes):
        """Load an image from bytes.

        Parameters
        ----------
        image : bytes
            The image in bytes.

        Returns
        -------
        torch.Tensor
            The image as a tensor.
        """
        pil_img = Image.open(io.BytesIO(image))
        if pil_img.mode == "RGBA":
            pil_img = pil_img.convert("RGB")
        tensor = self._model.transform(pil_img)
        return self._device_tensor(tensor)

    @abstractmethod
    def __call__(self, image: bytes, *args, **kwargs) -> tuple:
        """Evaluate an image.

        Parameters
        ----------
        image : bytes
            The image in bytes.


        Returns
        -------
        tuple
            The evaluation result.
        """
        pass
