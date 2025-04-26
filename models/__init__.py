from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from PIL.Image import Image
from torch import Tensor, nn

from models.processing import ProcessingModelType
from models.validation import ValidationModelType


class ModelCategory(Enum):
    """Model category enumeration.

    Parameters
    ----------
    Enum : EnumMeta
        Enumeration metaclass.
    """

    VALIDATION = ValidationModelType
    PROCESSING = ProcessingModelType

    @staticmethod
    def validate_model_category_type(
        model_category: ModelCategory,
        model_type: ValidationModelType | ProcessingModelType,
    ):
        """Validate the model category and type.

        Parameters
        ----------
        model_category : ModelCategory
            Model category.
        model_type : ValidationModelType | ProcessingModelType
            Model type.

        Raises
        ------
        ValueError
            If the model category and type are not valid.
        """
        if (
            model_category == ModelCategory.VALIDATION
            and model_type not in ValidationModelType
        ) or (
            model_category == ModelCategory.PROCESSING
            and model_type not in ProcessingModelType
        ):
            raise ValueError(f"Invalid model type: {model_type}")


@dataclass
class Model:
    """Model dataclass."""

    category: ModelCategory
    type: ValidationModelType | ProcessingModelType
    subtype: str
    module: str
    class_name: str


class ModelBase(ABC, nn.Module):
    """Base class for models.

    Parameters
    ----------
    ABC : _abc.ABCMeta
        Abstract base class.
    nn.Module : torch.nn.Module
        PyTorch neural network module.
    """

    @classmethod
    @abstractmethod
    def transform(cls, image: Image) -> Tensor:
        """Transform an image into a tensor.

        Parameters
        ----------
        image : Image
            The image to transform.

        Returns
        -------
        Tensor
            The transformed image.
        """
        pass
