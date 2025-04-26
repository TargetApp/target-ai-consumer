from abc import ABC, abstractmethod

from .. import Interface


class Storage(Interface, ABC):
    """Storage interface.

    Parameters
    ----------
    ABC : _abc.ABCMeta
        Abstract base class.
    """

    @abstractmethod
    def __init__(self, module_settings: dict):
        pass

    @abstractmethod
    def retrieve_weights(self, model_id: int) -> bytes:
        """Retrieve the weights of a model.

        Parameters
        ----------
        model_id : int
            Model ID.

        Returns
        -------
        bytes
            Weights.
        """
        pass

    @abstractmethod
    def store_mask(self, mask: bytes, report_id: int):
        """Store a mask.

        Parameters
        ----------
        mask : bytes
            Mask bytes.
        report_id : int
            Report ID.
        """
        pass

    @abstractmethod
    def store_weights(self, weights: bytes, model_id: int):
        """Store the weights of a model.

        Parameters
        ----------
        weights : bytes
            Weights.
        model_id : int
            Model ID.
        """
        pass

    @abstractmethod
    def store_image(self, image: bytes, image_id: int):
        """Store an image.

        Parameters
        ----------
        image : bytes
            Image bytes.
        image_id : int
            Image ID.
        """
        pass
