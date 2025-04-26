from typing import cast

import torch

from .. import ModelWrapper


class Classification(ModelWrapper):
    """Wrapper for a models that receives an image and returns disease and severity classes."""

    def __call__(self, image: bytes, *args, **kwargs) -> tuple[int, int]:
        """Classify an image.

        Parameters
        ----------
        image : bytes
            The image to classify.

        Returns
        -------
        int
            The disease class.
        int
            The severity class.
        """
        image = self.load_image(image)
        out_dis, out_sev = self.model(image)
        out_dis = torch.nn.functional.softmax(out_dis, dim=1)
        _, dis_cls = torch.max(out_dis, 1)
        out_sev = torch.nn.functional.softmax(out_sev, dim=1)
        _, sev_cls = torch.max(out_sev, 1)
        return cast(int, dis_cls.item()), cast(int, sev_cls.item())
