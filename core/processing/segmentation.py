from typing import cast

import numpy as np
import torch
from PIL import Image

from .. import ModelWrapper


class Segmentation(ModelWrapper):
    """Wrapper for a models that receives an image and returns a segmentation mask."""

    def _get_stress_ratio_and_severity(self, output) -> tuple[float, int]:
        """Calculate the stress ratio and severity.

        Parameters
        ----------
        output : torch.Tensor
            The model output.

        Returns
        -------
        tuple[float, int]
            The stress ratio and severity.
        """
        stress_pixels = torch.count_nonzero(
            (output[0, 2, ...] > output[0, 0, ...])
            & (output[0, 2, ...] > output[0, 1, ...])
        )
        leaf_pixels = torch.count_nonzero(
            (output[0, 1, ...] > output[0, 0, ...])
            & (output[0, 1, ...] > output[0, 2, ...])
        )
        stress_ratio: float = cast(
            float, stress_pixels.float() / (leaf_pixels + stress_pixels).float()
        )
        severity = self._ratio_to_severity(stress_ratio)
        return float(stress_ratio), severity

    @staticmethod
    def _ratio_to_severity(stress_ratio: float) -> int:
        """Convert stress ratio to severity.

        Parameters
        ----------
        stress_ratio : float
            The stress ratio.

        Returns
        -------
        int
            The severity.
        """
        if stress_ratio < 0.001:
            return 0
        elif 0.001 <= stress_ratio <= 0.05:
            return 1
        elif 0.05 < stress_ratio <= 0.1:
            return 2
        elif 0.1 < stress_ratio <= 0.15:
            return 3
        elif stress_ratio > 0.15:
            return 4
        return -1

    @staticmethod
    def _generate_mask(output) -> bytes:
        """Generate a mask from the model output.

        Parameters
        ----------
        output : torch.Tensor
            The model output.

        Returns
        -------
        bytes
            The mask.
        """
        output = output.cpu().numpy()
        output = np.squeeze(output, axis=0)
        output = np.transpose(output, (1, 2, 0))
        output = output.astype(np.uint8)
        output = output[..., ::-1]
        output[
            (output[..., 2] > output[..., 0]) & (output[..., 2] > output[..., 1])
        ] = [
            0,
            0,
            0,
        ]
        output[
            (output[..., 0] > output[..., 1]) & (output[..., 0] > output[..., 2])
        ] = [
            255,
            0,
            0,
        ]
        output[
            (output[..., 1] > output[..., 0]) & (output[..., 1] > output[..., 2])
        ] = [
            0,
            255,
            0,
        ]
        for i in range(1, output.shape[0] - 1):
            for j in range(1, output.shape[1] - 1):
                if np.all(output[i, j] == [0, 0, 0]):
                    continue
                elif np.all(output[i, j] == [255, 0, 0]):
                    continue
                elif np.all(output[i, j] == [0, 255, 0]):
                    continue
                neighborhood = output[i - 1 : i + 2, j - 1 : j + 2]  # noqa
                unique, counts = np.unique(
                    neighborhood.reshape(-1, 3), axis=0, return_counts=True
                )
                predominant_pixel = unique[np.argmax(counts)]
                output[i, j] = predominant_pixel
        return Image.fromarray(output).tobytes()

    def __call__(
        self, image: bytes, generate_mask=False
    ) -> tuple[float, int, bytes | None]:
        """Process an image.

        Parameters
        ----------
        image : bytes
            Image bytes.
        generate_mask : bool, optional
            Indicates if a mask should be generated, by default False.

        Returns
        -------
        tuple[float, int, bytes | None]
            The stress ratio, severity and mask.
        """
        img = self.load_image(image)
        with torch.no_grad():
            output, cls = self.model(img)
        output = (output - output.min()) * (255 / (output.max() - output.min()))
        stress_ratio, severity = self._get_stress_ratio_and_severity(output)
        mask = None
        if generate_mask:
            mask = self._generate_mask(output)
        return stress_ratio, severity, mask
